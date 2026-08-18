# Full write-up: https://xfinlink.com/blog/high-dividend-yield-cut-risk-python
"""Does a high dividend yield warn that the dividend itself is about to be cut?

The yield is measured at the end of formation year Y from cash that actually
reached shareholders during Y. The outcome is measured over Y+1 and Y+2, so the
signal is always fixed before the event it is asked to predict. Index
membership is point in time: the roster for year Y is the roster as it stood
that December, which keeps companies that later left the index in the sample.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2013, 2024))
CUT = 0.90  # a cut is a fall of more than 10 percent in the regular dividend

rosters = {y: xfl.index("sp500", as_of=f"{y}-12-31") for y in YEARS}
ids = sorted({int(i) for r in rosters.values() for i in r["entity_id"].dropna()})

px = pd.concat([xfl.prices(entity_id=ids[i:i + 40], start="2011-01-01", end="2026-08-01",
                           interval="1mo", fields=["close", "adj_close", "dividend"],
                           max_rows=200000)
                for i in range(0, len(ids), 40)], ignore_index=True)
fun = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 50], period_type="annual",
                                  start="2012-01-01", end="2026-12-31",
                                  fields=["dividends_paid_common", "net_income"],
                                  max_rows=20000)
                 for i in range(0, len(ids), 50)], ignore_index=True)

px["date"] = pd.to_datetime(px["date"])
px = px[(px["close"] > 0) & (px["adj_close"] > 0)].sort_values(["entity_id", "date"])
px["year"] = px["date"].dt.year
# close is the raw as-traded price and adj_close is split-adjusted, so their ratio
# is the split factor still ahead of that month. Scaling the cash dividend by it
# puts every year of payments on one share basis, splits and reverse splits alike.
px["div_adj"] = px["dividend"].fillna(0.0) * px["adj_close"] / px["close"]
px["step"] = px.groupby("entity_id")["adj_close"].pct_change()

def regular(payments):
    """Average regular payment: drop anything above 1.5x the year's median."""
    v = payments.values
    return v[v <= 1.5 * np.median(v)].mean()

paid = px[px["div_adj"] > 0]
rate = paid.groupby(["entity_id", "year"])["div_adj"].agg(rate=regular, k="size").reset_index()
freq = rate.groupby("entity_id")["k"].agg(lambda s: s.mode().iat[0])

year = px.groupby(["entity_id", "year"]).agg(
    price=("adj_close", "last"), months=("adj_close", "size"),
    jump=("step", lambda s: bool(((s < -0.7) | (s > 1.5)).any()))).reset_index()
year = year[year["months"] >= 10].merge(rate, on=["entity_id", "year"], how="left")
year["rate"] = year["rate"].fillna(0.0)
R = year.pivot(index="year", columns="entity_id", values="rate")
P = year.pivot(index="year", columns="entity_id", values="price")
J = year.pivot(index="year", columns="entity_id", values="jump")

fun["period_end"] = pd.to_datetime(fun["period_end"])
fun["cy"] = fun["period_end"].dt.year - (fun["period_end"].dt.month <= 6).astype(int)
fun = fun.sort_values("period_end").drop_duplicates(["entity_id", "cy"], keep="last")
fun["payout"] = np.where(fun["net_income"] > 0,
                         fun["dividends_paid_common"] / fun["net_income"], np.nan)
PAY = fun.pivot(index="cy", columns="entity_id", values="payout")

rows = []
for y in YEARS:
    members = set(rosters[y]["entity_id"].dropna().astype(int))
    fr = pd.DataFrame({"prior": R.loc[y - 1], "rate": R.loc[y],
                       "next1": R.loc[y + 1], "next2": R.loc[y + 2],
                       "price": P.loc[y], "price2": P.loc[y + 2], "payout": PAY.loc[y],
                       "jump": J.loc[y - 1:y + 2].fillna(False).astype(bool).any()})
    fr = fr[fr.index.isin(members)]
    fr["jump"] = fr["jump"].fillna(False).astype(bool)
    fr = fr[(fr["rate"] > 0) & (fr["prior"] > 0) & ~fr["jump"]
            & fr["price"].notna() & fr["price2"].notna()
            & fr["next1"].notna() & fr["next2"].notna()]
    fr["yield"] = fr["rate"] * freq.reindex(fr.index) / fr["price"]
    fr["cut"] = (fr[["next1", "next2"]].min(axis=1) < CUT * fr["rate"]).astype(int)
    fr["growth"] = fr["next2"] / fr["rate"] - 1
    fr["year"] = y
    rows.append(fr.reset_index())

d = pd.concat(rows, ignore_index=True)
d = d[(d["yield"] > 0.001) & (d["yield"] < 0.30)]
d["q"] = d.groupby("year")["yield"].transform(lambda s: pd.qcut(s, 5, labels=False) + 1)

tab = d.groupby("q").agg(n=("cut", "size"), yld=("yield", "median"), cut=("cut", "mean"),
                         growth=("growth", "median"), payout=("payout", "median"))
by_year = d.pivot_table(index="year", columns="q", values="cut", aggfunc="mean")
# A payout ratio does not exist where net income is zero or negative, or where no
# annual period maps to that year. Those company-years are their own bucket, so the
# four buckets account for every company-year in the quintile.
d["band"] = pd.cut(d["payout"], [-np.inf, 0.6, 1.0, np.inf],
                   labels=["under 60%", "60 to 100%", "over 100%"])
d["band"] = d["band"].cat.add_categories("no ratio").fillna("no ratio")
grid = pd.crosstab(d["q"], d["band"], values=d["cut"], aggfunc="mean")
counts = pd.crosstab(d["q"], d["band"])

top, bottom = d[d["q"] == 5], d[d["q"] == 1]
hits = np.array([top["cut"].sum(), bottom["cut"].sum()])
obs = np.array([len(top), len(bottom)])
pooled = hits.sum() / obs.sum()
z = (hits[0] / obs[0] - hits[1] / obs[1]) / np.sqrt(
    pooled * (1 - pooled) * (1 / obs[0] + 1 / obs[1]))

print(f"formation years {min(YEARS)}-{max(YEARS)}    company-years {len(d)}    "
      f"companies {d['entity_id'].nunique()}")
print(f"dividend cut within two years, whole sample: {d['cut'].mean():.1%}\n")
print("Quintile  median yield  cut within 2y  median 2y dividend growth  median payout")
for q, r in tab.iterrows():
    print(f"   Q{q}        {r['yld']:6.2%}        {r['cut']:6.1%}              "
          f"{r['growth']:+7.1%}                  {r['payout']:6.1%}")
print(f"\nQ5 minus Q1: {(tab.loc[5, 'cut'] - tab.loc[1, 'cut']) * 100:+.1f} points   "
      f"z = {z:.2f}, p = {2 * stats.norm.sf(abs(z)):.1e}")
print(f"Q5 cut rate exceeds Q1 in {int((by_year[5] > by_year[1]).sum())} "
      f"of {len(by_year)} formation years")
fifth = d[d["q"] == 5]
has = fifth[fifth["band"] != "no ratio"]
none = fifth[fifth["band"] == "no ratio"]
print(f"Top quintile: {has['cut'].mean():.1%} cut where a payout ratio exists "
      f"({len(has)}), {none['cut'].mean():.1%} where it does not ({len(none)})")
print("\nCut rate by yield quintile and payout ratio (company-years in brackets)")
print('"no ratio" = net income zero or negative, or no annual period matched to that year')
print(" " * 12 + "".join(f"{c:>11}      " for c in grid.columns).rstrip())
for q in grid.index:
    print(f"      Q{q}    " + "".join(
        f"{grid.loc[q, c]:>11.1%} ({counts.loc[q, c]:>3})" for c in grid.columns))
print(" " * 8 + "all" + "".join(
    f"{d.groupby('band')['cut'].mean()[c]:>11.1%} ({counts[c].sum():>3})" for c in grid.columns))

# ── chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))

bars = a1.bar(range(5), tab["cut"] * 100, color=["#3b82f6"] * 4 + ["#ef4444"], width=0.62)
for b, v in zip(bars, tab["cut"]):
    a1.text(b.get_x() + b.get_width() / 2, v * 100 + 0.4, f"{v:.1%}",
            ha="center", color="#e0e0e0", fontsize=10)
a1.set_xticks(range(5))
a1.set_xticklabels([f"Q{q}\n{r['yld']:.1%}" for q, r in tab.iterrows()])
a1.set_xlabel("Dividend yield quintile, median yield below")
a1.set_ylabel("Dividend cut within two years (%)")
a1.set_title("Cut risk sits in the top fifth alone", color="#fafafa", fontsize=11)
a1.set_ylim(0, tab["cut"].max() * 122)

shades = {"under 60%": "#93c5fd", "60 to 100%": "#3b82f6",
          "over 100%": "#ef4444", "no ratio": "#f59e0b"}
for k, band in enumerate(grid.columns):
    label = "no payout ratio" if band == "no ratio" else f"payout {band}"
    a2.bar(np.arange(5) + (k - 1.5) * 0.21, grid[band] * 100, width=0.21,
           color=shades[band], label=label)
a2.set_xticks(range(5))
a2.set_xticklabels([f"Q{q}" for q in grid.index])
a2.set_xlabel("Dividend yield quintile")
a2.set_ylabel("Dividend cut within two years (%)")
a2.set_title("A missing payout ratio beats a large one", color="#fafafa", fontsize=11)
a2.legend(frameon=False, fontsize=9, labelcolor="#e0e0e0")
for ax in (a1, a2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("high-dividend-yield-cut-risk-python.png", dpi=150, facecolor="#0a0a0a")
