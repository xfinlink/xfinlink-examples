# Full write-up: https://xfinlink.com/blog/how-much-of-stock-return-is-sector-python
"""Decompose S&P 500 monthly stock returns into market, sector and stock-specific parts.

Point-in-time membership, so the cross-section in any month is the index as it
actually stood, not today's survivors. Built from SEC EDGAR public filings and
market data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2015-12-01", "2026-07-31"
YEARS = range(2016, 2027)
WINSOR = 0.01
CHART = "how-much-of-stock-return-is-sector-python.png"

# ── 1. point-in-time rosters: membership as of each prior year end ───────────
rosters = {y: xfl.index("sp500", as_of=f"{y - 1}-12-31") for y in YEARS}
members = {y: set(r["entity_id"].dropna().astype(int)) for y, r in rosters.items()}
ids = sorted(set().union(*members.values()))

# ── 2. monthly prices, keyed on entity id so a symbol change keeps one series ─
CHUNK = 100
px = pd.concat(
    [
        xfl.prices(
            entity_id=ids[i : i + CHUNK],
            start=START,
            end=END,
            interval="1mo",
            fields=["date", "close", "adj_close", "gics_sector"],
            max_rows=50000,
        )
        for i in range(0, len(ids), CHUNK)
    ],
    ignore_index=True,
)
px["month"] = px["date"].dt.to_period("M")
sector = px.dropna(subset=["gics_sector"]).groupby("entity_id")["gics_sector"].last()

panel = px.pivot_table(index="month", columns="entity_id", values="adj_close")
ret = (panel / panel.shift(1) - 1).loc[pd.Period("2016-01") :]

# ── 3. one cross-sectional decomposition per month ───────────────────────────
def decompose(winsor):
    out = []
    for m in ret.index:
        r = ret.loc[m].dropna()
        r = r[[i for i in r.index if i in members[m.year] and i in sector.index]]
        if len(r) < 100:
            continue
        if winsor:
            r = r.clip(r.quantile(winsor), r.quantile(1 - winsor))
        s = sector.reindex(r.index)
        mkt = r.mean()                        # equal-weighted market effect
        sec_mean = r.groupby(s).mean()
        sec_dev = s.map(sec_mean) - mkt       # sector effect, net of the market
        idio = r - mkt - sec_dev              # stock-specific residual
        r2 = 1 - (idio ** 2).sum() / ((r - mkt) ** 2).sum()
        out.append(dict(month=m, n=len(r), mkt=mkt, r2=r2,
                        sec_dev=sec_dev, idio=idio, sec_mean=sec_mean))
    return out


def shares(rows):
    m = np.concatenate([np.full(d["n"], d["mkt"]) for d in rows])
    s = np.concatenate([d["sec_dev"].values for d in rows])
    e = np.concatenate([d["idio"].values for d in rows])
    v = (m + s + e).var()
    return dict(market=100 * m.var() / v, sector=100 * s.var() / v,
                stock=100 * e.var() / v, sd=100 * np.sqrt(v), obs=len(m))


rows = decompose(WINSOR)
sh = shares(rows)
alt = shares(decompose(0.005))
monthly = pd.DataFrame([{k: d[k] for k in ("month", "n", "mkt", "r2")} for d in rows])
monthly = monthly.set_index("month")

sec_panel = pd.DataFrame({d["month"]: d["sec_mean"] - d["mkt"] for d in rows}).T

# ── 4. output ────────────────────────────────────────────────────────────────
print(f"{len(ids)} companies held S&P 500 membership at some point, 2016-2026")
print(f"{len(monthly)} months, {sh['obs']:,} company-month returns, "
      f"{monthly['n'].mean():.0f} companies per month on average")
print()

print("share of a company's monthly return variance")
print(f"  market (all stocks together) {sh['market']:5.1f}%")
print(f"  sector (net of the market)   {sh['sector']:5.1f}%")
print(f"  stock-specific               {sh['stock']:5.1f}%")
print(f"  monthly standard deviation of the typical return: {sh['sd']:.2f}%")
print(f"  same three shares at a 0.5% winsor: {alt['market']:.1f}% / "
      f"{alt['sector']:.1f}% / {alt['stock']:.1f}%")
print()

sd_all = sh["sd"]
sd_nomkt = sd_all * np.sqrt((sh["sector"] + sh["stock"]) / 100)
sd_neither = sd_all * np.sqrt(sh["stock"] / 100)
print("monthly standard deviation after hedging")
print(f"  unhedged                {sd_all:5.2f}%")
print(f"  market hedged           {sd_nomkt:5.2f}%  ({100 * (1 - sd_nomkt / sd_all):.1f}% lower)")
print(f"  market + sector hedged  {sd_neither:5.2f}%  ({100 * (1 - sd_neither / sd_nomkt):.1f}% lower again)")
print()

r2 = monthly["r2"] * 100
print("sector share of the within-month cross-section (R-squared of sector dummies)")
print(f"  mean {r2.mean():.1f}%   median {r2.median():.1f}%   "
      f"min {r2.min():.1f}%   max {r2.max():.1f}%")
print("  highest months:")
for m, v in r2.nlargest(5).items():
    print(f"    {m}  {v:5.1f}%   market {monthly.loc[m, 'mkt'] * 100:+6.1f}%")
print("  lowest months:")
for m, v in r2.nsmallest(3).items():
    print(f"    {m}  {v:5.1f}%   market {monthly.loc[m, 'mkt'] * 100:+6.1f}%")
print()

tbl = pd.DataFrame({
    "sd": sec_panel.std() * 100,
    "mean": sec_panel.mean() * 100,
    "best": sec_panel.max() * 100,
    "worst": sec_panel.min() * 100,
}).sort_values("sd", ascending=False)
print("sector effect: monthly deviation of a sector from the market, %")
print(f"{'sector':<24}{'std dev':>9}{'mean':>8}{'best':>8}{'worst':>8}")
for k, v in tbl.iterrows():
    print(f"{k:<24}{v['sd']:9.2f}{v['mean']:8.2f}{v['best']:8.1f}{v['worst']:8.1f}")

# ── 5. chart ─────────────────────────────────────────────────────────────────
BG, FG, AC = "#0a0a0a", "#e0e0e0", "#3b82f6"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#333333", "font.size": 10})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
x = monthly.index.to_timestamp()
ax1.plot(x, r2.values, color=AC, lw=1.1)
ax1.plot(x, r2.rolling(12).mean().values, color="#f59e0b", lw=1.8,
         label="12-month average")
ax1.axhline(r2.mean(), color="#666666", lw=0.9, ls="--",
            label=f"full-period mean {r2.mean():.1f}%")
ax1.set_ylabel("Return spread explained by sector (%)")
ax1.set_title("How much of a stock's monthly return comes from its sector")
ax1.legend(frameon=False, loc="upper right", fontsize=9)
ax1.set_ylim(0, max(r2) * 1.15)

order = tbl.sort_values("sd")
ax2.barh(order.index, order["sd"], color=AC, height=0.65)
ax2.set_xlabel("Monthly swing of the sector away from the market (standard deviation, %)")
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor=BG)
print(f"\nchart saved to {CHART}")
