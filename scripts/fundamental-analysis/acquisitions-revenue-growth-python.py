# Full write-up: https://xfinlink.com/blog/acquisitions-revenue-growth-python
"""How much revenue does a dollar of acquisitions buy?

Measures the association between a decade of cash spent on acquisitions and
the revenue a company ended the decade with, across a point-in-time S&P 500
universe.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST, LAST = 2014, 2024
MIN_YEARS = 9
MIN_REVENUE = 500.0  # $m, keeps the ratio's denominator meaningful

# 1. Every company that sat in the index at any year end over the window,
#    addressed by entity id so a rename does not split a history.
ids = set()
for y in range(FIRST, LAST + 1):
    r = xfl.index("sp500", as_of="%d-12-31" % y)
    ids.update(int(i) for i in r["entity_id"].dropna())
ids = sorted(ids)

frames = []
for i in range(0, len(ids), 200):
    frames.append(xfl.fundamentals(
        entity_id=ids[i:i + 200], start="2013-06-30", end="2025-12-31",
        period_type="annual", fields=["revenue", "acquisitions_net"],
        max_rows=60000))
fun = pd.concat(frames, ignore_index=True)
fun["period_end"] = pd.to_datetime(fun["period_end"])
# The fiscal_year label collides on a few filers, so derive the year from
# period_end, which is correct on every row.
fun["year"] = fun["period_end"].dt.year - (fun["period_end"].dt.month <= 6).astype(int)
fun = fun[(fun["year"] >= FIRST) & (fun["year"] <= LAST)]
fun = fun.sort_values(["entity_id", "period_end"]).drop_duplicates(
    ["entity_id", "year"], keep="last")

rows = []
for eid, g in fun.groupby("entity_id"):
    g = g.sort_values("year")
    if len(g) < MIN_YEARS:
        continue
    first, last = g.iloc[0], g.iloc[-1]
    if last["year"] - first["year"] < MIN_YEARS - 1:
        continue
    r0, r1 = first["revenue"], last["revenue"]
    if pd.isna(r0) or pd.isna(r1) or r0 < MIN_REVENUE or r1 <= 0:
        continue
    spend = float(g["acquisitions_net"].clip(lower=0).sum())
    rows.append({"ticker": last["ticker"], "name": last["entity_name"],
                 "sector": last["gics_sector"], "years": len(g),
                 "rev0": r0, "rev1": r1, "spend": spend,
                 "intensity": spend / r0,
                 "rev_growth": (r1 - r0) / r0,
                 "cagr": (r1 / r0) ** (1 / (last["year"] - first["year"])) - 1})

res = pd.DataFrame(rows).dropna(subset=["sector"])

print("Point-in-time S&P 500 rosters at each year end %d-%d" % (FIRST, LAST))
print("companies with >=%d annual periods and revenue above $%.0fm at the start: %d"
      % (MIN_YEARS, MIN_REVENUE, len(res)))
print("total acquisition spending in the sample: $%.2f trillion"
      % (res["spend"].sum() / 1e6))
print("companies spending nothing on acquisitions in the decade: %d"
      % (res["spend"] == 0).sum())
print()

# Slope through the origin: a dollar of acquisitions against a dollar of
# added revenue, both scaled by starting revenue so large and small
# companies enter on the same footing.
x, y = res["intensity"].values, res["rev_growth"].values
slope = (x * y).sum() / (x * x).sum()
r = np.corrcoef(x, y)[0, 1]
pooled = (res["rev1"] - res["rev0"]).sum() / res["spend"].sum()
print("added revenue against acquisition spending, both as a multiple of starting revenue")
print("  slope through the origin  %.3f" % slope)
print("  correlation               %.3f" % r)
print("  pooled: $%.2ftn of extra annual revenue against $%.2ftn spent = %.0f cents"
      " of annual revenue per dollar"
      % ((res["rev1"] - res["rev0"]).sum() / 1e6, res["spend"].sum() / 1e6, 100 * pooled))
print()

res["q"] = pd.qcut(res["intensity"].rank(method="first"), 5,
                   labels=["Q1 least acquisitive", "Q2", "Q3", "Q4", "Q5 most acquisitive"])
tab = res.groupby("q", observed=True).agg(
    n=("cagr", "size"), spend=("intensity", "median"), cagr=("cagr", "median"),
    growth=("rev_growth", "median"))
tab["spend"] = (100 * tab["spend"]).round(1)
tab["cagr"] = (100 * tab["cagr"]).round(2)
tab["growth"] = (100 * tab["growth"]).round(1)
print("quintiles of acquisition spending as a share of starting revenue")
print(tab.to_string())
print()

print("by sector")
s = res.groupby("sector").agg(n=("cagr", "size"), intensity=("intensity", "median"),
                              cagr=("cagr", "median"))
s["intensity"] = (100 * s["intensity"]).round(1)
s["cagr"] = (100 * s["cagr"]).round(2)
print(s.sort_values("intensity", ascending=False).to_string())
print()

print("the ten heaviest acquirers, spending against starting revenue")
big = res.nlargest(10, "intensity")[["ticker", "name", "rev0", "rev1", "spend", "intensity", "cagr"]]
big["intensity"] = big["intensity"].round(2)
big["cagr"] = (100 * big["cagr"]).round(2)
print(big.round(1).to_string(index=False))
print()
none = res[res["spend"] == 0]
print("companies that bought nothing: median revenue CAGR %.2f%% (n=%d)"
      % (100 * none["cagr"].median(), len(none)))
print("companies in the top quintile: median revenue CAGR %.2f%%"
      % (100 * res[res["q"] == "Q5 most acquisitive"]["cagr"].median()))

# ---- chart --------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))

a1.scatter(res["intensity"], res["rev_growth"], s=14, color="#3b82f6", alpha=0.6,
           edgecolors="none")
xs = np.linspace(0, res["intensity"].quantile(0.99), 50)
a1.plot(xs, slope * xs, color="#f59e0b", lw=2)
a1.set_xlim(0, res["intensity"].quantile(0.99))
a1.set_ylim(-1, res["rev_growth"].quantile(0.98))
a1.set_xlabel("Acquisition spending, as a multiple of starting revenue")
a1.set_ylabel("Revenue growth over the decade")
a1.set_title("Spending buys revenue, at a rate")

x2 = np.arange(len(tab))
a2.bar(x2, tab["cagr"], color="#3b82f6")
a2.set_xticks(x2)
a2.set_xticklabels(["least\nacquisitive", "Q2", "Q3", "Q4", "most\nacquisitive"])
a2.set_ylabel("Median revenue CAGR (%)")
a2.set_title("Growth by acquisition intensity")
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("acquisitions-revenue-growth-python.png", dpi=150, facecolor="#0a0a0a")
