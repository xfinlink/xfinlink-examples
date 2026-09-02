# Full write-up: https://xfinlink.com/blog/inventory-build-margin-squeeze-python
#
# Does an inventory build predict a margin squeeze?
# Cross-sectional test on the point-in-time S&P 500, signal years 2014-2023.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

OUT_PNG = "inventory-build-margin-squeeze-python.png"
DROP_SECTORS = ["Financials", "Real Estate", "Utilities"]
MAX_MARGIN_MOVE = 20.0   # points, applied in both directions
MIN_INVENTORY = 0.02     # inventory as a share of revenue

# ---------------------------------------------------------------- 1. universe
ids = set()
for y in range(2013, 2025):
    ids.update(int(i) for i in xfl.index("sp500", as_of="%d-12-31" % y)["entity_id"].dropna())

fun = xfl.fundamentals(entity_id=sorted(ids), start="2012-06-30", end="2025-06-30",
                       period_type="annual",
                       fields=["revenue", "cost_of_revenue", "inventory"],
                       max_rows=60000)

# year from the period end, not the fiscal-year label, so 52-week and
# January-ending filers land in the right year
fun["year"] = fun["period_end"].dt.year - (fun["period_end"].dt.month <= 6).astype(int)

# ------------------------------------------------- 2. inventory-carrying names
fun = fun[~fun["gics_sector"].isin(DROP_SECTORS)]
fun = fun.dropna(subset=["revenue", "cost_of_revenue", "inventory", "gics_sector"])
fun = fun[(fun["revenue"] > 0) & (fun["cost_of_revenue"] > 0) & (fun["inventory"] > 0)]
fun = fun.sort_values(["entity_id", "year"]).drop_duplicates(["entity_id", "year"], keep="last")
fun["gm"] = (fun["revenue"] - fun["cost_of_revenue"]) / fun["revenue"] * 100

# --------------------------------------------- 3. triples of years t-1, t, t+1
g = fun.groupby("entity_id")
for c in ["revenue", "inventory", "gm", "year"]:
    fun["p_" + c] = g[c].shift(1)
    fun["n_" + c] = g[c].shift(-1)

d = fun[(fun["year"] - fun["p_year"] == 1) & (fun["n_year"] - fun["year"] == 1)].copy()
d = d[(d["year"] >= 2014) & (d["year"] <= 2023)]
d = d[(d["inventory"] / d["revenue"] >= MIN_INVENTORY)
      & (d["p_inventory"] / d["p_revenue"] >= MIN_INVENTORY)]

d["inv_growth"] = (d["inventory"] / d["p_inventory"] - 1) * 100
d["rev_growth"] = (d["revenue"] / d["p_revenue"] - 1) * 100
d["gap"] = d["inv_growth"] - d["rev_growth"]
d["d_gm_next"] = d["n_gm"] - d["gm"]
d["rev_growth_next"] = (d["n_revenue"] / d["revenue"] - 1) * 100

d = d[(d["gm"] > 0) & (d["gm"] < 100) & (d["n_gm"] > 0) & (d["n_gm"] < 100)]
d = d[(d["d_gm_next"].abs() <= MAX_MARGIN_MOVE)
      & ((d["gm"] - d["p_gm"]).abs() <= MAX_MARGIN_MOVE)]

# ----------------------------------------------- 4. quintiles within each year
LAB = ["Q1 inventory lags", "Q2", "Q3", "Q4", "Q5 inventory builds"]
d["q"] = d.groupby("year")["gap"].transform(lambda s: pd.qcut(s, 5, labels=LAB))
d = d.dropna(subset=["q"])

tab = d.groupby("q", observed=True).agg(
    n=("gap", "size"),
    gap=("gap", "median"),
    gm=("gm", "median"),
    d_gm=("d_gm_next", "median"),
    d_gm_mean=("d_gm_next", "mean"),
    fell=("d_gm_next", lambda s: (s < 0).mean() * 100),
    rev_next=("rev_growth_next", "median"),
)
q1, q5 = tab.index[0], tab.index[-1]
a, b = d.loc[d["q"] == q5, "d_gm_next"], d.loc[d["q"] == q1, "d_gm_next"]
tstat = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

print("Point-in-time S&P 500 rosters at each year end 2013-2024")
print("company-years after screens: %d (%d companies, signal years 2014-2023)"
      % (len(d), d["entity_id"].nunique()))
print()
print("quintiles of inventory growth minus revenue growth, formed within each year")
print(tab.round(2).to_string())
print()
print("top minus bottom quintile")
print("  next-year change in gross margin, median  %+.2f points" % (tab.loc[q5, "d_gm"] - tab.loc[q1, "d_gm"]))
print("  next-year change in gross margin, mean    %+.2f points" % (tab.loc[q5, "d_gm_mean"] - tab.loc[q1, "d_gm_mean"]))
print("  Welch t-statistic on the means            %.2f" % tstat)
print("  next-year revenue growth, median          %+.2f points" % (tab.loc[q5, "rev_next"] - tab.loc[q1, "rev_next"]))

# ------------------------------------------------------------ 5. robustness
d["gmt"] = d.groupby("year")["gm"].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]))
d["q_dbl"] = d.groupby(["year", "gmt"], observed=True)["gap"].transform(
    lambda s: pd.qcut(s, 5, labels=LAB) if s.notna().sum() >= 15 else np.nan)
dbl = d.dropna(subset=["q_dbl"]).groupby("q_dbl", observed=True)["d_gm_next"].median()
print()
print("double sort, quintiles formed within year and starting-margin tercile")
print("  spread %+.2f points (n=%d)" % (dbl.iloc[-1] - dbl.iloc[0], d["q_dbl"].notna().sum()))

yr = d.pivot_table(index="year", columns="q", values="d_gm_next", aggfunc="median", observed=True)
yr["spread"] = yr[q5] - yr[q1]
print()
print("spread by signal year")
print(yr[["spread"]].round(2).T.to_string())
print("  negative in %d of %d years" % ((yr["spread"] < 0).sum(), len(yr)))

sec = d.pivot_table(index="gics_sector", columns="q", values="d_gm_next", aggfunc="median", observed=True)
sec["spread"] = sec[q5] - sec[q1]
sec["n"] = d.groupby("gics_sector").size()
print()
print("spread by sector")
print(sec[[q1, q5, "spread", "n"]].round(2).sort_values("spread").to_string())

# ------------------------------------------------------------------ 6. chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
labels = [str(i) for i in tab.index]
ax1.bar(labels, tab["d_gm"], color="#3b82f6", width=0.62)
ax1.set_ylabel("Change in gross margin\nnext year (points)")
ax1.set_title("Does an inventory build predict a margin squeeze?\n"
              "S&P 500 companies that carry inventory, signal years 2014-2023", pad=14)
ax2.bar(labels, tab["rev_next"], color="#3b82f6", width=0.62)
ax2.set_ylabel("Revenue growth\nnext year (%)")
ax2.set_xlabel("Inventory growth minus revenue growth in the signal year")
for ax, col, fmt in ((ax1, "d_gm", "%+.2f"), (ax2, "rev_next", "%.1f")):
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.margins(y=0.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, v in enumerate(tab[col]):
        ax.text(i, v, fmt % v, ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, color="#e0e0e0")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
print("\nchart saved: %s" % OUT_PNG)
