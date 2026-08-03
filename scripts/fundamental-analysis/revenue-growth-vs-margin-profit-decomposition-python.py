# Full write-up: https://xfinlink.com/blog/revenue-growth-vs-margin-profit-decomposition-python
"""Does revenue growth explain profit growth across the S&P 500?

Five-year net profit growth splits exactly into a revenue term and a net
margin term:

    ln(profit2 / profit1) = ln(revenue2 / revenue1) + ln(margin2 / margin1)

The script builds both terms for every non-financial S&P 500 member of the
2019 roster and measures which one drives the spread across companies.
"""
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

BASE = [2017, 2018, 2019]
RECENT = [2022, 2023, 2024]
HELD_OUT = {"Financials", "Real Estate", "Utilities"}
SLUG = "revenue-growth-vs-margin-profit-decomposition-python"


def fiscal_year(period_end):
    """Fiscal year Y runs from June of Y to May of Y+1."""
    d = pd.to_datetime(period_end)
    return np.where(d.dt.month < 6, d.dt.year - 1, d.dt.year)


members = xfl.index("sp500", as_of="2019-12-31").dropna(subset=["entity_id"])
ids = sorted(set(members["entity_id"].astype(int)))

frames = [xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                           start="2017-06-01", end="2025-05-31",
                           fields=["revenue", "net_income"], max_rows=100000)
          for i in range(0, len(ids), 100)]
f = pd.concat([x for x in frames if len(x)], ignore_index=True)
f["fy"] = fiscal_year(f["period_end"])
f = f[~f["gics_sector"].isin(HELD_OUT)].dropna(subset=["revenue", "net_income"])
f = f.sort_values("period_end").groupby(["entity_id", "fy"], as_index=False).tail(1)
n_universe = f["entity_id"].nunique()

rows = []
for eid, g in f.groupby("entity_id"):
    g = g.set_index("fy")
    if not all(y in g.index for y in BASE + RECENT):
        continue
    a, b = g.loc[BASE], g.loc[RECENT]
    r1, r2 = a["revenue"].sum(), b["revenue"].sum()
    p1, p2 = a["net_income"].sum(), b["net_income"].sum()
    if min(r1, r2, p1, p2) <= 0:
        continue
    rows.append({"sector": g["gics_sector"].iloc[-1],
                 "m1": 100 * p1 / r1, "m2": 100 * p2 / r2,
                 "g_rev": np.log(r2 / r1),
                 "g_mar": np.log((p2 / r2) / (p1 / r1)),
                 "g_prof": np.log(p2 / p1)})
w = pd.DataFrame(rows)

pct = lambda x: 100 * (np.exp(x) - 1)
var_p, var_r, var_m = w["g_prof"].var(), w["g_rev"].var(), w["g_mar"].var()
cov2 = 2 * w[["g_rev", "g_mar"]].cov().iloc[0, 1]
r2_fit = stats.linregress(w["g_rev"], w["g_prof"]).rvalue ** 2
lo, hi = w["g_prof"].quantile([0.01, 0.99])
t = w[(w["g_prof"] >= lo) & (w["g_prof"] <= hi)]

print("Five-year profit growth decomposition, S&P 500 members outside")
print("banks, property trusts and regulated utilities")
print("Roster point-in-time at 2019-12-31; fiscal 2017-2019 totals against")
print("fiscal 2022-2024 totals")
print("%d on the roster, %d after the sector hold-out, %d in the sample\n"
      % (len(ids), n_universe, len(w)))

print("Median five-year change (each line a separate median, so the two")
print("terms need not multiply out to the profit figure)")
print("  net profit                    %+7.1f%%" % pct(w["g_prof"].median()))
print("  revenue term                  %+7.1f%%" % pct(w["g_rev"].median()))
print("  net margin term               %+7.1f%%" % pct(w["g_mar"].median()))
print("  median net margin       %5.1f%% -> %5.1f%%\n" % (w["m1"].median(), w["m2"].median()))

print("Share of the cross-sectional variance in profit growth")
print("  revenue growth                %7.1f%%" % (100 * var_r / var_p))
print("  net margin change             %7.1f%%" % (100 * var_m / var_p))
print("  covariance of the two         %7.1f%%" % (100 * cov2 / var_p))
print("  margin share, 1st/99th trim   %7.1f%%  (n = %d)" % (100 * t["g_mar"].var() / t["g_prof"].var(), len(t)))
print("  R-squared, profit growth on revenue growth   %.3f" % r2_fit)
print("  revenue up and profit down    %7.1f%% of companies" % (100 * ((w["g_rev"] > 0) & (w["g_prof"] < 0)).mean()))
print("  revenue down and profit up    %7.1f%% of companies\n" % (100 * ((w["g_rev"] < 0) & (w["g_prof"] > 0)).mean()))

w["q"] = pd.qcut(w["g_rev"], 4, labels=["Q1 slowest", "Q2", "Q3", "Q4 fastest"])
q = w.groupby("q", observed=True).agg(n=("g_prof", "size"), rev=("g_rev", "median"),
                                      mar=("g_mar", "median"), prof=("g_prof", "median"),
                                      fell=("g_prof", lambda x: 100 * (x < 0).mean()))
print("By revenue-growth quartile      n   revenue    margin    profit   profit fell")
for k, r in q.iterrows():
    print("  %-12s %6d  %+7.1f%%  %+7.1f%%  %+7.1f%%   %8.0f%%"
          % (k, r["n"], pct(r["rev"]), pct(r["mar"]), pct(r["prof"]), r["fell"]))

s = w.groupby("sector").agg(n=("g_prof", "size"), rev=("g_rev", "median"), mar=("g_mar", "median"),
                            prof=("g_prof", "median"), m1=("m1", "median"), m2=("m2", "median"))
s = s.sort_values("prof", ascending=False)
print("\nBy sector                       n   revenue    margin    profit   net margin")
for k, r in s.iterrows():
    print("  %-24s %4d  %+7.1f%%  %+7.1f%%  %+7.1f%%   %4.1f%% -> %4.1f%%"
          % (k, r["n"], pct(r["rev"]), pct(r["mar"]), pct(r["prof"]), r["m1"], r["m2"]))

# ── chart ────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
data = [w["g_mar"].values, w["g_rev"].values, w["g_prof"].values]
labels = ["From net margin change", "From revenue growth", "Net profit growth"]
bp = ax.boxplot(data, orientation="horizontal", widths=0.55, whis=(5, 95),
                showfliers=False, patch_artist=True,
                medianprops={"color": "#0a0a0a", "lw": 2})
for box in bp["boxes"]:
    box.set(facecolor="#3b82f6", edgecolor="#3b82f6", alpha=0.85)
for part in ("whiskers", "caps"):
    for item in bp[part]:
        item.set(color="#e0e0e0", lw=1.2)
ax.axvline(0, color="#6b7280", lw=1, ls="--")
ax.set_yticklabels(labels)
marks = [-60, -30, 0, 50, 100, 200, 400]
ax.set_xticks([np.log(1 + m / 100) for m in marks])
ax.set_xticklabels(["%+d%%" % m for m in marks])
ax.set_xlim(-1.25, 1.75)
ax.set_xlabel("Five-year change. Box spans the middle half of companies, "
              "whiskers the 5th to 95th percentile")
ax.set_title("Margin change, not revenue growth, drives the spread in profit growth", pad=12)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("%s.png" % SLUG, dpi=150, facecolor="#0a0a0a")
