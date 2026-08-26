# Full write-up: https://xfinlink.com/blog/deferred-revenue-leading-indicator-python
"""Does deferred revenue lead reported revenue?

Tests whether year-over-year growth in the deferred revenue balance predicts
next quarter's revenue growth, and whether it adds anything to the far simpler
predictor of this quarter's revenue growth.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ids = xfl.index("sp500")["entity_id"].dropna().astype(int).tolist()
q = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="quarterly",
                                start="2019-01-01",
                                fields=["revenue", "deferred_revenue_current"])
               for i in range(0, len(ids), 100)], ignore_index=True)

q = q[~q["gics_sector"].isin(["Financials", "Real Estate"])]
q = q[q["gics_sector"].notna()]
q = q.dropna(subset=["revenue", "deferred_revenue_current"])
q = q[(q["revenue"] > 0) & (q["deferred_revenue_current"] > 0)]
q = (q.sort_values(["entity_id", "period_end", "filing_date"])
       .groupby(["entity_id", "period_end"], as_index=False).last())

panel = []
for eid, g in q.groupby("entity_id"):
    g = g.sort_values("period_end").reset_index(drop=True)
    if len(g) < 12:
        continue
    # Keep businesses that actually bill in advance: a deferred revenue balance
    # worth at least half a quarter of sales.
    if (g["deferred_revenue_current"] / g["revenue"]).median() < 0.5:
        continue
    g["rev_yoy"] = g["revenue"].pct_change(4) * 100
    g["dr_yoy"] = g["deferred_revenue_current"].pct_change(4) * 100
    g["rev_next"] = g["rev_yoy"].shift(-1)
    g = g.dropna(subset=["rev_yoy", "dr_yoy", "rev_next"])
    if len(g) >= 8:
        panel.append(g)

d = pd.concat(panel, ignore_index=True)
print(f"Companies: {d['entity_id'].nunique()}   quarter observations: {len(d)}")
print(f"Window: {d['period_end'].min().date()} to {d['period_end'].max().date()}")
print()

print("Rank correlation with NEXT quarter's revenue growth")
for label, col in [("deferred revenue growth", "dr_yoy"), ("revenue growth", "rev_yoy")]:
    r = stats.spearmanr(d[col], d["rev_next"])
    print(f"  {label:24s} rho {r.statistic:+.3f}   p = {r.pvalue:.1e}")

m1 = sm.OLS(d["rev_next"], sm.add_constant(d[["rev_yoy"]])).fit()
m2 = sm.OLS(d["rev_next"], sm.add_constant(d[["rev_yoy", "dr_yoy"]])).fit()
print(f"\nR-squared, revenue growth alone      {m1.rsquared:.3f}")
print(f"R-squared, adding deferred revenue   {m2.rsquared:.3f}")
print(f"deferred revenue coefficient {m2.params['dr_yoy']:+.3f} "
      f"(t = {m2.tvalues['dr_yoy']:.2f}, p = {m2.pvalues['dr_yoy']:.4f})")

# Where does the incremental information live?
TRAVEL = {"MAR", "RCL", "DAL", "UAL", "LUV", "CCL", "NCLH", "AAL", "HLT", "ABNB",
          "EXPE", "BKNG"}
print("\nSubsamples (R-squared before and after adding deferred revenue)")
cuts = [("all quarters, all companies", d),
        ("2022 onwards", d[d["period_end"] >= "2022-01-01"]),
        ("2022 onwards, excluding travel", d[(d["period_end"] >= "2022-01-01")
                                             & (~d["ticker"].isin(TRAVEL))])]
for label, sub in cuts:
    a = sm.OLS(sub["rev_next"], sm.add_constant(sub[["rev_yoy"]])).fit()
    b = sm.OLS(sub["rev_next"], sm.add_constant(sub[["rev_yoy", "dr_yoy"]])).fit()
    print(f"  {label:32s} n={len(sub):4d}  {a.rsquared:.3f} -> {b.rsquared:.3f}"
          f"   t = {b.tvalues['dr_yoy']:5.2f}")

per = []
for eid, g in d.groupby("entity_id"):
    if len(g) < 12:
        continue
    per.append({"ticker": g["ticker"].iloc[0], "sector": g["gics_sector"].iloc[0],
                "n": len(g),
                "dr": stats.spearmanr(g["dr_yoy"], g["rev_next"]).statistic,
                "rev": stats.spearmanr(g["rev_yoy"], g["rev_next"]).statistic})
per = pd.DataFrame(per)
print(f"\nPer company (n = {len(per)})")
print(f"  median rho, deferred revenue {per['dr'].median():+.3f}")
print(f"  median rho, revenue growth   {per['rev'].median():+.3f}")
print(f"  deferred revenue is the better predictor for "
      f"{(per['dr'] > per['rev']).sum()} of {len(per)} companies")
print("\nStrongest deferred revenue signal:")
print(per.nlargest(8, "dr")[["ticker", "sector", "n", "dr", "rev"]].round(3).to_string(index=False))

# Chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
ax.axline((0, 0), slope=1, color="#6b7280", linewidth=1)
ax.scatter(per["rev"], per["dr"], s=34, color="#3b82f6", alpha=0.85, edgecolors="none")
ax.set_xlabel("Rank correlation using this quarter's revenue growth")
ax.set_ylabel("Rank correlation using\ndeferred revenue growth")
ax.set_title("Predicting next quarter's revenue: deferred revenue against the revenue line")
ax.text(-0.08, 0.80, "below the line:\nrevenue growth predicts better",
        color="#9ca3af", fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("deferred-revenue-leading-indicator-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart saved")
