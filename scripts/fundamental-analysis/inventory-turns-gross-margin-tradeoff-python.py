# Full write-up: https://xfinlink.com/blog/inventory-turns-gross-margin-tradeoff-python
"""Do faster inventory turns come with thinner margins?

Cross-section of current S&P 500 members: inventory turnover against gross
margin, and gross margin return on inventory (GMROI) as the combined measure.
Built from SEC EDGAR public filings and market data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIELDS = ["revenue", "cost_of_revenue", "gross_profit", "inventory"]
SKIP_SECTORS = {"Financials", "Utilities", "Real Estate"}

# ── Universe: companies in the S&P 500 today ─────────────────────────────
uni = xfl.index("sp500")
tickers = sorted(uni.loc[uni["removed_date"].isna(), "ticker"].dropna().unique())

f = xfl.fundamentals(tickers, period_type="annual", start="2025-06-01",
                     version="restated", fields=FIELDS)
latest = f.sort_values("period_end").groupby("ticker").tail(1)

# ── Sample rules ─────────────────────────────────────────────────────────
s = latest[~latest["gics_sector"].isin(SKIP_SECTORS)].dropna(subset=FIELDS)
s = s[(s[FIELDS] > 0).all(axis=1)].copy()
s["gross_profit_calc"] = s["revenue"] - s["cost_of_revenue"]
gap = (s["gross_profit"] - s["gross_profit_calc"]).abs() / s["revenue"]
s = s[gap <= 0.02]                                   # two gross profit measures must agree
s = s[s["inventory"] >= 0.02 * s["revenue"]]         # firms that actually carry stock

# ── Measures ─────────────────────────────────────────────────────────────
s["gross_margin"] = s["gross_profit_calc"] / s["revenue"]
s["turns"] = s["cost_of_revenue"] / s["inventory"]
s["days_inventory"] = 365 / s["turns"]
s["gmroi"] = s["gross_profit_calc"] / s["inventory"]

fit = stats.linregress(np.log(s["turns"]), np.log(s["gross_margin"]))
rho_m, p_m = stats.spearmanr(s["turns"], s["gross_margin"])
rho_g, p_g = stats.spearmanr(s["turns"], s["gmroi"])

s["quintile"] = pd.qcut(s["turns"], 5, labels=["Q1 slowest", "Q2", "Q3", "Q4", "Q5 fastest"])
tab = s.groupby("quintile", observed=True).agg(
    n=("ticker", "size"),
    days=("days_inventory", "median"),
    margin=("gross_margin", "median"),
    gmroi=("gmroi", "median"),
)

# ── Output ───────────────────────────────────────────────────────────────
print(f"Sample: {len(s)} current S&P 500 members, latest annual filing "
      f"({s['period_end'].min():%Y-%m-%d} to {s['period_end'].max():%Y-%m-%d})")
print(f"\nlog gross margin on log turns: slope {fit.slope:+.3f}  "
      f"R2 {fit.rvalue ** 2:.3f}  p {fit.pvalue:.2g}")
print(f"Spearman turns vs gross margin: {rho_m:+.3f} (p {p_m:.2g})")
print(f"Spearman turns vs GMROI:        {rho_g:+.3f} (p {p_g:.2g})")

print("\nTurnover quintile      n   days of inventory   gross margin   GMROI")
for q, r in tab.iterrows():
    print(f"{q:<18}{r['n']:5.0f}{r['days']:16.0f}{r['margin'] * 100:14.0f}%{r['gmroi']:9.2f}")

print("\nHighest gross profit per dollar of inventory")
for _, r in s.nlargest(5, "gmroi").iterrows():
    print(f"  {r['ticker']:<6}{r['entity_name'][:28]:<30}"
          f"{r['days_inventory']:6.0f} days{r['gross_margin'] * 100:6.0f}%{r['gmroi']:8.2f}")
print("Lowest gross profit per dollar of inventory")
for _, r in s.nsmallest(5, "gmroi").iterrows():
    print(f"  {r['ticker']:<6}{r['entity_name'][:28]:<30}"
          f"{r['days_inventory']:6.0f} days{r['gross_margin'] * 100:6.0f}%{r['gmroi']:8.2f}")

sec = s.groupby("gics_sector").agg(n=("ticker", "size"), days=("days_inventory", "median"),
                                   margin=("gross_margin", "median"), gmroi=("gmroi", "median"))
print("\nSector medians (sectors with at least 5 names)")
for name, r in sec[sec["n"] >= 5].sort_values("gmroi").iterrows():
    print(f"  {name:<24}{r['n']:4.0f}{r['days']:8.0f} days{r['margin'] * 100:6.0f}%{r['gmroi']:8.2f}")

# ── Chart ────────────────────────────────────────────────────────────────
BG, FG, ACCENT = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG,
                               gridspec_kw={"height_ratios": [1.6, 1]})
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#333333")

ax1.scatter(s["turns"], s["gross_margin"] * 100, s=26, color=ACCENT, alpha=0.65,
            edgecolors="none")
grid = np.linspace(np.log(s["turns"].min()), np.log(s["turns"].max()), 100)
ax1.plot(np.exp(grid), np.exp(fit.intercept + fit.slope * grid) * 100,
         color="#f59e0b", lw=2,
         label=f"fitted power law, slope {fit.slope:+.2f}")
for tk in ["COST", "ORLY", "AVGO", "CAH", "LMT", "NKE"]:
    row = s[s["ticker"] == tk]
    if not row.empty:
        ax1.annotate(tk, (row["turns"].iloc[0], row["gross_margin"].iloc[0] * 100),
                     textcoords="offset points", xytext=(6, 4), color=FG, fontsize=9)
ax1.set_xscale("log")
ax1.set_xlabel("Inventory turns per year (log scale)", color=FG)
ax1.set_ylabel("Gross margin (%)", color=FG)
ax1.set_title("Faster inventory, thinner margin: S&P 500 cross-section",
              color=FG, fontsize=13)
leg = ax1.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, fontsize=9)

ax2.bar(range(len(tab)), tab["gmroi"], color=ACCENT, width=0.6)
ax2.set_xticks(range(len(tab)))
ax2.set_xticklabels([f"{q}\n{r['days']:.0f} days" for q, r in tab.iterrows()],
                    color=FG, fontsize=9)
ax2.set_ylabel("Gross profit per $1\nof inventory", color=FG)
ax2.set_title("Median gross margin return on inventory by turnover quintile",
              color=FG, fontsize=11)
for i, v in enumerate(tab["gmroi"]):
    ax2.text(i, v + 0.08, f"{v:.2f}", ha="center", color=FG, fontsize=9)

plt.tight_layout()
plt.savefig("inventory-turns-gross-margin-tradeoff-python.png", dpi=150, facecolor=BG)
