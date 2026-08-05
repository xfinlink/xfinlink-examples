# Full write-up: https://xfinlink.com/blog/dividend-free-cash-flow-coverage-python
"""Rank S&P 500 dividend payers on free-cash-flow coverage and compare that
ranking with the conventional earnings payout ratio."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl
from scipy.stats import spearmanr

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIELDS = [
    "period_end", "fiscal_year", "net_income", "operating_cash_flow",
    "capital_expenditures", "dividends_paid_common",
]

tickers = xfl.index("sp500")["ticker"].dropna().unique().tolist()
raw = xfl.fundamentals(tickers, period_type="annual", start="2024-06-01",
                       fields=FIELDS, max_rows=10000)

latest = raw.sort_values("period_end").groupby("ticker").tail(1)
df = latest[
    (latest["dividends_paid_common"] > 0)
    & (latest["net_income"] > 0)
    & latest["operating_cash_flow"].notna()
    & latest["capital_expenditures"].notna()
    & ~latest["gics_sector"].isin(["Financials", "Real Estate"])
].copy()

df["dividends"] = df["dividends_paid_common"]
df["fcf"] = df["operating_cash_flow"] - df["capital_expenditures"]
df["payout"] = df["dividends"] / df["net_income"]
df["coverage"] = df["fcf"] / df["dividends"]

rho, pval = spearmanr(df["payout"], df["coverage"])
uncovered = df[df["coverage"] < 1.0]
blind_spot = df[(df["payout"] < 0.60) & (df["coverage"] < 1.0)].sort_values("coverage")
inverse = df[(df["payout"] > 1.0) & (df["coverage"] > 1.5)].sort_values("coverage",
                                                                       ascending=False)

print("=== Dividend Coverage: Cash vs Earnings ===")
print(f"Universe: {len(df)} S&P 500 dividend payers "
      f"(latest annual filing, ex-Financials, ex-Real Estate)")
print(f"Fiscal periods end {df['period_end'].min():%Y-%m-%d} to "
      f"{df['period_end'].max():%Y-%m-%d}")
print(f"Spearman rank correlation, payout vs coverage: {rho:.3f} (p={pval:.1e})")
print(f"Dividend not covered by free cash flow: {len(uncovered)} of {len(df)} "
      f"({len(uncovered)/len(df):.1%})")
print(f"Median coverage: {df['coverage'].median():.2f}x   "
      f"Median payout: {df['payout'].median():.1%}")

print(f"\n--- Payout ratio under 60%, free cash flow below the dividend "
      f"({len(blind_spot)} names) ---")
print(f"{'Ticker':<7}{'Sector':<24}{'NetInc':>9}{'OCF':>9}{'Capex':>9}"
      f"{'FCF':>10}{'Div':>9}{'Payout':>9}{'Cover':>9}")
for _, r in blind_spot.iterrows():
    print(f"{r['ticker']:<7}{r['gics_sector'][:23]:<24}{r['net_income']:>9,.0f}"
          f"{r['operating_cash_flow']:>9,.0f}{r['capital_expenditures']:>9,.0f}"
          f"{r['fcf']:>10,.0f}{r['dividends']:>9,.0f}"
          f"{r['payout']:>8.1%}{r['coverage']:>9.2f}x")

print("\nSector mix of that group:")
for sector, n in blind_spot["gics_sector"].value_counts().items():
    print(f"  {sector:<24}{n:>3}")

print(f"\n--- Payout ratio above 100%, cash covers the dividend more than 1.5x "
      f"({len(inverse)} names) ---")
print(f"{'Ticker':<7}{'Sector':<24}{'NetInc':>9}{'FCF':>10}{'Div':>9}"
      f"{'Payout':>9}{'Cover':>9}")
for _, r in inverse.iterrows():
    print(f"{r['ticker']:<7}{r['gics_sector'][:23]:<24}{r['net_income']:>9,.0f}"
          f"{r['fcf']:>10,.0f}{r['dividends']:>9,.0f}"
          f"{r['payout']:>8.1%}{r['coverage']:>9.2f}x")

# ---------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, ax = plt.subplots(figsize=(10, 6))
x = df["payout"].clip(upper=1.6) * 100
y = df["coverage"].clip(lower=-5, upper=8)
util = df["gics_sector"] == "Utilities"
ax.axhspan(-5, 1.0, xmin=0, xmax=60 / 160, color="#3b82f6", alpha=0.10, zorder=0)
ax.scatter(x[~util], y[~util], s=26, color="#9ca3af", alpha=0.75,
           edgecolors="none", label="All other sectors", zorder=2)
ax.scatter(x[util], y[util], s=34, color="#3b82f6", alpha=0.95,
           edgecolors="none", label="Utilities", zorder=3)
ax.axhline(1.0, color="#e0e0e0", lw=0.9, ls="--", zorder=1)
ax.axvline(60, color="#e0e0e0", lw=0.9, ls="--", zorder=1)
for tk in ["ORCL", "PCG", "AEP", "MMM", "NUE", "ABBV", "CVS"]:
    if tk in set(df["ticker"]):
        r = df[df["ticker"] == tk].iloc[0]
        ax.annotate(tk, (min(r["payout"], 1.6) * 100,
                         max(min(r["coverage"], 8), -5)),
                    xytext=(5, 4), textcoords="offset points",
                    color="#e0e0e0", fontsize=8.5)
ax.set_xlabel("Dividends as a share of net income (%, capped at 160)")
ax.set_ylabel("Free cash flow divided by dividends paid (x, capped at 8)")
ax.set_title("Dividend coverage: cash generated vs earnings reported")
ax.set_ylim(-5.4, 8.4)
ax.set_xlim(0, 165)
ax.legend(frameon=False, loc="upper right")
ax.text(3, 0.35, "Looks safe on earnings, not covered by cash",
        color="#3b82f6", fontsize=9)
plt.tight_layout()
plt.savefig("dividend-free-cash-flow-coverage-python.png", dpi=150)
