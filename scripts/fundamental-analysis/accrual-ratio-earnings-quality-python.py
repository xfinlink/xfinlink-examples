# Full write-up: https://xfinlink.com/blog/accrual-ratio-earnings-quality-python

import xfinlink as xfl
import pandas as pd

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "XOM", "JNJ"]

# -- Fetch latest annual fundamentals --------------------------------------
df = xfl.fundamentals(
    tickers,
    period_type="annual",
    fields=["net_income", "operating_cash_flow", "total_assets"],
    period="3y",
)

# -- Keep most recent annual period per ticker ------------------------------
latest = df.sort_values("period_end").groupby("ticker").tail(1).copy()

# -- Compute accrual ratio -------------------------------------------------
latest["accrual_ratio"] = (
    (latest["net_income"] - latest["operating_cash_flow"]) / latest["total_assets"]
)

# -- Classify quality -------------------------------------------------------
def quality_label(ar):
    if ar > 0.05:
        return "Low"
    elif ar > 0:
        return "Neutral"
    elif ar > -0.10:
        return "High"
    else:
        return "Highest"

latest["quality"] = latest["accrual_ratio"].apply(quality_label)
latest = latest.sort_values("accrual_ratio", ascending=False)

# -- Print results ----------------------------------------------------------
print("=== Accrual Ratio: Earnings Quality Screen (Latest Annual) ===")
header = (
    f"{'Ticker':6s}  {'Period':>12s}  {'Net Income':>12s}  {'OCF':>12s}  "
    f"{'Total Assets':>14s}  {'Accrual Ratio':>14s}   Quality"
)
print(header)
print("-" * 87)

for _, r in latest.iterrows():
    ni_str = f"${r['net_income'] / 1e3:>7.1f}B"
    ocf_str = f"${r['operating_cash_flow'] / 1e3:>7.1f}B"
    ta_str = f"${r['total_assets'] / 1e3:>8.1f}B"
    ar_str = f"{r['accrual_ratio']:>+8.3f}"
    print(
        f"{r['ticker']:6s}  {str(r['period_end'])[:10]:>12s}  {ni_str:>12s}  "
        f"{ocf_str:>12s}  {ta_str:>14s}  {ar_str:>14s}   {r['quality']}"
    )

# -- Summary ----------------------------------------------------------------
print("\n=== Summary ===")
best = latest.iloc[-1]
worst = latest.iloc[0]
spread = worst["accrual_ratio"] - best["accrual_ratio"]
print(
    f"Highest quality (most negative accrual ratio): {best['ticker']}"
    f" at {best['accrual_ratio']:+.3f}"
)
print(
    f"Lowest quality (most positive accrual ratio):  {worst['ticker']}"
    f" at {worst['accrual_ratio']:+.3f}"
)
print(f"Spread: {spread:.3f} — significant divergence in earnings quality")
