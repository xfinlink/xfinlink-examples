# Full write-up: https://xfinlink.com/blog/rolling-sharpe-alpha-decay-python

import xfinlink as xfl
import pandas as pd
import numpy as np

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JNJ"]
WINDOW = 126  # 6 months of trading days
RF_ANNUAL = 0.05  # 5% annualized risk-free rate
RF_DAILY = RF_ANNUAL / 252

# -- Fetch 3Y daily prices -------------------------------------------------
df = xfl.prices(tickers, period="3y", fields=["close", "return_daily"])

# -- Compute rolling Sharpe per ticker -------------------------------------
results = []

for ticker in tickers:
    t_df = df[df["ticker"] == ticker].sort_values("date").copy()
    t_df["excess_return"] = t_df["return_daily"] - RF_DAILY

    t_df["rolling_mean"] = t_df["excess_return"].rolling(WINDOW).mean()
    t_df["rolling_std"] = t_df["excess_return"].rolling(WINDOW).std()
    t_df["rolling_sharpe"] = (t_df["rolling_mean"] / t_df["rolling_std"]) * np.sqrt(252)

    valid = t_df.dropna(subset=["rolling_sharpe"])

    if len(valid) == 0:
        continue

    median_sharpe = valid["rolling_sharpe"].median()
    current_sharpe = valid["rolling_sharpe"].iloc[-1]
    pct_above_1 = (valid["rolling_sharpe"] > 1.0).mean() * 100
    trend = "Improving" if current_sharpe > median_sharpe else "Deteriorating"

    results.append({
        "ticker": ticker,
        "median_sharpe": median_sharpe,
        "current_sharpe": current_sharpe,
        "pct_above_1": pct_above_1,
        "trend": trend,
    })

results_df = pd.DataFrame(results).sort_values("pct_above_1", ascending=False)

print("=== Rolling 6-Month Sharpe Ratio: Alpha Persistence (3Y) ===")
header = f"{'Ticker':8s} {'Median Sharpe':>14s} {'Current Sharpe':>16s} {'% Days > 1.0':>14s}   Trend"
print(header)
print("-" * 62)

for _, r in results_df.iterrows():
    print(
        f"{r['ticker']:8s} {r['median_sharpe']:>14.2f} {r['current_sharpe']:>16.2f} "
        f"{r['pct_above_1']:>13.1f}%   {r['trend']}"
    )

# -- Interpretation ---------------------------------------------------------
print("\n=== Interpretation ===")
nvda = results_df[results_df["ticker"] == "NVDA"].iloc[0]
xom = results_df[results_df["ticker"] == "XOM"].iloc[0]
aapl = results_df[results_df["ticker"] == "AAPL"].iloc[0]

print(
    f"NVDA: Sharpe above 1.0 for {nvda['pct_above_1']:.0f}% of the period — strongest sustained"
)
print(
    f"      alpha. But current {nvda['current_sharpe']:.2f} < median {nvda['median_sharpe']:.2f}"
    f" → alpha is decaying."
)
print(
    f"XOM:  Sharpe above 1.0 for only {xom['pct_above_1']:.0f}% of the period, but current"
    f" {xom['current_sharpe']:.2f}"
)
print(
    f"      is far above median {xom['median_sharpe']:.2f} → regime shift, alpha surging."
)
print(
    f"AAPL: Median Sharpe near zero over 3Y, but current {aapl['current_sharpe']:.2f} suggests"
)
print("      a recovery from a prolonged underperformance regime.")
