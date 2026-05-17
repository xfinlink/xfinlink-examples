# Full write-up: https://xfinlink.com/blog/equity-duration-rate-sensitivity-python

import xfinlink as xfl
import pandas as pd
import numpy as np

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "XOM", "PG", "JNJ", "KO"]
bond_etf = "TLT"

# -- Fetch 3Y daily prices -------------------------------------------------
all_tickers = tickers + [bond_etf]
df = xfl.prices(all_tickers, period="3y", fields=["close", "return_daily"])

# -- Pivot returns to wide format -------------------------------------------
returns = df.pivot_table(index="date", columns="ticker", values="return_daily").dropna()

# -- Regress each stock against TLT ----------------------------------------
results = []

for ticker in tickers:
    if ticker not in returns.columns or bond_etf not in returns.columns:
        continue

    y = returns[ticker].values
    x = returns[bond_etf].values

    # OLS regression: y = alpha + beta * x
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - beta * x_mean

    # Residuals and statistics
    y_hat = alpha + beta * x
    residuals = y - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Standard error of beta
    se_beta = np.sqrt(ss_res / (n - 2) / np.sum((x - x_mean) ** 2))
    t_stat = beta / se_beta

    # Interpretation
    if abs(t_stat) < 2.0:
        interp = "No significant relationship"
    elif beta > 0.2:
        interp = "Moves WITH bonds (rate-sensitive)"
    elif beta > 0:
        interp = "Weak positive (borderline)"
    elif beta > -0.2:
        interp = "Weak negative (borderline)"
    else:
        interp = "Moves AGAINST bonds (benefits from rate hikes)"

    results.append({
        "ticker": ticker,
        "tlt_beta": beta,
        "t_stat": t_stat,
        "r_squared": r_squared,
        "interp": interp,
    })

results_df = pd.DataFrame(results).sort_values("tlt_beta", ascending=False)

print("=== Equity Duration: TLT-Beta Analysis (3Y Daily) ===")
header = f"{'Ticker':8s} {'TLT-Beta':>10s} {'t-stat':>8s} {'R²':>7s}   Interpretation"
print(header)
print("-" * 53)

for _, r in results_df.iterrows():
    print(
        f"{r['ticker']:8s} {r['tlt_beta']:>+10.2f} {r['t_stat']:>8.2f} "
        f"{r['r_squared']:>7.3f}   {r['interp']}"
    )

print(
    """
=== Key Finding ===
Defensive/staples stocks (PG, JNJ, KO) behave as bond proxies — they
fall when rates rise. Energy (XOM) is the opposite — it benefits from
rising rates via inflation expectations. Most tech stocks (AAPL, MSFT,
NVDA) show NO statistically significant TLT-beta despite being called
"long duration" in market commentary."""
)
