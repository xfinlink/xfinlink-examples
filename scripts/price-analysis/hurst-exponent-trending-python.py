# Full write-up: https://xfinlink.com/blog/hurst-exponent-trending-python

import xfinlink as xfl
import pandas as pd
import numpy as np

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JNJ", "SPY"]

# -- Fetch 5Y daily prices -------------------------------------------------
df = xfl.prices(tickers, period="5y", fields=["close"])


def hurst_rs(prices: pd.Series, min_window: int = 20, max_window: int = 500) -> float:
    """Compute Hurst exponent using the rescaled range (R/S) method."""
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_returns)

    # Generate window sizes (powers of 2, plus intermediate values)
    window_sizes = []
    size = min_window
    while size <= min(max_window, n // 2):
        window_sizes.append(size)
        size = int(size * 1.4)

    if len(window_sizes) < 3:
        return np.nan

    log_rs = []
    log_n = []

    for w in window_sizes:
        num_blocks = n // w
        if num_blocks < 1:
            continue

        rs_values = []
        for i in range(num_blocks):
            block = log_returns[i * w : (i + 1) * w]
            mean_block = block.mean()
            deviations = block - mean_block
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = block.std(ddof=1)
            if s > 0:
                rs_values.append(r / s)

        if rs_values:
            log_rs.append(np.log(np.mean(rs_values)))
            log_n.append(np.log(w))

    if len(log_rs) < 3:
        return np.nan

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_n, log_rs, 1)
    return coeffs[0]


# -- Compute Hurst exponent per ticker -------------------------------------
results = []
for ticker in tickers:
    prices = df[df["ticker"] == ticker].sort_values("date")["close"]
    h = hurst_rs(prices)
    results.append({"ticker": ticker, "hurst": h})

results_df = pd.DataFrame(results).sort_values("hurst", ascending=False)

print("=== Hurst Exponent: Trending or Mean-Reverting? (5Y Daily) ===")
print(f"{'Ticker':8s} {'Hurst H':>8s}   Interpretation")
print("-" * 43)

for _, r in results_df.iterrows():
    h = r["hurst"]
    if h > 0.6:
        interp = "Moderate persistence (trending)"
    elif h > 0.5:
        interp = "Weak persistence (slightly trending)"
    elif h > 0.4:
        interp = "Weak anti-persistence (slightly mean-reverting)"
    else:
        interp = "Strong anti-persistence (mean-reverting)"
    print(f"{r['ticker']:8s} {h:>8.3f}   {interp}")

# -- Summary statistics -----------------------------------------------------
print("\n=== Statistical Context ===")
print(f"Mean H across tickers:   {results_df['hurst'].mean():.3f}")
print(f"Std dev of H:            {results_df['hurst'].std():.3f}")
print(
    f"All tickers fall in range [{results_df['hurst'].min():.3f}, "
    f"{results_df['hurst'].max():.3f}]"
)

print(
    """
H = 0.5  → random walk (no predictability)
H > 0.5  → persistent / trending
H < 0.5  → anti-persistent / mean-reverting

=== Comparison with ADF Test ===
The ADF test (augmented Dickey-Fuller) tests whether price LEVELS are
stationary — i.e., whether prices revert to a fixed mean. Hurst measures
whether RETURNS exhibit long-range dependence — i.e., whether the direction
of returns persists across timescales. A stock can fail the ADF test
(non-stationary levels, as expected for prices) while showing H ≈ 0.5
(no return persistence). These are complementary, not redundant, tests."""
)
