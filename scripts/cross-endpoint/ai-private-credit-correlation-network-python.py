# Full write-up: https://xfinlink.com/blog/ai-private-credit-correlation-network-python

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-private-credit-correlation-network-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

AI_TICKERS = ["NVDA", "ORCL", "MSFT", "AMZN", "META"]
PRIVATE_CREDIT_TICKERS = ["APO", "KKR", "BX", "ARES", "BAM"]
MARKET = "SPY"
TICKERS = AI_TICKERS + PRIVATE_CREDIT_TICKERS + [MARKET]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.prices(TICKERS, period="2y", fields=["close", "return_daily"])
df = df[df["ticker"].isin(TICKERS)].copy()

require(not df.empty, "prices returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

returns = (
    df.pivot_table(index="date", columns="ticker", values="return_daily", aggfunc="last")
    .sort_index()
    .dropna(subset=TICKERS)
)
require(len(returns) >= 250, "not enough complete daily return observations")

corr = returns[TICKERS].corr()


def upper_triangle_average(block: pd.DataFrame) -> float:
    values = block.to_numpy()
    mask = np.triu(np.ones(values.shape, dtype=bool), k=1)
    return float(values[mask].mean())


intra_ai = upper_triangle_average(corr.loc[AI_TICKERS, AI_TICKERS])
intra_private_credit = upper_triangle_average(corr.loc[PRIVATE_CREDIT_TICKERS, PRIVATE_CREDIT_TICKERS])
cross = corr.loc[AI_TICKERS, PRIVATE_CREDIT_TICKERS]
ai_private = float(cross.to_numpy().mean())
ai_market = float(corr.loc[AI_TICKERS, MARKET].mean())
private_market = float(corr.loc[PRIVATE_CREDIT_TICKERS, MARKET].mean())

pair_rows = []
for ai in AI_TICKERS:
    for private in PRIVATE_CREDIT_TICKERS:
        pair_rows.append((ai, private, corr.loc[ai, private]))
pair_df = pd.DataFrame(pair_rows, columns=["ai_ticker", "private_credit_ticker", "correlation"])
pair_df = pair_df.sort_values("correlation", ascending=False)

rolling = []
window = 60
for i in range(window - 1, len(returns)):
    sample = returns.iloc[i - window + 1 : i + 1]
    c = sample[TICKERS].corr().loc[AI_TICKERS, PRIVATE_CREDIT_TICKERS]
    rolling.append((returns.index[i], float(c.to_numpy().mean())))
rolling_df = pd.DataFrame(rolling, columns=["date", "cross_correlation"]).set_index("date")

betas = []
for ticker in PRIVATE_CREDIT_TICKERS:
    beta_nvda = returns[ticker].cov(returns["NVDA"]) / returns["NVDA"].var()
    beta_spy = returns[ticker].cov(returns[MARKET]) / returns[MARKET].var()
    betas.append((ticker, beta_nvda, beta_spy))
beta_df = pd.DataFrame(betas, columns=["ticker", "beta_to_nvda", "beta_to_spy"])

print("=== AI vs Private-Credit Market Linkage ===")
print(f"Sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print()
print("Average daily return correlations:")
print(f"Intra-AI basket:             {intra_ai:.3f}")
print(f"Intra-private-credit basket: {intra_private_credit:.3f}")
print(f"AI / private-credit cross:   {ai_private:.3f}")
print(f"AI / SPY average:            {ai_market:.3f}")
print(f"Private credit / SPY average:{private_market:.3f}")
print()
print(f"Latest 60-day AI/private-credit cross-correlation: {rolling_df.iloc[-1, 0]:.3f}")
print(f"Median 60-day cross-correlation:                  {rolling_df['cross_correlation'].median():.3f}")
print()

print("Highest AI/private-credit pair correlations:")
for _, row in pair_df.head(5).iterrows():
    print(f"{row['ai_ticker']:4s} / {row['private_credit_ticker']:4s}: {row['correlation']:.3f}")

print()
print("Private-credit beta estimates:")
for _, row in beta_df.sort_values("beta_to_spy", ascending=False).iterrows():
    print(f"{row['ticker']:4s} beta_to_NVDA={row['beta_to_nvda']:.2f}  beta_to_SPY={row['beta_to_spy']:.2f}")

plt.rcParams.update(
    {
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#0a0a0a",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "font.size": 10,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1.35]})

labels = ["AI", "Private\ncredit", "AI/private", "AI/SPY", "Private/SPY"]
values = [intra_ai, intra_private_credit, ai_private, ai_market, private_market]
colors = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ef4444"]
bars = ax1.bar(labels, values, color=colors)
ax1.set_ylim(0, max(values) + 0.15)
ax1.set_ylabel("Average correlation")
ax1.set_title("Correlation Blocks")
ax1.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
for bar, value in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

ax2.plot(rolling_df.index, rolling_df["cross_correlation"], color="#3b82f6", linewidth=2)
ax2.axhline(ai_private, color="#f59e0b", linewidth=1.2, linestyle="--", label="Full-period average")
ax2.set_title("60-Day AI/Private-Credit Linkage")
ax2.set_xlabel("Date")
ax2.set_ylabel("Average cross-correlation")
ax2.legend(frameon=False)
ax2.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
