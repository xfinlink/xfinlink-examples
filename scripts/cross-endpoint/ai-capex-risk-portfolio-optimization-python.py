# Full write-up: https://xfinlink.com/blog/ai-capex-risk-portfolio-optimization-python

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-capex-risk-portfolio-optimization-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

AI_TICKERS = ["MSFT", "AMZN", "META", "GOOG", "ORCL", "NVDA", "AVGO"]
DEFENSIVE_TICKERS = ["PG", "KO", "JNJ", "WMT", "MCD"]
TICKERS = AI_TICKERS + DEFENSIVE_TICKERS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


prices_raw = xfl.prices(TICKERS, period="2y", fields=["adj_close"])
prices_raw = prices_raw[prices_raw["ticker"].isin(TICKERS)].copy()
require(not prices_raw.empty, "prices returned no rows")
require(set(TICKERS).issubset(set(prices_raw["ticker"])), "missing one or more requested tickers")

prices = prices_raw.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").dropna(subset=TICKERS)
returns = prices[TICKERS].pct_change().dropna()
require(len(returns) >= 250, "not enough complete return observations")

fund = xfl.fundamentals(
    TICKERS,
    period_type="annual",
    period="4y",
    fields=["revenue", "capital_expenditures", "free_cash_flow"],
)
fund = fund[fund["ticker"].isin(TICKERS)].sort_values(["ticker", "period_end"]).groupby("ticker", group_keys=False).tail(1)
fund = fund.set_index("ticker")
require(set(TICKERS).issubset(set(fund.index)), "missing one or more latest fundamentals")
require(fund[["revenue", "capital_expenditures", "free_cash_flow"]].notna().all().all(), "latest fundamentals contain missing values")
require((fund["revenue"] > 0).all(), "latest revenue must be positive")

fund["capex_intensity"] = fund["capital_expenditures"].abs() / fund["revenue"]
fund["fcf_margin"] = fund["free_cash_flow"] / fund["revenue"]

cov = returns[TICKERS].cov() * 252
capex_vector = fund.loc[TICKERS, "capex_intensity"].to_numpy()
equal_weights = np.ones(len(TICKERS)) / len(TICKERS)
equal_capex = float(equal_weights @ capex_vector)
equal_vol = float(np.sqrt(equal_weights @ cov.to_numpy() @ equal_weights))
equal_ai_weight = sum(equal_weights[TICKERS.index(ticker)] for ticker in AI_TICKERS)

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    {"type": "ineq", "fun": lambda w: 0.60 * equal_capex - w @ capex_vector},
    {"type": "ineq", "fun": lambda w: 0.45 - sum(w[TICKERS.index(ticker)] for ticker in AI_TICKERS)},
]
bounds = [(0, 0.25)] * len(TICKERS)

result = minimize(
    lambda w: np.sqrt(w @ cov.to_numpy() @ w),
    equal_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000},
)
require(result.success, f"optimization failed: {result.message}")

weights = pd.Series(result.x, index=TICKERS).sort_values(ascending=False)
optimized_vol = float(np.sqrt(result.x @ cov.to_numpy() @ result.x))
optimized_capex = float(result.x @ capex_vector)
optimized_ai_weight = sum(result.x[TICKERS.index(ticker)] for ticker in AI_TICKERS)

print("=== AI Capex-Risk Portfolio Optimization ===")
print(f"Return sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print("Constraints: capex intensity <= 60% of equal-weight portfolio; AI basket weight <= 45%; single name <= 25%")
print()
print(f"Equal-weight annualized volatility: {equal_vol:.1%}")
print(f"Optimized annualized volatility:    {optimized_vol:.1%}")
print(f"Equal-weight capex intensity:       {equal_capex:.1%}")
print(f"Optimized capex intensity:          {optimized_capex:.1%}")
print(f"Equal-weight AI exposure:           {equal_ai_weight:.1%}")
print(f"Optimized AI exposure:              {optimized_ai_weight:.1%}")
print()
print("Optimized weights:")
for ticker, weight in weights.items():
    if weight >= 0.005:
        print(
            f"{ticker:5s} weight={weight:5.1%}  "
            f"capex_intensity={fund.loc[ticker, 'capex_intensity']:5.1%}  "
            f"FCF_margin={fund.loc[ticker, 'fcf_margin']:5.1%}"
        )

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

fig, ax = plt.subplots(figsize=(10, 5))
plot_weights = weights[weights >= 0.005]
colors = ["#3b82f6" if ticker in AI_TICKERS else "#22c55e" for ticker in plot_weights.index]
ax.bar(plot_weights.index, plot_weights * 100, color=colors)
ax.set_title("AI Capex-Risk Portfolio Optimization")
ax.set_xlabel("Company")
ax.set_ylabel("Optimized weight (%)")
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
