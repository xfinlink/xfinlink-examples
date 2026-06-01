# Full write-up: https://xfinlink.com/blog/ai-capex-payback-hurdle-python

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-capex-payback-hurdle-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

TICKERS = ["MSFT", "AMZN", "META", "GOOG", "ORCL"]
FIELDS = ["revenue", "operating_income", "capital_expenditures", "free_cash_flow"]
PAYBACK_YEARS = 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.fundamentals(TICKERS, period_type="annual", period="5y", fields=FIELDS)
df = df[df["ticker"].isin(TICKERS)].sort_values(["ticker", "period_end"]).copy()

require(not df.empty, "fundamentals returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

recent = df.groupby("ticker", group_keys=False).tail(2).copy()
require((recent.groupby("ticker").size() == 2).all(), "each ticker needs two annual observations")
require(recent[FIELDS].notna().all().all(), "recent annual fundamentals contain missing values")

latest = recent.groupby("ticker", group_keys=False).tail(1).set_index("ticker").copy()
prior = recent.groupby("ticker", group_keys=False).head(1).set_index("ticker").copy()

require((latest["revenue"] > 0).all(), "latest revenue must be positive")
require((latest["operating_income"] > 0).all(), "latest operating income must be positive")

latest["capex_abs"] = latest["capital_expenditures"].abs()
latest["operating_margin"] = latest["operating_income"] / latest["revenue"]
latest["revenue_growth_dollars"] = latest["revenue"] - prior["revenue"]
latest["revenue_growth_rate"] = latest["revenue"] / prior["revenue"] - 1
latest["required_revenue"] = latest["capex_abs"] / PAYBACK_YEARS / latest["operating_margin"]
latest["hurdle_pct_revenue"] = latest["required_revenue"] / latest["revenue"]
latest["coverage_ratio"] = latest["revenue_growth_dollars"] / latest["required_revenue"]

latest = latest.sort_values("coverage_ratio")

aggregate_capex = latest["capex_abs"].sum()
aggregate_growth = latest["revenue_growth_dollars"].sum()
aggregate_required = latest["required_revenue"].sum()
aggregate_coverage = aggregate_growth / aggregate_required
aggregate_hurdle = aggregate_required / latest["revenue"].sum()

print("=== AI Capex Payback Hurdle ===")
print(f"Payback assumption: {PAYBACK_YEARS} years at each company's current operating margin")
print(f"Latest annual periods: {latest['period_end'].min().date()} to {latest['period_end'].max().date()}")
print()
print(f"Combined latest capex: ${aggregate_capex / 1000:.1f}B")
print(f"Required annual incremental revenue: ${aggregate_required / 1000:.1f}B")
print(f"Actual latest annual revenue growth: ${aggregate_growth / 1000:.1f}B")
print(f"Growth / required revenue: {aggregate_coverage:.2f}x")
print(f"Required revenue as share of current revenue: {aggregate_hurdle:.1%}")
print()
print("Company-level hurdle:")
for ticker, row in latest.iterrows():
    print(
        f"{ticker:5s} capex=${row['capex_abs'] / 1000:6.1f}B  "
        f"op_margin={row['operating_margin']:5.1%}  "
        f"required_rev=${row['required_revenue'] / 1000:6.1f}B  "
        f"actual_growth=${row['revenue_growth_dollars'] / 1000:6.1f}B  "
        f"coverage={row['coverage_ratio']:4.2f}x"
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
x = range(len(latest))
ax.bar([i - 0.18 for i in x], latest["required_revenue"] / 1000, width=0.36, color="#3b82f6", label="Required revenue")
ax.bar([i + 0.18 for i in x], latest["revenue_growth_dollars"] / 1000, width=0.36, color="#22c55e", label="Actual revenue growth")
ax.axhline(0, color="#e0e0e0", linewidth=0.8)
ax.set_title("AI Capex Payback Hurdle")
ax.set_xlabel("Company")
ax.set_ylabel("Dollars, billions")
ax.set_xticks(list(x))
ax.set_xticklabels(latest.index.tolist())
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
