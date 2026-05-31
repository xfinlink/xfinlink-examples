# Full write-up: https://xfinlink.com/blog/ai-earnings-cash-flow-quality-python

import os

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-earnings-cash-flow-quality-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

TICKERS = ["NVDA", "ORCL", "MSFT", "AMZN", "META", "GOOG", "AVGO", "PLTR"]
FIELDS = [
    "revenue",
    "net_income",
    "operating_cash_flow",
    "free_cash_flow",
    "capital_expenditures",
    "total_assets",
    "stock_based_compensation_cf",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.fundamentals(TICKERS, period_type="annual", period="4y", fields=FIELDS)
df = df[df["ticker"].isin(TICKERS)].copy()

require(not df.empty, "fundamentals returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

latest = df.sort_values("period_end").groupby("ticker").tail(1).copy()
require(latest[FIELDS].notna().all().all(), "latest annual data contains missing values")
require((latest["revenue"] > 0).all(), "revenue must be positive")
require((latest["net_income"] > 0).all(), "net income must be positive for this screen")
require((latest["operating_cash_flow"] > 0).all(), "operating cash flow must be positive")

latest["capex_abs"] = latest["capital_expenditures"].abs()
latest["cash_conversion"] = latest["operating_cash_flow"] / latest["net_income"]
latest["fcf_margin"] = latest["free_cash_flow"] / latest["revenue"]
latest["capex_to_ocf"] = latest["capex_abs"] / latest["operating_cash_flow"]
latest["accrual_ratio"] = (latest["net_income"] - latest["operating_cash_flow"]) / latest["total_assets"]
latest["sbc_to_revenue"] = latest["stock_based_compensation_cf"] / latest["revenue"]

latest = latest.sort_values("capex_to_ocf", ascending=False)

print("=== AI Earnings Cash-Flow Quality Screen ===")
print(f"Latest annual period per company (max period_end {latest['period_end'].max().date()})")
print()
for _, row in latest.iterrows():
    print(
        f"{row['ticker']:5s} OCF/net_income={row['cash_conversion']:4.2f}x  "
        f"FCF_margin={row['fcf_margin']:6.1%}  "
        f"capex/OCF={row['capex_to_ocf']:5.1%}  "
        f"accrual_ratio={row['accrual_ratio']:+6.1%}  "
        f"SBC/revenue={row['sbc_to_revenue']:5.1%}"
    )

strong_cash = latest[latest["cash_conversion"] >= 1.2]
weak_fcf = latest[latest["fcf_margin"] < 0]
print()
print(f"Companies with OCF/net income >= 1.2x: {', '.join(strong_cash['ticker'])}")
print(f"Companies with negative free cash flow margin: {', '.join(weak_fcf['ticker']) if len(weak_fcf) else 'none'}")

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
colors = ["#ef4444" if value < 0 else "#3b82f6" for value in latest["fcf_margin"]]
ax.scatter(latest["capex_to_ocf"] * 100, latest["cash_conversion"], s=95, color=colors, edgecolor="#e0e0e0", linewidth=0.8)
ax.axhline(1.0, color="#f59e0b", linewidth=1.2, linestyle="--", label="OCF = net income")
ax.axvline(50.0, color="#666666", linewidth=1.0, linestyle="--", label="Capex = 50% of OCF")

for _, row in latest.iterrows():
    ax.text(row["capex_to_ocf"] * 100 + 1.3, row["cash_conversion"] + 0.015, row["ticker"], fontsize=9)

ax.set_title("AI Earnings Quality: Cash Conversion vs Capex Burden")
ax.set_xlabel("Capital expenditures / operating cash flow (%)")
ax.set_ylabel("Operating cash flow / net income")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
