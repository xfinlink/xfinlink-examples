# Full write-up: https://xfinlink.com/blog/apollo-peer-balance-sheet-risk-python

import os

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "apollo-peer-balance-sheet-risk-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

TICKERS = ["APO", "KKR", "BX", "BLK", "ARES", "OWL", "BEN", "TROW"]
FIELDS = [
    "revenue",
    "net_income",
    "operating_cash_flow",
    "total_assets",
    "total_equity",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.fundamentals(TICKERS, period_type="annual", period="3y", fields=FIELDS)
df = df[df["ticker"].isin(TICKERS)].copy()

require(not df.empty, "fundamentals returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

latest = df.sort_values("period_end").groupby("ticker").tail(1).copy()
require(latest[FIELDS].notna().all().all(), "latest annual data contains missing values")
require((latest["revenue"] > 0).all(), "revenue must be positive")
require((latest["total_assets"] > 0).all(), "assets must be positive")
require((latest["total_equity"] > 0).all(), "equity must be positive")

latest["assets_to_equity"] = latest["total_assets"] / latest["total_equity"]
latest["revenue_to_assets"] = latest["revenue"] / latest["total_assets"]
latest["ocf_to_assets"] = latest["operating_cash_flow"] / latest["total_assets"]
latest["net_margin"] = latest["net_income"] / latest["revenue"]
latest = latest.sort_values("assets_to_equity", ascending=False)

apo = latest[latest["ticker"] == "APO"].iloc[0]
median_assets_to_equity = latest[latest["ticker"] != "APO"]["assets_to_equity"].median()
apo_vs_median = apo["assets_to_equity"] / median_assets_to_equity

print("=== Asset-Manager Balance-Sheet Sensitivity Screen ===")
print(f"Latest annual period per company (max period_end {latest['period_end'].max().date()})")
print()
print(f"APO assets/equity: {apo['assets_to_equity']:.1f}x")
print(f"Peer median assets/equity: {median_assets_to_equity:.1f}x")
print(f"APO multiple vs peer median: {apo_vs_median:.1f}x")
print()
print("Balance-sheet ranking:")
for _, row in latest.iterrows():
    print(
        f"{row['ticker']:5s} assets=${row['total_assets'] / 1000:7.1f}B  "
        f"equity=${row['total_equity'] / 1000:6.1f}B  "
        f"assets/equity={row['assets_to_equity']:5.1f}x  "
        f"revenue/assets={row['revenue_to_assets']:5.1%}  "
        f"OCF/assets={row['ocf_to_assets']:5.1%}"
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
colors = ["#3b82f6" if ticker == "APO" else "#3a3a3a" for ticker in latest["ticker"]]
bars = ax.bar(latest["ticker"], latest["assets_to_equity"], color=colors)
ax.axhline(median_assets_to_equity, color="#f59e0b", linewidth=1.2, linestyle="--", label="Peer median")
ax.set_title("Balance-Sheet Intensity Across Asset Managers")
ax.set_xlabel("Company")
ax.set_ylabel("Total assets / total equity")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
for bar, value in zip(bars, latest["assets_to_equity"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}x", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
