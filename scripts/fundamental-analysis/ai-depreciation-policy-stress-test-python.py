# Full write-up: https://xfinlink.com/blog/ai-depreciation-policy-stress-test-python

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-depreciation-policy-stress-test-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

TICKERS = ["AMZN", "MSFT", "GOOG", "META", "ORCL", "NVDA", "AVGO", "TSM"]
FIELDS = [
    "revenue",
    "operating_income",
    "capital_expenditures",
    "depreciation_amortization",
    "property_plant_equipment_net",
]
DEPRECIATION_STRESS = 0.25


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.fundamentals(TICKERS, period_type="annual", period="5y", fields=FIELDS)
df = df[df["ticker"].isin(TICKERS)].sort_values(["ticker", "period_end"]).copy()

require(not df.empty, "fundamentals returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

latest = df.groupby("ticker", group_keys=False).tail(1).set_index("ticker").copy()
require(latest[FIELDS].notna().all().all(), "latest annual fundamentals contain missing values")
require((latest["revenue"] > 0).all(), "latest revenue must be positive")
require((latest["operating_income"] > 0).all(), "latest operating income must be positive")
require((latest["depreciation_amortization"] > 0).all(), "depreciation must be positive")

latest["capex_abs"] = latest["capital_expenditures"].abs()
latest["capex_to_depreciation"] = latest["capex_abs"] / latest["depreciation_amortization"]
latest["implied_ppe_life"] = latest["property_plant_equipment_net"] / latest["depreciation_amortization"]
latest["extra_depreciation"] = latest["depreciation_amortization"] * DEPRECIATION_STRESS
latest["operating_income_after_stress"] = latest["operating_income"] - latest["extra_depreciation"]
latest["operating_income_hit"] = latest["extra_depreciation"] / latest["operating_income"]

latest = latest.sort_values("operating_income_hit", ascending=False)

aggregate_capex_to_depreciation = latest["capex_abs"].sum() / latest["depreciation_amortization"].sum()
aggregate_income_hit = latest["extra_depreciation"].sum() / latest["operating_income"].sum()

print("=== AI Depreciation Policy Stress Test ===")
print(f"Stress case: depreciation and amortization rises {DEPRECIATION_STRESS:.0%}")
print(f"Latest annual periods: {latest['period_end'].min().date()} to {latest['period_end'].max().date()}")
print()
print(f"Combined capex / depreciation: {aggregate_capex_to_depreciation:.1f}x")
print(f"Aggregate operating-income hit: {aggregate_income_hit:.1%}")
print()
print("Company-level depreciation sensitivity:")
for ticker, row in latest.iterrows():
    print(
        f"{ticker:5s} capex/depr={row['capex_to_depreciation']:4.1f}x  "
        f"implied_PPE_life={row['implied_ppe_life']:4.1f}y  "
        f"depr=${row['depreciation_amortization'] / 1000:5.1f}B  "
        f"op_income_hit={row['operating_income_hit']:5.1%}  "
        f"stressed_op_income=${row['operating_income_after_stress'] / 1000:6.1f}B"
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
colors = ["#ef4444" if value > 0.10 else "#3b82f6" for value in latest["operating_income_hit"]]
bars = ax.bar(latest.index, latest["operating_income_hit"] * 100, color=colors)
ax.axhline(10, color="#f59e0b", linewidth=1.2, linestyle="--", label="10% operating-income hit")
ax.set_title("AI Depreciation Policy Stress Test")
ax.set_xlabel("Company")
ax.set_ylabel("Operating income reduction (%)")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
for bar, value in zip(bars, latest["operating_income_hit"] * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
