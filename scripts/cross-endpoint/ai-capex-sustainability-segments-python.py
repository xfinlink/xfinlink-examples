# Full write-up: https://xfinlink.com/blog/ai-capex-sustainability-segments-python

import xfinlink as xfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Fetch annual fundamentals with segments -----------------------------------
tickers = ["AAPL", "MSFT", "AMZN", "META", "NVDA"]
df = xfl.fundamentals(
    tickers, period_type="annual", period="3y",
    fields=["revenue", "capital_expenditure", "free_cash_flow", "operating_cash_flow"],
    include_segments=True,
)

# -- Keep latest two annual periods per ticker for YoY comparison --------------
recent = df.sort_values("period_end").groupby("ticker").tail(2)
latest = df.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")
prior = df.sort_values("period_end").groupby("ticker").apply(
    lambda g: g.iloc[-2] if len(g) >= 2 else None
).dropna(how="all")
if "ticker" in prior.columns:
    prior = prior.set_index("ticker")

# -- Compute metrics -----------------------------------------------------------
latest["capex_abs"] = latest["capital_expenditure"].abs()
latest["capex_fcf"] = (latest["capex_abs"] / latest["free_cash_flow"]).round(1)
latest["capex_intensity"] = (latest["capex_abs"] / latest["revenue"] * 100).round(1)

# Capex YoY
if len(prior) > 0:
    prior["capex_abs"] = prior["capital_expenditure"].abs()
    latest["capex_yoy"] = (
        (latest["capex_abs"] - prior["capex_abs"]) / prior["capex_abs"] * 100
    ).round(0)

# -- Classify sustainability ---------------------------------------------------
def classify(ratio):
    if ratio < 1.0:
        return "SUSTAINABLE"
    elif ratio < 3.0:
        return "SUSTAINABLE"
    else:
        return "UNSUSTAINABLE"

latest["status"] = latest["capex_fcf"].apply(classify)

# -- Print results -------------------------------------------------------------
print("=== AI Capex Race: Who Is Funding It Sustainably? ===\n")
for ticker in tickers:
    row = latest.loc[ticker]
    rev = row["revenue"] / 1e9
    capex = row["capex_abs"] / 1e9
    fcf = row["free_cash_flow"] / 1e9
    ratio = row["capex_fcf"]
    intensity = row["capex_intensity"]
    status = row["status"]
    yoy = row.get("capex_yoy", float("nan"))
    print(
        f"{ticker:6s} rev=${rev:.0f}B  capex=${capex:.0f}B  FCF=${fcf:.0f}B  "
        f"capex/FCF={ratio}x  intensity={intensity}%  [{status}]"
    )
    if not np.isnan(yoy):
        print(f"       capex YoY: {yoy:+.0f}%")
    print()

# -- Chart: grouped bar chart --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: capex/FCF ratio
colors = ["#16a34a" if latest.loc[t, "status"] == "SUSTAINABLE" else "#dc2626" for t in tickers]
bars1 = ax1.bar(tickers, [latest.loc[t, "capex_fcf"] for t in tickers], color=colors, edgecolor="white")
ax1.axhline(1.0, color="#f59e0b", linestyle="--", linewidth=1, label="1.0x threshold")
ax1.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1, label="5.0x warning")
ax1.set_ylabel("Capex / Free Cash Flow (x)")
ax1.set_title("AI Capex Sustainability: Capex/FCF Ratio", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

for bar, ticker in zip(bars1, tickers):
    val = latest.loc[ticker, "capex_fcf"]
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val}x", ha="center", va="bottom", fontweight="bold", fontsize=11)

# Right: capex intensity
bars2 = ax2.bar(tickers, [latest.loc[t, "capex_intensity"] for t in tickers],
                color="#2563eb", edgecolor="white")
ax2.set_ylabel("Capex / Revenue (%)")
ax2.set_title("Capex Intensity (% of Revenue)", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

for bar, ticker in zip(bars2, tickers):
    val = latest.loc[ticker, "capex_intensity"]
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("ai-capex-sustainability-segments-python.png", dpi=150, bbox_inches="tight")
plt.show()
