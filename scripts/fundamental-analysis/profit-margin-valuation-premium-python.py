# Full write-up: https://xfinlink.com/blog/profit-margin-valuation-premium-python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

# Large non-financial companies. Enterprise-value-to-sales is not meaningful
# for banks, insurers, or property trusts, so the sample excludes them.
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL",
           "ADBE", "CRM", "CSCO", "TXN", "QCOM", "AMD", "INTC", "JNJ", "UNH",
           "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "PG", "KO", "PEP",
           "COST", "WMT", "MCD", "NKE", "SBUX", "XOM", "CVX", "CAT", "HON",
           "DE", "LIN"]

m = xfl.metrics(TICKERS, period_type="annual", period="2y",
                fields=["operating_margin", "ev_revenue"])

# Keep the latest completed fiscal year per company, drop any without both
# figures for that year.
latest = m.sort_values("period_end").groupby("ticker", as_index=False).last()
d = latest.dropna(subset=["operating_margin", "ev_revenue"]).copy()
d["op_margin"] = d["operating_margin"] * 100

r = np.corrcoef(d["op_margin"], d["ev_revenue"])[0, 1]
slope, intercept = np.polyfit(d["op_margin"], d["ev_revenue"], 1)

print(f"{len(d)} companies")
print(f"correlation of operating margin with EV/Sales: {r:.3f}  (r^2 = {r**2:.3f})")
print(f"regression: EV/Sales = {intercept:.2f} + {slope:.3f} x operating margin(%)")
print()
print(d.sort_values("op_margin")[["ticker", "op_margin", "ev_revenue"]]
        .round(2).to_string(index=False))

# ---- chart ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333",
})
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(d["op_margin"], d["ev_revenue"], s=45, color="#3b82f6",
           edgecolor="#93c5fd", linewidth=0.5, zorder=3)
xs = np.linspace(d["op_margin"].min(), d["op_margin"].max(), 50)
ax.plot(xs, intercept + slope * xs, color="#e0e0e0", linewidth=1.2,
        linestyle="--", zorder=2, label=f"fit (r² = {r**2:.2f})")
for _, row in d.iterrows():
    if row["ticker"] in {"NVDA", "AVGO", "AMD", "ADBE", "INTC", "MRK",
                          "WMT", "MSFT", "META"}:
        ax.annotate(row["ticker"], (row["op_margin"], row["ev_revenue"]),
                    xytext=(5, 4), textcoords="offset points",
                    color="#9ca3af", fontsize=9)
ax.set_xlabel("Operating margin (%)")
ax.set_ylabel("Enterprise value to sales")
ax.set_title("Higher margins earn a higher sales multiple")
ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")
plt.tight_layout()
plt.savefig("profit-margin-valuation-premium-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart saved")
