# Full write-up: https://xfinlink.com/blog/stock-return-skewness-python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

# The market (SPY) and sixteen of its larger members. A long window is
# deliberate: skewness only settles once the sample contains real drawdowns.
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "JNJ",
           "XOM", "PG", "KO", "HD", "WMT", "UNH", "V", "CVX", "CAT"]

px = xfl.prices(TICKERS, start="2016-01-01", end="2026-08-21",
                fields=["return_daily"])
px["date"] = pd.to_datetime(px["date"])

rows = []
for t in TICKERS:
    sub = px[px["ticker"] == t][["date", "return_daily"]].dropna()
    daily = sub["return_daily"]
    monthly = sub.set_index("date")["return_daily"].add(1).resample("ME").prod().sub(1)
    rows.append({"ticker": t,
                 "daily_skew": skew(daily),
                 "monthly_skew": skew(monthly.dropna())})

d = pd.DataFrame(rows)
spy = d[d["ticker"] == "SPY"].iloc[0]
stocks = d[d["ticker"] != "SPY"]

print(d.sort_values("daily_skew").round(3).to_string(index=False))
print()
print(f"SPY daily skew:   {spy['daily_skew']:.3f}   monthly skew: {spy['monthly_skew']:.3f}")
print(f"single stocks, median daily skew: {stocks['daily_skew'].median():.3f}, "
      f"mean: {stocks['daily_skew'].mean():.3f}")
print(f"stocks less negatively skewed than SPY (daily):   "
      f"{(stocks['daily_skew'] > spy['daily_skew']).sum()} of {len(stocks)}")
print(f"stocks less negatively skewed than SPY (monthly): "
      f"{(stocks['monthly_skew'] > spy['monthly_skew']).sum()} of {len(stocks)}")

# ---- chart ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333",
})
ds = d.sort_values("daily_skew")
colors = ["#ef4444" if t == "SPY" else "#3b82f6" for t in ds["ticker"]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(ds["ticker"], ds["daily_skew"], color=colors)
ax.axvline(0, color="#e0e0e0", linewidth=1)
ax.set_xlabel("Skewness of daily returns, 2016 to 2026")
ax.set_title("The index is more downside-skewed than most of its members")
ax.invert_yaxis()
ax.text(spy["daily_skew"] - 0.05, list(ds["ticker"]).index("SPY"),
        "SPY ", color="#ef4444", fontsize=9, ha="right", va="center")
plt.tight_layout()
plt.savefig("stock-return-skewness-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart saved")
