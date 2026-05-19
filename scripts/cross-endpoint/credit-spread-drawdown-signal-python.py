# Full write-up: https://xfinlink.com/blog/credit-spread-drawdown-signal-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup
fred = Fred(api_key="YOUR_FRED_KEY")

# -- Fetch BAA and AAA yields from FRED -------------------------------------
baa = fred.get_series("BAA", observation_start="2016-01-01").rename("baa")
aaa = fred.get_series("AAA", observation_start="2016-01-01").rename("aaa")
spreads = pd.concat([baa, aaa], axis=1).dropna()
spreads["spread"] = spreads["baa"] - spreads["aaa"]

# -- Fetch SPY daily prices --------------------------------------------------
spy_df = xfl.prices("SPY", start="2016-01-01", fields=["close"])
spy_df["date"] = pd.to_datetime(spy_df["date"])
spy = spy_df.set_index("date")["close"]

# -- Merge on date -----------------------------------------------------------
merged = pd.concat([spreads["spread"], spy], axis=1).ffill().dropna()

# -- Rolling z-score of spread (252-day) -------------------------------------
merged["spread_sma252"] = merged["spread"].rolling(252).mean()
merged["spread_std252"] = merged["spread"].rolling(252).std()
merged["z"] = (merged["spread"] - merged["spread_sma252"]) / merged["spread_std252"]

# -- Forward 3-month max drawdown (63 trading days) --------------------------
fwd_dd = []
close_vals = merged["close"].values
for i in range(len(close_vals)):
    if i + 63 < len(close_vals):
        window = close_vals[i : i + 63]
        dd = window.min() / window[0] - 1
        fwd_dd.append(dd)
    else:
        fwd_dd.append(np.nan)
merged["fwd_dd"] = fwd_dd

# -- Drop missing and bin into quintiles -------------------------------------
analysis = merged.dropna(subset=["z", "fwd_dd"]).copy()
analysis["quintile"] = pd.qcut(
    analysis["z"], 5, labels=["Q1 (Tight)", "Q2", "Q3", "Q4", "Q5 (Wide)"]
)

# -- Print results -----------------------------------------------------------
print("=== BAA-AAA Credit Spread vs Forward 3-Month SPY Max Drawdown ===\n")
for q in ["Q1 (Tight)", "Q2", "Q3", "Q4", "Q5 (Wide)"]:
    subset = analysis[analysis["quintile"] == q]
    mean_dd = subset["fwd_dd"].mean()
    med_dd = subset["fwd_dd"].median()
    n = len(subset)
    print(f"  {q:14s}  mean_dd={mean_dd:+.1%}  median_dd={med_dd:+.1%}  n={n}")

latest = merged.dropna(subset=["z"]).iloc[-1]
print(f"\nCurrent spread: {latest['spread']:.2f}%  z-score: {latest['z']:.2f}")

# -- Chart: bar chart of mean drawdown by quintile ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))

quintiles = ["Q1 (Tight)", "Q2", "Q3", "Q4", "Q5 (Wide)"]
means = []
medians = []
for q in quintiles:
    subset = analysis[analysis["quintile"] == q]
    means.append(subset["fwd_dd"].mean() * 100)
    medians.append(subset["fwd_dd"].median() * 100)

x = np.arange(len(quintiles))
width = 0.35
bars1 = ax.bar(x - width / 2, means, width, label="Mean Drawdown", color="#dc2626")
bars2 = ax.bar(x + width / 2, medians, width, label="Median Drawdown", color="#f97316")

ax.set_xlabel("BAA-AAA Spread Z-Score Quintile")
ax.set_ylabel("Forward 3-Month Max Drawdown (%)")
ax.set_title(
    "Credit Spread Quintile vs Forward SPY Drawdown\n"
    "Widest spreads predict worst drawdowns",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(quintiles)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 0.3,
        f"{bar.get_height():.1f}%",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

plt.tight_layout()
plt.savefig(
    "credit-spread-drawdown-signal-python.png", dpi=150, bbox_inches="tight"
)
plt.show()
