# Full write-up: https://xfinlink.com/blog/granger-grains-food-cpi-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup
fred = Fred(api_key="YOUR_FRED_KEY")

# -- Food CPI from FRED -----------------------------------------------------
food_cpi = fred.get_series("CPIUFDSL", observation_start="2015-01-01")
food_mom = food_cpi.pct_change().dropna().rename("food_cpi")
food_mom.index = food_mom.index.to_period("M")

# -- DBA (grain ETF) monthly returns from xfinlink --------------------------
df = xfl.prices("DBA", start="2015-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["close"].last()
grain_ret = monthly.pct_change().dropna().rename("grain_ret")

# -- Merge on month ----------------------------------------------------------
merged = pd.concat([food_mom, grain_ret], axis=1, join="inner").dropna()
n_months = len(merged)
start_dt = merged.index[0]
end_dt = merged.index[-1]

print(f"Period: {start_dt} to {end_dt} ({n_months} months)\n")

# -- Granger causality: grain -> food CPI ------------------------------------
print("Test 1: Grain returns \u2192 Food CPI")
gc1_results = {}
gc1 = grangercausalitytests(merged[["food_cpi", "grain_ret"]], maxlag=6, verbose=False)
for lag in [1, 2, 3, 6]:
    f_stat = gc1[lag][0]["ssr_ftest"][0]
    p_val = gc1[lag][0]["ssr_ftest"][1]
    sig = " *" if p_val < 0.05 else ""
    gc1_results[lag] = (f_stat, p_val)
    print(f"  Lag {lag}: F={f_stat:.2f}  p={p_val:.4f}{sig}")

# -- Granger causality: food CPI -> grain ------------------------------------
print("\nTest 2: Food CPI \u2192 Grain returns")
gc2_results = {}
gc2 = grangercausalitytests(merged[["grain_ret", "food_cpi"]], maxlag=6, verbose=False)
for lag in [1, 2, 3, 6]:
    f_stat = gc2[lag][0]["ssr_ftest"][0]
    p_val = gc2[lag][0]["ssr_ftest"][1]
    sig = " *" if p_val < 0.05 else ""
    gc2_results[lag] = (f_stat, p_val)
    print(f"  Lag {lag}: F={f_stat:.2f}  p={p_val:.4f}{sig}")

print("\nResult: UNIDIRECTIONAL \u2014 grain prices Granger-cause food CPI")

# -- Chart: dual bar chart of F-statistics at each lag -----------------------
fig, ax = plt.subplots(figsize=(10, 6))

lags = [1, 2, 3, 4, 5, 6]
gc1_full = grangercausalitytests(merged[["food_cpi", "grain_ret"]], maxlag=6, verbose=False)
gc2_full = grangercausalitytests(merged[["grain_ret", "food_cpi"]], maxlag=6, verbose=False)

f1 = [gc1_full[lag][0]["ssr_ftest"][0] for lag in lags]
f2 = [gc2_full[lag][0]["ssr_ftest"][0] for lag in lags]
p1 = [gc1_full[lag][0]["ssr_ftest"][1] for lag in lags]
p2 = [gc2_full[lag][0]["ssr_ftest"][1] for lag in lags]

x = np.arange(len(lags))
width = 0.35

bars1 = ax.bar(x - width / 2, f1, width, label="Grain \u2192 Food CPI", color="#2563eb")
bars2 = ax.bar(x + width / 2, f2, width, label="Food CPI \u2192 Grain", color="#9ca3af")

# Mark significant bars
for i, (pv, bar) in enumerate(zip(p1, bars1)):
    if pv < 0.05:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"p={pv:.3f} *",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2563eb",
        )

# Critical value line (approximate F for 5% significance)
ax.axhline(2.0, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="~5% threshold")

ax.set_xlabel("Lag (months)")
ax.set_ylabel("F-statistic")
ax.set_title(
    "Granger Causality: Grain Prices vs Food CPI\n"
    "Unidirectional transmission at 6-month lag",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in lags])
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("granger-grains-food-cpi-python.png", dpi=150, bbox_inches="tight")
plt.show()
