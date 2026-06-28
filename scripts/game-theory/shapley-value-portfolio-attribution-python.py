# Full write-up: https://xfinlink.com/blog/shapley-value-portfolio-attribution-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from itertools import combinations
from math import factorial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "UNH"]
n = len(tickers)

# Pull 1 year of daily returns
df = xfl.prices(tickers, period="1y", fields=["return_daily"])
ret = df.pivot_table(index="date", columns="ticker", values="return_daily").dropna()
weights = {t: 1.0 / n for t in tickers}


def coalition_return(members):
    """Compound return of an equal-weight sub-portfolio of `members`."""
    if not members:
        return 0.0
    daily = ret[list(members)].mean(axis=1)
    return (1 + daily).prod() - 1


full_return = coalition_return(tickers)

# ---------- Shapley values ----------
shapley = {}
for i in tickers:
    sv = 0.0
    others = [t for t in tickers if t != i]
    for size in range(0, n):
        for S in combinations(others, size):
            S_set = set(S)
            v_with = coalition_return(list(S_set | {i}))
            v_without = coalition_return(list(S_set))
            marginal = v_with - v_without
            weight = factorial(len(S_set)) * factorial(n - len(S_set) - 1) / factorial(n)
            sv += weight * marginal
    shapley[i] = sv

# ---------- Naive attribution ----------
naive = {t: weights[t] * ((1 + ret[t]).prod() - 1) for t in tickers}

# ---------- Results table ----------
results = pd.DataFrame({
    "Ticker": tickers,
    "Shapley Attribution": [shapley[t] for t in tickers],
    "Naive Attribution": [naive[t] for t in tickers],
    "Difference (pp)": [(shapley[t] - naive[t]) * 100 for t in tickers],
})
results = results.sort_values("Shapley Attribution", ascending=False).reset_index(drop=True)

print("=" * 72)
print("SHAPLEY VALUE vs NAIVE WEIGHT-BASED ATTRIBUTION")
print("=" * 72)
for _, row in results.iterrows():
    print(
        f"{row['Ticker']:>5}   Shapley: {row['Shapley Attribution']*100:>+7.2f}%   "
        f"Naive: {row['Naive Attribution']*100:>+7.2f}%   "
        f"Diff: {row['Difference (pp)']:>+6.2f} pp"
    )

shapley_sum = sum(shapley.values())
naive_sum = sum(naive.values())
print(
    f"\n{'SUM':>5}   Shapley: {shapley_sum*100:>+7.2f}%   "
    f"Naive: {naive_sum*100:>+7.2f}%   "
    f"Portfolio: {full_return*100:>+7.2f}%"
)

# ---------- Chart ----------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

x = np.arange(len(results))
width = 0.35

ax.bar(
    x - width / 2,
    results["Shapley Attribution"] * 100,
    width,
    label="Shapley Attribution",
    color="#3b82f6",
    edgecolor="none",
)
ax.bar(
    x + width / 2,
    results["Naive Attribution"] * 100,
    width,
    label="Naive Attribution",
    color="#6b7280",
    edgecolor="none",
)

ax.set_xlabel("Stock", color="#e0e0e0", fontsize=11)
ax.set_ylabel("Contribution to Portfolio Return (%)", color="#e0e0e0", fontsize=11)
ax.set_title(
    "Shapley Value vs Naive Weight-Based Attribution", color="#e0e0e0", fontsize=13
)
ax.set_xticks(x)
ax.set_xticklabels(results["Ticker"], color="#e0e0e0", fontsize=10)
ax.tick_params(axis="y", colors="#e0e0e0")
ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#e0e0e0")
ax.axhline(y=0, color="#333333", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333333")
ax.spines["bottom"].set_color("#333333")

plt.tight_layout()
plt.savefig(
    "shapley-value-portfolio-attribution-python.png", dpi=150, facecolor="#0a0a0a"
)
print("\nChart saved.")
plt.close()
