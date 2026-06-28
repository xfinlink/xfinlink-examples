# Full write-up: https://xfinlink.com/blog/bond-returns-granger-causality-stock-python
import xfinlink as xfl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# ── Data retrieval ───────────────────────────────────────────────────
tickers = ["SPY", "TLT", "IEF", "SHY", "AGG"]
df = xfl.prices(tickers, period="7y", interval="1mo", fields=["close"])

# ── Compute monthly returns ──────────────────────────────────────────
returns = {}
for t in tickers:
    sub = df[df["ticker"] == t].sort_values("date").copy()
    sub["return"] = sub["close"].pct_change()
    sub = sub.dropna(subset=["return"])
    returns[t] = sub[["date", "return"]].set_index("date")
    returns[t].columns = [t]

merged = returns["SPY"]
for t in tickers[1:]:
    merged = merged.join(returns[t], how="inner")

print(f"Observation period: {merged.index.min().strftime('%Y-%m')} to "
      f"{merged.index.max().strftime('%Y-%m')}")
print(f"Monthly observations: {len(merged)}\n")

# ── Granger causality: Bond → Stock ─────────────────────────────────
bond_etfs = ["TLT", "IEF", "SHY", "AGG"]
max_lag = 4

print(f"{'Bond ETF':<10} {'Lag':<5} {'F-stat':>10} {'p-value':>10} {'Significant?':>14}")
print("-" * 55)

results_for_chart = []

for bond in bond_etfs:
    test_data = merged[["SPY", bond]].copy()
    gc = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag + 1):
        f_stat = gc[lag][0]["ssr_ftest"][0]
        p_val = gc[lag][0]["ssr_ftest"][1]
        sig = "YES" if p_val < 0.05 else "no"
        print(f"{bond:<10} {lag:<5} {f_stat:>10.3f} {p_val:>10.4f} {sig:>14}")
        results_for_chart.append({"bond": bond, "lag": lag, "f_stat": f_stat, "p_val": p_val})

# ── Reverse test: Stock → Bond ──────────────────────────────────────
print(f"\nReverse test: Do stock returns Granger-cause bond returns?")
print(f"{'Bond ETF':<10} {'Lag':<5} {'F-stat':>10} {'p-value':>10} {'Significant?':>14}")
print("-" * 55)

for bond in bond_etfs:
    test_data_rev = merged[[bond, "SPY"]].copy()
    gc_rev = grangercausalitytests(test_data_rev, maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag + 1):
        f_stat = gc_rev[lag][0]["ssr_ftest"][0]
        p_val = gc_rev[lag][0]["ssr_ftest"][1]
        sig = "YES" if p_val < 0.05 else "no"
        print(f"{bond:<10} {lag:<5} {f_stat:>10.3f} {p_val:>10.4f} {sig:>14}")

# ── Chart ────────────────────────────────────────────────────────────
chart_df = pd.DataFrame(results_for_chart)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

bar_width = 0.18
lags = sorted(chart_df["lag"].unique())
x = np.arange(len(bond_etfs))

for i, lag in enumerate(lags):
    subset = chart_df[chart_df["lag"] == lag]
    p_vals = [subset[subset["bond"] == b]["p_val"].values[0] for b in bond_etfs]
    ax.bar(x + i * bar_width, p_vals, bar_width, label=f"Lag {lag}",
           color=plt.cm.Blues(0.4 + 0.15 * i), edgecolor="none")

ax.axhline(y=0.05, color="#ef4444", linestyle="--", linewidth=1.2, label="5% significance")
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(bond_etfs, color="#e0e0e0", fontsize=11)
ax.set_ylabel("p-value", color="#e0e0e0", fontsize=11)
ax.set_title("Granger Causality: Bond Returns → Stock Returns",
             color="#e0e0e0", fontsize=13, fontweight="bold")
ax.tick_params(colors="#e0e0e0")
ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="#e0e0e0", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#333")
ax.spines["left"].set_color("#333")
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("bond-returns-granger-causality-stock-python.png", dpi=150, facecolor="#0a0a0a")
print("\nChart saved to bond-returns-granger-causality-stock-python.png")
