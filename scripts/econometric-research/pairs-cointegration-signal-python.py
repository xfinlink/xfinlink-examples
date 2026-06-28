# Full write-up: https://xfinlink.com/blog/pairs-cointegration-signal-python
import xfinlink as xfl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# ── Data ─────────────────────────────────────────────────────────────────
tickers = ["KO", "PEP", "CVX", "XOM", "HD", "LOW"]
df = xfl.prices(tickers, period="5y", fields=["adj_close"])

pairs = [("KO", "PEP"), ("CVX", "XOM"), ("HD", "LOW")]

results = []
pair_data = {}

for t1, t2 in pairs:
    s1 = df[df["ticker"] == t1].sort_values("date").set_index("date")["adj_close"]
    s2 = df[df["ticker"] == t2].sort_values("date").set_index("date")["adj_close"]

    # Align on common dates
    aligned = pd.concat([s1, s2], axis=1, keys=[t1, t2]).dropna()

    # OLS regression: t2 = alpha + beta * t1
    X = add_constant(aligned[t1].values)
    model = OLS(aligned[t2].values, X).fit()
    hedge_ratio = model.params[1]

    # Compute spread
    spread = aligned[t2].values - hedge_ratio * aligned[t1].values

    # ADF test on spread
    adf_result = adfuller(spread, autolag="AIC")
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    coint_tag = "COINTEGRATED" if p_value < 0.05 else "NOT COINTEGRATED"

    # Z-score
    z = (spread - spread.mean()) / spread.std()

    results.append({
        "pair": f"{t1}/{t2}",
        "hedge_ratio": hedge_ratio,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "tag": coint_tag,
        "z_mean": z.mean(),
        "z_std": z.std(),
        "z_current": z[-1]
    })

    pair_data[(t1, t2)] = {
        "dates": aligned.index,
        "z": z
    }

# ── Output ───────────────────────────────────────────────────────────────
print(f"{'Pair':<10} {'Hedge Ratio':>12} {'ADF Stat':>10} {'p-value':>9} {'Status':<20} {'Z (now)':>8}")
print("-" * 75)
for r in results:
    print(f"{r['pair']:<10} {r['hedge_ratio']:>12.4f} {r['adf_stat']:>10.4f} {r['p_value']:>9.4f} {r['tag']:<20} {r['z_current']:>8.2f}")

print()
for r in results:
    print(f"{r['pair']} z-score stats: mean={r['z_mean']:.4f}, std={r['z_std']:.4f}")

# ── Chart ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 7), facecolor="#0a0a0a")
colors = ["#3b82f6", "#10b981", "#f59e0b"]

for i, ((t1, t2), color) in enumerate(zip(pairs, colors)):
    ax = axes[i]
    ax.set_facecolor("#0a0a0a")
    d = pair_data[(t1, t2)]
    ax.plot(d["dates"], d["z"], color=color, linewidth=0.8, label=f"{t1}/{t2}")
    ax.axhline(y=2, color="#ef4444", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(y=-2, color="#ef4444", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(y=0, color="#555555", linestyle="-", linewidth=0.5)

    status = [r for r in results if r["pair"] == f"{t1}/{t2}"][0]
    tag_short = "Coint." if status["tag"] == "COINTEGRATED" else "Not coint."
    ax.set_title(f"{t1}/{t2} (p={status['p_value']:.3f}, {tag_short})",
                 color="#e0e0e0", fontsize=11, pad=6)
    ax.set_ylabel("Z-score", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#e0e0e0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")

axes[-1].set_xlabel("Date", color="#e0e0e0", fontsize=9)
fig.suptitle("Pairs Trading Z-Scores: Cointegration Signal", color="#e0e0e0",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("pairs-cointegration-signal-python.png",
            dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
print("\nChart saved.")
