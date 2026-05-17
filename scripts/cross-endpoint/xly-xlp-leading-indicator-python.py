# Full write-up: https://xfinlink.com/blog/xly-xlp-leading-indicator-python

import xfinlink as xfl
import pandas as pd
import numpy as np

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
tickers = ["XLY", "XLP", "SPY"]

# -- Fetch 5Y daily prices -------------------------------------------------
df = xfl.prices(tickers, period="5y", fields=["close"])

# -- Pivot to wide format ---------------------------------------------------
wide = df.pivot_table(index="date", columns="ticker", values="close").dropna()

# -- Compute XLY/XLP ratio -------------------------------------------------
wide["ratio"] = wide["XLY"] / wide["XLP"]
wide["ratio_sma200"] = wide["ratio"].rolling(200).mean()
wide["ratio_std200"] = wide["ratio"].rolling(200).std()
wide["z_score"] = (wide["ratio"] - wide["ratio_sma200"]) / wide["ratio_std200"]
wide["ratio_sma50"] = wide["ratio"].rolling(50).mean()

# -- Forward 3-month SPY return (63 trading days) ---------------------------
wide["fwd_3m_ret"] = wide["SPY"].shift(-63) / wide["SPY"] - 1

# -- Drop rows missing z-score or forward return ---------------------------
analysis = wide.dropna(subset=["z_score", "fwd_3m_ret"]).copy()

# -- Quintile analysis ------------------------------------------------------
analysis["quintile"] = pd.qcut(analysis["z_score"], 5, labels=False) + 1

print("=== XLY/XLP Ratio: Current State ===")
latest = wide.dropna(subset=["z_score"]).iloc[-1]
pctl = (wide["ratio"].dropna() <= latest["ratio"]).mean() * 100
print(f"Current ratio:             {latest['ratio']:.3f}")
print(f"200-day SMA:               {latest['ratio_sma200']:.3f}")
print(f"Current z-score:          {latest['z_score']:.2f}")
print(f"Percentile (5Y):          {pctl:.1f}%")

print("\n=== Forward 3-Month SPY Return by XLY/XLP Z-Score Quintile ===")
header = f"{'Quintile':12s}  {'Z-Score Range':22s}  {'Avg Fwd 3M Return':>18s}  {'N obs':>6s}"
print(header)
print("-" * 60)

labels = {1: "Q1 (low)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (high)"}
for q in range(1, 6):
    subset = analysis[analysis["quintile"] == q]
    z_min = subset["z_score"].min()
    z_max = subset["z_score"].max()
    avg_ret = subset["fwd_3m_ret"].mean()
    n = len(subset)
    label = labels[q]
    print(
        f"{label:12s}  {z_min:>6.2f} to {z_max:>5.2f}        "
        f"{avg_ret:>+5.1%}              {n:>4d}"
    )

# -- Current signal ---------------------------------------------------------
print("\n=== Current Signal ===")
current_q = "Q1" if latest["z_score"] <= analysis[analysis["quintile"] == 1]["z_score"].max() else "Q2-Q3"
print(f"Current z-score ({latest['z_score']:.2f}) is in {current_q} territory.")
print(f"50-day SMA of ratio: {latest['ratio_sma50']:.3f} (current ratio {'below' if latest['ratio'] < latest['ratio_sma50'] else 'above'} 50d SMA)")
regime = "Defensive (staples outperforming discretionary)" if latest["z_score"] < 0 else "Risk-on (discretionary outperforming staples)"
print(f"Regime: {regime}")
