# Full write-up: https://xfinlink.com/blog/oil-gold-ratio-recession-signal-python

import xfinlink as xfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Fetch 10 years of XLE, GLD, SPY ------------------------------------------
tickers = ["XLE", "GLD", "SPY"]
df = xfl.prices(tickers, start="2016-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])

# -- Pivot to wide format ------------------------------------------------------
wide = df.pivot_table(index="date", columns="ticker", values="close")
wide = wide.dropna()

# -- Compute XLE/GLD ratio and 12-month rate of change ------------------------
wide["ratio"] = wide["XLE"] / wide["GLD"]
wide["ratio_roc_12m"] = wide["ratio"].pct_change(252)

# -- Forward 6-month SPY return ------------------------------------------------
wide["fwd_6m"] = wide["SPY"].pct_change(126).shift(-126)

# -- Drop NaN and bin into quintiles -------------------------------------------
analysis = wide.dropna(subset=["ratio_roc_12m", "fwd_6m"]).copy()
analysis["quintile"] = pd.qcut(
    analysis["ratio_roc_12m"], 5,
    labels=["Q1 (Falling)", "Q2", "Q3", "Q4", "Q5 (Rising)"],
)

# -- Print results -------------------------------------------------------------
print(f"Current XLE/GLD ratio: {wide['ratio'].iloc[-1]:.3f}")
print(f"12-month change: {wide['ratio_roc_12m'].iloc[-1]*100:+.1f}%\n")

for q in ["Q1 (Falling)", "Q2", "Q3", "Q4", "Q5 (Rising)"]:
    sub = analysis[analysis["quintile"] == q]
    mean_ret = sub["fwd_6m"].mean() * 100
    n = len(sub)
    print(f"  {q:14s}  fwd_ret={mean_ret:+.1f}%  n={n}")

# -- Chart: bar chart of forward returns by quintile ---------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 2]})

# Top panel: XLE/GLD ratio over time
ax1.plot(wide.index, wide["ratio"], color="#2563eb", linewidth=1.5)
ax1.set_ylabel("XLE / GLD Ratio")
ax1.set_title(
    "Oil-to-Gold Ratio (XLE/GLD) as a Contrarian Equity Signal\n"
    "10 years of daily data, quintile analysis of 12-month rate of change",
    fontsize=13,
    fontweight="bold",
)
ax1.grid(True, alpha=0.3)

# Bottom panel: bar chart of mean forward 6m return by quintile
quintiles = ["Q1 (Falling)", "Q2", "Q3", "Q4", "Q5 (Rising)"]
means = []
for q in quintiles:
    sub = analysis[analysis["quintile"] == q]
    means.append(sub["fwd_6m"].mean() * 100)

colors = ["#16a34a" if m > 0 else "#dc2626" for m in means]
bars = ax2.bar(quintiles, means, color=colors, edgecolor="white", width=0.6)
ax2.set_ylabel("Mean Forward 6-Month SPY Return (%)")
ax2.set_xlabel("XLE/GLD 12-Month Rate of Change Quintile")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.grid(True, alpha=0.3, axis="y")

for bar, m in zip(bars, means):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
        f"{m:+.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11,
    )

plt.tight_layout()
plt.savefig("oil-gold-ratio-recession-signal-python.png", dpi=150, bbox_inches="tight")
plt.show()
