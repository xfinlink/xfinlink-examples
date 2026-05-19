# Full write-up: https://xfinlink.com/blog/oil-hmm-regime-detection-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Configuration ----------------------------------------------------------
ticker = "XLE"

# -- Fetch 5Y daily prices --------------------------------------------------
df = xfl.prices(ticker, period="5y", fields=["close", "return_daily"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").dropna(subset=["return_daily"]).reset_index(drop=True)
returns = df["return_daily"].values.reshape(-1, 1)

# -- Fit 2-state Gaussian HMM -----------------------------------------------
model = GaussianHMM(
    n_components=2, covariance_type="full", n_iter=200, random_state=42
)
model.fit(returns)
states = model.predict(returns)
df["state"] = states

# -- Identify calm vs crisis by volatility -----------------------------------
vols = [np.sqrt(model.covars_[i][0, 0]) for i in range(2)]
calm_idx = np.argmin(vols)
crisis_idx = np.argmax(vols)

# -- Print regime statistics -------------------------------------------------
for label, idx in [("CALM", calm_idx), ("CRISIS", crisis_idx)]:
    mu = model.means_[idx][0] * 100
    vol = vols[idx] * 100
    ann_vol = vol * np.sqrt(252)
    print(f"State {idx} ({label}):  mean={mu:+.3f}%/day  vol={vol:.3f}%/day  ann_vol={ann_vol:.1f}%")

print(
    f"\nTransition: CALM\u2192CALM={model.transmat_[calm_idx, calm_idx]:.3f}  "
    f"CALM\u2192CRISIS={model.transmat_[calm_idx, crisis_idx]:.3f}  "
    f"CRISIS\u2192CALM={model.transmat_[crisis_idx, calm_idx]:.3f}"
)

# Time in each regime
calm_pct = (states == calm_idx).mean() * 100
crisis_pct = (states == crisis_idx).mean() * 100
print(f"Time in regime: CALM={calm_pct:.0f}%  CRISIS={crisis_pct:.0f}%")
print(f"Current regime: {'CALM' if states[-1] == calm_idx else 'CRISIS'}")

# Recent regime switches
switches = []
for i in range(1, len(states)):
    if states[i] != states[i - 1]:
        new_label = "CRISIS" if states[i] == crisis_idx else "CALM"
        switches.append(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}\u2192{new_label}")
if switches:
    print(f"\nRecent switches: {', '.join(switches[-4:])}")

# -- Chart: price with regime overlay ----------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# Top panel: price with crisis days highlighted
ax1.plot(df["date"], df["close"], color="#2563eb", linewidth=1, label="XLE Close")
crisis_mask = df["state"] == crisis_idx
if crisis_mask.any():
    ax1.scatter(
        df.loc[crisis_mask, "date"],
        df.loc[crisis_mask, "close"],
        color="red",
        s=20,
        zorder=5,
        label="Crisis state",
        alpha=0.8,
    )
ax1.set_ylabel("XLE Price ($)")
ax1.set_title(
    "Hidden Markov Model: XLE Regime Detection (2-State Gaussian HMM)\n"
    f"Calm: {calm_pct:.0f}% of days | Crisis: {crisis_pct:.0f}% of days",
    fontsize=13,
    fontweight="bold",
)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Bottom panel: daily returns colored by regime
colors = ["#2563eb" if s == calm_idx else "#dc2626" for s in states]
ax2.bar(df["date"], df["return_daily"] * 100, color=colors, width=1, alpha=0.7)
ax2.set_ylabel("Daily Return (%)")
ax2.set_xlabel("Date")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("oil-hmm-regime-detection-python.png", dpi=150, bbox_inches="tight")
plt.show()
