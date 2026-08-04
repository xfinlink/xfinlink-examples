# Full write-up: https://xfinlink.com/blog/range-based-volatility-estimators-python
"""How many days of data does a volatility estimate need?

Compares close-to-close variance against four estimators that read the daily
open, high and low. Precision is measured by splitting every 22-day block into
its odd and even days: both halves see the same volatility regime, so their
disagreement is sampling noise and nothing else.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "DIA", "EFA", "EEM", "TLT", "XLK", "XLF"]
NAMES = ["close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
LABELS = ["Close-to-close", "Parkinson", "Garman-Klass", "Rogers-Satchell", "Yang-Zhang"]
NHALF = 11  # trading days in each half of a 22-day block

px = xfl.prices(TICKERS, start="2016-08-01", end="2026-08-03",
                fields=["open", "high", "low", "close", "adj_close"])


def estimators(o, h, l, c):
    """Daily variance estimates. Inputs are logs relative to the prior close."""
    n = len(c)
    hl, co, u, d = h - l, c - o, h - o, l - o
    rs = np.mean(u * (u - co) + d * (d - co))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return {
        "close_to_close": np.mean(c ** 2),
        "parkinson": np.mean(hl ** 2) / (4 * np.log(2)),
        "garman_klass": np.mean(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2),
        "rogers_satchell": rs,
        "yang_zhang": np.var(o, ddof=1) + k * np.var(co, ddof=1) + (1 - k) * rs,
    }


gaps, levels, full = [], [], []
for tk in TICKERS:
    t = px[px["ticker"] == tk].sort_values("date")
    f = t["adj_close"] / t["close"]          # split factor: puts o/h/l on the adj_close basis
    prev = t["adj_close"].shift(1)
    bars = pd.DataFrame({"o": np.log(t["open"] * f / prev), "h": np.log(t["high"] * f / prev),
                         "l": np.log(t["low"] * f / prev), "c": np.log(t["adj_close"] / prev)})
    m = bars.dropna().to_numpy()
    full.append({"ticker": tk, **{k: np.sqrt(v * 252) * 100
                                  for k, v in estimators(*m.T).items()}})
    for b in range(len(m) // (2 * NHALF)):
        blk = m[b * 2 * NHALF:(b + 1) * 2 * NHALF]
        A, B = estimators(*blk[0::2].T), estimators(*blk[1::2].T)
        if min(min(A.values()), min(B.values())) <= 0:
            continue
        gaps.append({"ticker": tk, **{k: np.log(A[k]) - np.log(B[k]) for k in NAMES}})
        levels.append({k: 0.5 * (A[k] + B[k]) for k in NAMES})

G, L, F = pd.DataFrame(gaps), pd.DataFrame(levels), pd.DataFrame(full).set_index("ticker")
base = G["close_to_close"].var(ddof=1)
eff = {k: base / G[k].var(ddof=1) for k in NAMES}
per_ticker = {k: [G[G["ticker"] == t]["close_to_close"].var(ddof=1) / G[G["ticker"] == t][k].var(ddof=1)
                  for t in TICKERS] for k in NAMES}

print(f"Annualised volatility by estimator, {len(px):,} daily bars, "
      f"{px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}\n")
print(F.round(2).rename(columns=dict(zip(NAMES, LABELS))).to_string())
print(f"\nSampling noise, {len(G)} odd/even half-block pairs of {NHALF} days each\n")
print(f"{'estimator':>16}{'ann vol %':>11}{'noise sd':>10}{'precision':>11}{'days for 21':>13}")
for k, lab in zip(NAMES, LABELS):
    print(f"{lab:>16}{np.sqrt(L[k].mean() * 252) * 100:11.2f}{G[k].std(ddof=1):10.3f}"
          f"{eff[k]:11.2f}{21 / eff[k]:13.1f}")
print("\nprecision range across the eight funds")
for k, lab in zip(NAMES[1:], LABELS[1:]):
    print(f"  {lab:>15}: {min(per_ticker[k]):.2f} to {max(per_ticker[k]):.2f}")

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0"})
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(NAMES))
colors = ["#6b7280", "#3b82f6", "#3b82f6", "#3b82f6", "#22d3ee"]
ax.barh(y, [eff[k] for k in NAMES], color=colors, height=0.6, zorder=2)
for i, k in enumerate(NAMES[1:], start=1):
    ax.scatter(per_ticker[k], np.full(8, i), color="#e0e0e0", s=14, alpha=0.75, zorder=3)
right = max(max(v) for v in per_ticker.values()) + 0.35
for i, k in enumerate(NAMES):
    ax.text(right, i, f"{eff[k]:.2f}x   {21 / eff[k]:.0f} days", va="center",
            fontsize=9.5, color="#e0e0e0")
ax.set_yticks(y, LABELS)
ax.invert_yaxis()
ax.set_xlim(0, right + 2.0)
ax.set_xlabel("Precision per day of data, relative to close-to-close "
              "(dots: each of the eight funds)")
ax.set_title("How many days of data a volatility estimate needs", color="#e0e0e0")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#3a3a3a")
plt.tight_layout()
plt.savefig("range-based-volatility-estimators-python.png", dpi=150, facecolor="#0a0a0a")
