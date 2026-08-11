# Full write-up: https://xfinlink.com/blog/does-buying-the-dip-work-volatility-regime-python
"""Short-term reversal in eight ETFs, split by volatility regime.

Yesterday's direction predicts today's return only when volatility is high.
Regimes are assigned from data available on the day, so there is no lookahead.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "EFA", "EEM", "XLK", "XLP", "TLT", "GLD"]
START, END = "2004-01-01", "2026-08-07"
REGIMES = ["calm", "normal", "stressed"]

frames = []
for ticker in TICKERS:
    frames.append(xfl.prices(ticker, start=START, end=END,
                             fields=["return_daily"]))
px = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


def build(g):
    g = g.copy()
    r = g["return_daily"]
    g["vol"] = r.rolling(20).std()
    g["pct"] = g["vol"].rolling(504).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    g["fwd"] = r.shift(-1)
    return g


panel = pd.concat([build(g) for _, g in px.groupby("ticker")], ignore_index=True)
panel = panel.dropna(subset=["pct", "fwd"])
panel["regime"] = pd.cut(panel["pct"], [-0.001, 1 / 3, 2 / 3, 1.001], labels=REGIMES)
panel["down"] = (panel["return_daily"] < 0).astype(float)
panel["z"] = panel["return_daily"] / panel["vol"]


def spread(frame):
    """Mean next-day return after a down day minus after an up day, and its
    t-statistic with standard errors clustered on date."""
    X = np.column_stack([np.ones(len(frame)), frame["down"].values])
    y = frame["fwd"].values
    A = np.linalg.inv(X.T @ X)
    b = A @ X.T @ y
    u = y - X @ b
    meat = np.zeros((2, 2))
    for idx in frame.groupby("date").indices.values():
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    return b[1], b[1] / np.sqrt((A @ meat @ A)[1, 1])


print(f"{' '.join(TICKERS)}: daily total returns "
      f"{px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}, "
      f"{len(px):,} bars, {len(panel):,} usable observations")
print("regime = trailing two-year percentile of the 20-day return standard deviation\n")

print("next-day return, pooled across the eight funds")
print(f"{'regime':<10}{'days':>8}{'after a down day':>19}{'after an up day':>18}"
      f"{'spread':>10}{'t':>7}")
for regime in REGIMES:
    x = panel[panel["regime"] == regime]
    down, up = x[x["down"] == 1], x[x["down"] == 0]
    sp, t = spread(x)
    print(f"{regime:<10}{len(x):>8,}{down['fwd'].mean() * 1e4:>16.2f} bp"
          f"{up['fwd'].mean() * 1e4:>15.2f} bp{sp * 1e4:>7.2f} bp{t:>7.2f}")

print("\nreversal spread by fund (basis points, t in brackets)")
print(f"{'fund':<7}{'calm':>18}{'stressed':>18}")
for ticker in TICKERS:
    x = panel[panel["ticker"] == ticker]
    cells = []
    for regime in ["calm", "stressed"]:
        sp, t = spread(x[x["regime"] == regime])
        cells.append(f"{sp * 1e4:8.2f} ({t:5.2f})")
    print(f"{ticker:<7}{cells[0]:>18}{cells[1]:>18}")

print("\nnext-day return by the size of yesterday's move "
      "(z = return divided by the 20-day standard deviation)")
cuts = [(-9, -1.5), (-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5), (1.5, 9)]
labels = ["below -1.5", "-1.5 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1.5", "above 1.5"]
grid = {}
print(f"{'yesterday z':<15}{'calm days':>11}{'calm':>11}{'stressed days':>15}{'stressed':>11}")
for (lo, hi), label in zip(cuts, labels):
    row = []
    for regime in ["calm", "stressed"]:
        x = panel[(panel["regime"] == regime) & (panel["z"] > lo) & (panel["z"] <= hi)]
        row.append((len(x), x["fwd"].mean() * 1e4))
    grid[label] = row
    print(f"{label:<15}{row[0][0]:>11,}{row[0][1]:>8.2f} bp"
          f"{row[1][0]:>15,}{row[1][1]:>8.2f} bp")

print("\nrobustness")
for name, sub in [("first half, 2006-2014", panel[panel["date"] < "2015-01-01"]),
                  ("second half, 2015-2026", panel[panel["date"] >= "2015-01-01"]),
                  ("excluding the 2008-09 and 2020 crisis windows", panel[
                      ~(((panel["date"] >= "2008-09-01") & (panel["date"] <= "2009-06-30"))
                        | ((panel["date"] >= "2020-02-15") & (panel["date"] <= "2020-05-31")))])]:
    sp, t = spread(sub[sub["regime"] == "stressed"])
    calm_sp, calm_t = spread(sub[sub["regime"] == "calm"])
    print(f"{name:<46} stressed {sp * 1e4:6.2f} bp (t={t:4.2f})   "
          f"calm {calm_sp * 1e4:5.2f} bp (t={calm_t:5.2f})")

hit = panel.groupby(["regime", "down"], observed=True)["fwd"].apply(lambda s: (s > 0).mean() * 100)
print("\nshare of next days that close higher")
for regime in REGIMES:
    print(f"{regime:<10}after a down day {hit[(regime, 1.0)]:.1f}%   "
          f"after an up day {hit[(regime, 0.0)]:.1f}%")

# ---- chart -------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = np.arange(len(labels))
ax1.bar(x - 0.2, [grid[l][0][1] for l in labels], 0.4, color="#6b7280", label="Calm markets")
ax1.bar(x + 0.2, [grid[l][1][1] for l in labels], 0.4, color="#3b82f6", label="Stressed markets")
ax1.axhline(0, color="#e0e0e0", lw=0.8)
ax1.set_xticks(x, labels)
ax1.set_xlabel("Yesterday's move, in standard deviations")
ax1.set_ylabel("Average next-day return\n(basis points)")
ax1.set_title("Buying the dip pays only when volatility is high", color="#e0e0e0", pad=10)
ax1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", frameon=False)

y = np.arange(len(TICKERS))
calm = [spread(panel[(panel["ticker"] == t) & (panel["regime"] == "calm")])[0] * 1e4 for t in TICKERS]
stress = [spread(panel[(panel["ticker"] == t) & (panel["regime"] == "stressed")])[0] * 1e4 for t in TICKERS]
ax2.barh(y + 0.2, calm, 0.4, color="#6b7280", label="Calm markets")
ax2.barh(y - 0.2, stress, 0.4, color="#3b82f6", label="Stressed markets")
ax2.axvline(0, color="#e0e0e0", lw=0.8)
ax2.set_yticks(y, TICKERS)
ax2.invert_yaxis()
ax2.set_xlabel("Next-day return after a down day minus after an up day (basis points)")
ax2.set_title("Equity funds reverse under stress, bonds and gold barely move",
              color="#e0e0e0", pad=10)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("does-buying-the-dip-work-volatility-regime-python.png", dpi=150,
            facecolor="#0a0a0a")
