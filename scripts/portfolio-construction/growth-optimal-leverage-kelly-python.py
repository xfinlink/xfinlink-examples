# Full write-up: https://xfinlink.com/blog/growth-optimal-leverage-kelly-python
#
# How much leverage maximises long-run growth?
# Constant daily-rebalanced leverage on a broad market fund and nine sector
# funds, financed at the Treasury bill rate, 2007-2026.

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(600)

FUNDS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CASH = "BIL"
GRID = np.round(np.arange(0.0, 6.001, 0.05), 2)
IMG = "growth-optimal-leverage-kelly-python.png"


def retry(fn, *a, **k):
    for attempt in range(5):
        try:
            return fn(*a, **k)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


px = pd.concat([retry(xfl.prices, t, start="2006-01-01", end="2026-09-10",
                      fields=["return_daily"], max_rows=200000)
                for t in FUNDS + [CASH]], ignore_index=True)
R = px.pivot(index="date", columns="ticker", values="return_daily").sort_index().dropna()
cash = R[CASH].values


def growth(r):
    """Annualised geometric growth rate of a daily return series."""
    if np.min(r) <= -1.0:
        return -1.0
    return float(np.expm1(np.log1p(r).sum() / len(r) * 252))


def maxdd(r):
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def levered(r, c, L, spread=0.0):
    """Constant leverage L reset every day, borrowing at the bill rate plus a spread."""
    return L * r - (L - 1.0) * (c + spread / 252.0)


def curve(r, c, spread=0.0):
    return np.array([growth(levered(r, c, L, spread)) for L in GRID])


g_cash = growth(cash)
rows, curves = [], {}
for f in FUNDS:
    r = R[f].values
    cv = curve(r, cash)
    curves[f] = cv
    L = GRID[int(np.argmax(cv))]
    half = round(L / 2, 2)
    capped = GRID[np.array([maxdd(levered(r, cash, x)) for x in GRID]) >= -0.50].max()
    rows.append({"fund": f, "vol": r.std(ddof=1) * np.sqrt(252), "g1": growth(r),
                 "dd1": maxdd(r), "L": L, "gL": cv.max(), "ddL": maxdd(levered(r, cash, L)),
                 "gH": growth(levered(r, cash, half)), "ddH": maxdd(levered(r, cash, half)),
                 "keep": (growth(levered(r, cash, half)) - g_cash) / (cv.max() - g_cash),
                 "L50": capped, "gL50": growth(levered(r, cash, capped)),
                 "kelly": (r.mean() - cash.mean()) / r.var(ddof=1)})
full = pd.DataFrame(rows).set_index("fund")

mid = len(R) // 2
A, B = R.iloc[:mid], R.iloc[mid:]
rows = []
for f in FUNDS:
    cvA, cvB = curve(A[f].values, A[CASH].values), curve(B[f].values, B[CASH].values)
    LA, LB = GRID[int(np.argmax(cvA))], GRID[int(np.argmax(cvB))]
    rb, cb = B[f].values, B[CASH].values
    rows.append({"fund": f, "L_fit": LA, "L_best": LB, "g_1x": growth(rb),
                 "g_fit": growth(levered(rb, cb, LA)), "g_best": cvB.max(),
                 "dd_fit": maxdd(levered(rb, cb, LA))})
split = pd.DataFrame(rows).set_index("fund")

bar = "=" * 92
print(bar)
print("GROWTH-OPTIMAL LEVERAGE ON A BROAD MARKET FUND AND NINE SECTOR FUNDS")
print(bar)
print("Sample     SPY and the nine sector funds with a continuous daily series across the")
print("           window; the cash leg is BIL, a Treasury bill fund")
print(f"Window     {R.index.min():%Y-%m-%d} to {R.index.max():%Y-%m-%d}, {len(R):,} trading days, "
      f"bills compounded at {g_cash * 100:.2f}% a year")
print("Method     levered return = L x fund return - (L - 1) x bill return, reset daily;")
print("           growth is annualised geometric; L searched from 0.00 to 6.00 in steps of 0.05")
print(f"Worst single session in the sample: {R[FUNDS].values.min() * 100:.2f}%   "
      f"optima sitting at the edge of the search: {int((full.L >= GRID.max()).sum())}")
print()
print("                 unlevered      growth-optimal L        half of it      drawdown held to 50%")
print("fund    vol     growth  maxDD     L   growth  maxDD    growth  maxDD      L    growth")
for f, x in full.iterrows():
    print(f"{f:5s} {x.vol * 100:5.1f}%  {x.g1 * 100:6.2f}% {x.dd1 * 100:6.1f}%  {x.L:4.2f} "
          f"{x.gL * 100:6.2f}% {x.ddL * 100:6.1f}%  {x.gH * 100:6.2f}% {x.ddH * 100:6.1f}%   "
          f"{x.L50:4.2f}  {x.gL50 * 100:6.2f}%")
print()
print(f"Half the growth-optimal leverage kept {full.keep.min() * 100:.0f}% to {full.keep.max() * 100:.0f}% "
      f"of the excess growth rate (median {full.keep.median() * 100:.0f}%).")
print(f"Excess return over variance, the textbook Kelly fraction: SPY {full.loc['SPY', 'kelly']:.2f} "
      f"against a searched optimum of {full.loc['SPY', 'L']:.2f}.")
print()
print("Leverage fitted on the first half of the window, scored on the second half")
print(f"  first half {A.index.min():%Y-%m-%d} to {A.index.max():%Y-%m-%d}, "
      f"second half {B.index.min():%Y-%m-%d} to {B.index.max():%Y-%m-%d}")
print("fund    fitted L   best L    growth at 1.0x   at fitted L   at best L   maxDD at fitted L")
for f, x in split.iterrows():
    print(f"{f:5s}   {x.L_fit:5.2f}     {x.L_best:5.2f}      {x.g_1x * 100:8.2f}%     "
          f"{x.g_fit * 100:8.2f}%    {x.g_best * 100:8.2f}%       {x.dd_fit * 100:6.1f}%")
print(f"Fitted leverage beat 1.0x in {int((split.g_fit > split.g_1x).sum())} of {len(split)} funds; "
      f"median gap between fitted and best {float((split.L_best - split.L_fit).abs().median()):.2f}x; "
      f"median growth\ngiven up against the best possible "
      f"{float(((split.g_best - split.g_fit) * 100).median()):.2f} points.")
print()
print("Financing spread charged on the borrowed leg, broad market fund:", end=" ")
print("   ".join(f"{s * 10000:.0f}bp -> {GRID[int(np.argmax(curve(R['SPY'].values, cash, s)))]:.2f}x, "
                 f"{curve(R['SPY'].values, cash, s).max() * 100:.2f}%" for s in [0.0, 0.01, 0.02]))
print(bar)

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for f in FUNDS:
    lead = f == "SPY"
    ax1.plot(GRID, curves[f] * 100, color="#3b82f6" if lead else "#5a6472",
             lw=2.0 if lead else 1.0, zorder=3 if lead else 2)
    i = int(np.argmax(curves[f]))
    ax1.plot(GRID[i], curves[f][i] * 100, "o", ms=5 if lead else 3.5,
             color="#3b82f6" if lead else "#8b95a3", zorder=4)
ax1.axhline(0, color="#3a3a3a", lw=0.8)
ax1.axvline(1.0, color="#8b95a3", lw=0.8, ls="--")
ax1.set_xlim(0, 5)
ax1.set_ylim(-25, 35)
ax1.set_xlabel("Leverage on the fund (1.0 is unlevered)")
ax1.set_ylabel("Annualised growth rate (%)")
ax1.set_title("Growth against leverage, 2007-2026", color="#e0e0e0", fontsize=10)
ax1.text(1.08, -23, "unlevered", color="#8b95a3", fontsize=8)
ax1.text(0.03, 0.96, "blue: broad market fund\ngrey: nine sector funds\ndots: growth-optimal leverage",
         transform=ax1.transAxes, va="top", color="#8b95a3", fontsize=8)

hi = float(max(split.L_fit.max(), split.L_best.max())) + 0.5
ax2.plot([0, hi], [0, hi], color="#8b95a3", lw=0.8, ls="--")
ax2.scatter(split.L_fit, split.L_best, color="#3b82f6", s=30, zorder=3)
for f, x in split.iterrows():
    ax2.annotate(f, (x.L_fit, x.L_best), textcoords="offset points", xytext=(5, 3),
                 color="#8b95a3", fontsize=8)
ax2.set_xlim(0, hi)
ax2.set_ylim(0, hi)
ax2.set_xlabel("Best leverage over 2007-2017")
ax2.set_ylabel("Best leverage over 2017-2026")
ax2.set_title("The winning leverage does not repeat", color="#e0e0e0", fontsize=10)
plt.tight_layout()
plt.savefig(IMG, dpi=150, facecolor="#0a0a0a")
