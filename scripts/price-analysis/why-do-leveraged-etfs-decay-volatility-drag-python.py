# Full write-up: https://xfinlink.com/blog/why-do-leveraged-etfs-decay-volatility-drag-python
#
# Why Do Leveraged ETFs Decay? Measuring Volatility Drag in Python
#
# Tests whether the decay of leveraged ETFs is explained by the standard
# arithmetic: a leveraged position's expected log return carries a penalty of
# (L^2 - L)/2 * sigma^2 per year. Four 3x funds and two 2x funds are measured
# against their underlying index ETFs.
#
# Built from SEC EDGAR public filings and market data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# (leveraged fund, underlying index ETF, stated daily leverage)
PAIRS = [
    ("TQQQ", "QQQ", 3),
    ("UPRO", "SPY", 3),
    ("SOXL", "SOXX", 3),
    ("TNA", "IWM", 3),
    ("QLD", "QQQ", 2),
    ("SSO", "SPY", 2),
]
PERIOD = "3y"


def daily_returns(ticker, period=PERIOD):
    """Total daily returns. Never compute returns from raw `close`: leveraged
    ETFs reverse-split often and raw close steps across splits."""
    d = xfl.prices(ticker, period=period, fields=["return_daily"]).sort_values("date")
    return d[["date", "return_daily"]].rename(columns={"return_daily": ticker})


def aligned(lev, und, period=PERIOD):
    """Inner-join on date so the two calendars cannot drift apart."""
    m = daily_returns(lev, period).merge(daily_returns(und, period), on="date", how="inner")
    return m.dropna(subset=[lev, und]).reset_index(drop=True)


def analyse(lev, und, L, period=PERIOD):
    m = aligned(lev, und, period)
    rl, ru = m[lev].values, m[und].values
    years = len(m) / 252.0

    # 1. Does the fund keep its DAILY promise? Slope should be ~L.
    beta, alpha = np.polyfit(ru, rl, 1)
    r2 = np.corrcoef(ru, rl)[0, 1] ** 2

    # 2. Multi-period reality.
    actual = np.prod(1 + rl) - 1                 # what the fund did
    index = np.prod(1 + ru) - 1                  # what the index did
    naive = L * index                            # "L times the annual return"
    ideal = (1 + index) ** L - 1                 # correct no-drag benchmark

    # 3. Decomposition of the log shortfall, annualised.
    sigma = np.std(ru, ddof=1) * np.sqrt(252)
    realized_drag = (L * np.log(1 + index) - np.log(1 + actual)) / years
    theory_drag = (L ** 2 - L) / 2 * sigma ** 2  # the variance penalty
    carry = -alpha * 252                         # fees + financing, from the intercept
    residual = realized_drag - theory_drag - carry

    return dict(
        lev=lev, und=und, L=L, n=len(m), years=years, beta=beta, r2=r2,
        alpha_ann=alpha * 252, index=index, naive=naive, ideal=ideal, actual=actual,
        sigma=sigma, realized_drag=realized_drag, theory_drag=theory_drag,
        carry=carry, residual=residual, m=m,
    )


res = [analyse(*p) for p in PAIRS]

print("=" * 78)
print("1. DOES THE DAILY PROMISE HOLD?  regression: fund daily return ~ index daily return")
print("=" * 78)
print(f"{'pair':<12}{'target':>7}{'slope':>9}{'R2':>9}{'intercept /yr':>16}")
for r in res:
    print(f"{r['lev']+'/'+r['und']:<12}{r['L']:>6}x{r['beta']:>9.3f}{r['r2']:>9.4f}"
          f"{r['alpha_ann']*100:>15.2f}%")

print()
print("=" * 78)
print(f"2. MULTI-PERIOD REALITY ({PERIOD}, {res[0]['n']} trading days)")
print("=" * 78)
print(f"{'pair':<12}{'index':>10}{'naive Lx':>11}{'(1+U)^L':>11}{'actual':>11}{'vs naive':>11}")
for r in res:
    print(f"{r['lev']+'/'+r['und']:<12}{r['index']*100:>9.1f}%{r['naive']*100:>10.1f}%"
          f"{r['ideal']*100:>10.1f}%{r['actual']*100:>10.1f}%"
          f"{(r['actual']-r['naive'])*100:>+10.1f}pp")

print()
print("=" * 78)
print("3. WHERE THE SHORTFALL COMES FROM (annualised, log terms)")
print("=" * 78)
print(f"{'pair':<12}{'index vol':>11}{'realized':>10}{'(L2-L)/2 s2':>13}{'fees+fin':>10}{'residual':>10}")
for r in res:
    print(f"{r['lev']+'/'+r['und']:<12}{r['sigma']*100:>10.1f}%{r['realized_drag']*100:>9.2f}%"
          f"{r['theory_drag']*100:>12.2f}%{r['carry']*100:>9.2f}%{r['residual']*100:>+9.2f}%")

# 4. Does drag really scale with sigma^2? Test across 6 sub-periods per pair.
print()
print("=" * 78)
print("4. SCALING TEST: drag vs variance across 6 sub-periods x 6 pairs")
print("=" * 78)
rows = []
for lev, und, L in PAIRS:
    m = aligned(lev, und)
    for i, ix in enumerate(np.array_split(np.arange(len(m)), 6)):
        c = m.iloc[ix]
        rl, ru = c[lev].values, c[und].values
        years = len(c) / 252.0
        _, a = np.polyfit(ru, rl, 1)
        var = np.var(ru, ddof=1) * 252
        idx, act = np.prod(1 + ru) - 1, np.prod(1 + rl) - 1
        drag = (L * np.log(1 + idx) - np.log(1 + act)) / years
        rows.append(dict(L=L, var=var, net=drag - (-a * 252)))
w = pd.DataFrame(rows)
for L in (2, 3):
    s = w[w.L == L]
    slope = np.sum(s["var"] * s["net"]) / np.sum(s["var"] ** 2)
    print(f"  {L}x funds: fitted drag/variance slope = {slope:.3f}   "
          f"theory (L^2-L)/2 = {(L**2-L)/2:.1f}   (n={len(s)} sub-periods)")
print(f"  correlation between realized drag and theoretical drag: "
      f"{np.corrcoef((w.L**2-w.L)/2*w['var'], w.net)[0,1]:.4f}")

# ----------------------------- chart -----------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#404040", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# Panel A: TQQQ actual vs the two benchmarks
t = res[0]
m = t["m"]
u_cum = np.cumprod(1 + m["QQQ"].values)
l_cum = np.cumprod(1 + m["TQQQ"].values)
ax1.plot(m["date"], (1 + 3 * (u_cum - 1)), color="#6b7280", lw=1.6,
         label="Naive expectation: 1 + 3 x QQQ return")
ax1.plot(m["date"], u_cum ** 3, color="#f59e0b", lw=1.6,
         label="No-drag benchmark: (1 + QQQ return)$^3$")
ax1.plot(m["date"], l_cum, color="#3b82f6", lw=2.0, label="TQQQ actual")
ax1.set_ylabel("Growth of $1")
ax1.set_title("TQQQ versus 3x the Nasdaq 100: the gap is volatility drag")
ax1.legend(frameon=False, loc="upper left", fontsize=9)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

# Panel B: drag scales with variance
for L, colour, mk in ((3, "#3b82f6", "o"), (2, "#f59e0b", "s")):
    s = w[w.L == L]
    ax2.scatter(s["var"] * 100, s["net"] * 100, color=colour, s=34, alpha=0.85,
                marker=mk, label=f"{L}x funds", zorder=3)
    x = np.linspace(0, w["var"].max() * 100 * 1.05, 10)
    ax2.plot(x, (L ** 2 - L) / 2 * x, color=colour, lw=1.0, ls="--", alpha=0.7)
ax2.set_xlabel("Annualised variance of the underlying index (%)")
ax2.set_ylabel("Annualised drag (%)")
ax2.set_title("Drag scales with variance; dashed lines are theory, not fits")
ax2.legend(frameon=False, loc="upper left", fontsize=9)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("worker/src/site/blog-images/why-do-leveraged-etfs-decay-volatility-drag-python.png",
            dpi=150)
