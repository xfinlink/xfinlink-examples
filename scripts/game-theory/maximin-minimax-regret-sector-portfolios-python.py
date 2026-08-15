# Full write-up: https://xfinlink.com/blog/maximin-minimax-regret-sector-portfolios-python
"""Maximin and minimax regret sector portfolios, fitted 2006-2015, tested 2016-2025.

Sector allocation treated as a one-shot game against nature: nature picks the
calendar year, the allocator picks long-only weights across nine sector funds.
Two classical decision criteria are solved as linear programs and then carried
forward, unchanged, into a holdout decade.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
NAMES = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Cons. Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons. Discret.",
}
FIT = range(2006, 2016)
HOLD = range(2016, 2026)

px = xfl.prices(SECTORS + ["SPY"], start="2006-01-01", end="2025-12-31",
                fields=["close", "return_daily"], max_rows=200000)
px["year"] = px["date"].dt.year
annual = (px.groupby(["year", "ticker"])["return_daily"]
            .apply(lambda s: (1 + s).prod() - 1).unstack()[SECTORS + ["SPY"]])

R = annual[SECTORS]
fit, hold = R.loc[FIT], R.loc[HOLD]


def maximin(r):
    """Weights maximising the worst annual portfolio return over the window."""
    n = r.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-r.values, np.ones((len(r), 1))])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    sol = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(r)), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * n + [(None, None)])
    return sol.x[:n]


def minimax_regret(r):
    """Weights minimising the largest shortfall against the best sector each year."""
    n = r.shape[1]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.hstack([-r.values, -np.ones((len(r), 1))])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    sol = linprog(c, A_ub=A_ub, b_ub=-r.max(axis=1).values, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * n + [(0, None)])
    return sol.x[:n]


rules = {
    "Maximin": maximin(fit),
    "Minimax regret": minimax_regret(fit),
    "Equal weight": np.ones(len(SECTORS)) / len(SECTORS),
}


def score(w, r):
    p = pd.Series(r.values @ w, index=r.index)
    regret = r.max(axis=1) - p
    cagr = np.prod(1 + p) ** (1 / len(p)) - 1
    return p, regret, cagr


print(f"Sector allocation as a game against nature: {len(SECTORS)} sector funds")
print(f"Fit window 2006-2015, holdout 2016-2025, annual total returns\n")

print("Weights fitted on 2006-2015 (%)")
print("                  " + "".join(f"{t:>7}" for t in SECTORS))
for name, w in rules.items():
    print(f"{name:<18}" + "".join(f"{x * 100:7.1f}" for x in w))

print("\n                     ---- fit 2006-2015 ----   -- holdout 2016-2025 --")
print("                     worst  max reg    CAGR     worst  max reg    CAGR")
for name, w in rules.items():
    pf, gf, cf = score(w, fit)
    ph, gh, ch = score(w, hold)
    print(f"{name:<18}  {pf.min() * 100:6.1f}%  {gf.max() * 100:5.1f}pp  {cf * 100:6.2f}%  "
          f"{ph.min() * 100:6.1f}%  {gh.max() * 100:5.1f}pp  {ch * 100:6.2f}%")

spy_f = np.prod(1 + annual.loc[FIT, "SPY"]) ** 0.1 - 1
spy_h = np.prod(1 + annual.loc[HOLD, "SPY"]) ** 0.1 - 1
print(f"{'S&P 500 (SPY)':<18}  {annual.loc[FIT, 'SPY'].min() * 100:6.1f}%"
      f"{'':10}{spy_f * 100:6.2f}%  {annual.loc[HOLD, 'SPY'].min() * 100:6.1f}%"
      f"{'':10}{spy_h * 100:6.2f}%")

print("\nWhere the binding years sit")
for name, w in rules.items():
    ph, gh, _ = score(w, hold)
    print(f"{name:<18} holdout worst year {ph.idxmin()} at {ph.min() * 100:.1f}%,"
          f" largest regret {gh.idxmax()} at {gh.max() * 100:.1f}pp")
print(f"Best sector of 2022: {R.loc[2022].idxmax()} at {R.loc[2022].max() * 100:.1f}%,"
      f" worst {R.loc[2022].idxmin()} at {R.loc[2022].min() * 100:.1f}%")
print(f"Mean gap between best and worst sector: fit "
      f"{(fit.max(axis=1) - fit.min(axis=1)).mean() * 100:.1f}pp, holdout "
      f"{(hold.max(axis=1) - hold.min(axis=1)).mean() * 100:.1f}pp")

# ---- chart -------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                               gridspec_kw={"width_ratios": [1.5, 1]})

x = np.arange(len(SECTORS))
ax1.bar(x - 0.2, rules["Maximin"] * 100, 0.4, color="#3b82f6", label="Maximin")
ax1.bar(x + 0.2, rules["Minimax regret"] * 100, 0.4, color="#94a3b8",
        label="Minimax regret")
ax1.axhline(100 / len(SECTORS), color="#f59e0b", ls="--", lw=1, label="Equal weight")
ax1.set_xticks(x)
ax1.set_xticklabels([NAMES[t] for t in SECTORS], rotation=45, ha="right")
ax1.set_ylabel("Weight (%)")
ax1.set_title("Weights chosen on 2006-2015", color="#e0e0e0")
ax1.set_ylim(0, 118)
ax1.legend(frameon=False, labelcolor="#e0e0e0", loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

labels = list(rules)
xr = np.arange(len(labels))
fit_reg = [score(w, fit)[1].max() * 100 for w in rules.values()]
hold_reg = [score(w, hold)[1].max() * 100 for w in rules.values()]
ax2.bar(xr - 0.2, fit_reg, 0.4, color="#3b82f6", label="Fit 2006-2015")
ax2.bar(xr + 0.2, hold_reg, 0.4, color="#94a3b8", label="Holdout 2016-2025")
ax2.set_xticks(xr)
ax2.set_xticklabels(["Maximin", "Minimax\nregret", "Equal\nweight"])
ax2.set_ylabel("Largest annual regret (pp)")
ax2.set_title("Worst-case regret, fit vs holdout", color="#e0e0e0")
ax2.set_ylim(0, 84)
ax2.legend(frameon=False, labelcolor="#e0e0e0", loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("maximin-minimax-regret-sector-portfolios-python.png", dpi=130,
            facecolor="#0a0a0a")
