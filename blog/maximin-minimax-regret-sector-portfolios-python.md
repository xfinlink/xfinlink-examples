**Do Worst-Case Sector Portfolios Hold Up Out of Sample? Maximin and Minimax Regret in Python**

August 15, 2026 · GAME-THEORY

**What's the question?**

An allocator dividing capital across sectors without a return forecast still has to choose weights. One honest way to choose is to abandon forecasting and pick weights that behave acceptably whichever year arrives next. Decision theory supplies two criteria for that, both from the study of games played against an indifferent opponent, conventionally called nature.

The first is maximin, from Abraham Wald: choose the allocation whose worst outcome is the least bad. The second is minimax regret, from Leonard Savage: for each year, measure the gap between the portfolio return and the best single sector's return, then choose the allocation whose largest gap is smallest. Regret is not loss. A portfolio can lose nothing and still carry enormous regret, which is what a defensive position does in a year when energy gains 64%.

Each criterion reduces to a linear program, with no simulation, no covariance estimate, and no tuning parameter, so what it chooses is exact. Whether the choice means anything is the open question: weights fitted to the worst case of one decade should, if the criterion captures something structural, still control the worst case of the next.

**The approach**

Nine sector funds cover the S&P 500 as originally divided, all trading continuously across the window. Calendar-year total returns from 2006 through 2025 give a payoff matrix of 20 rows and 9 columns.

1. Compound daily returns for the nine funds within each calendar year.
2. Split the matrix in half: 2006 to 2015 is the fit window, 2016 to 2025 is held out.
3. Solve maximin on the fit window: maximise t subject to every fit-year portfolio return being at least t, long-only weights summing to one.
4. Solve minimax regret: minimise t subject to each year's best sector return minus the portfolio return being at most t.
5. Carry both weight vectors into the holdout decade without refitting and measure the same quantities there, next to equal weight and the index.

Equal weight is the control that matters: it uses no information from the fit window, so a criterion that cannot beat it out of sample has contributed nothing.

**Code**

```python
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FIT, HOLD = range(2006, 2016), range(2016, 2026)

px = xfl.prices(SECTORS + ["SPY"], start="2006-01-01", end="2025-12-31",
                fields=["close", "return_daily"], max_rows=200000)
px["year"] = px["date"].dt.year
annual = (px.groupby(["year", "ticker"])["return_daily"]
            .apply(lambda s: (1 + s).prod() - 1).unstack()[SECTORS + ["SPY"]])

R = annual[SECTORS]
fit, hold = R.loc[FIT], R.loc[HOLD]


def maximin(r):
    n = r.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_ub = np.hstack([-r.values, np.ones((len(r), 1))])
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    sol = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(r)), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * n + [(None, None)])
    return sol.x[:n]


def minimax_regret(r):
    n = r.shape[1]
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.hstack([-r.values, -np.ones((len(r), 1))])
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    sol = linprog(c, A_ub=A_ub, b_ub=-r.max(axis=1).values, A_eq=A_eq,
                  b_eq=[1.0], bounds=[(0, 1)] * n + [(0, None)])
    return sol.x[:n]


rules = {"Maximin": maximin(fit), "Minimax regret": minimax_regret(fit),
         "Equal weight": np.ones(len(SECTORS)) / len(SECTORS)}

for name, w in rules.items():
    for label, r in [("fit", fit), ("holdout", hold)]:
        p = pd.Series(r.values @ w, index=r.index)
        print(f"{name} {label}: worst {p.min():.1%}  "
              f"max regret {(r.max(axis=1) - p).max():.1%}")
```

Full script with formatting and visualisation: [maximin-minimax-regret-sector-portfolios-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/game-theory/maximin-minimax-regret-sector-portfolios-python.py)

**Output**

<img src="/blog-images/maximin-minimax-regret-sector-portfolios-python.png" alt="Left panel: sector weights chosen by maximin and minimax regret on 2006 to 2015, with maximin holding only consumer staples. Right panel: largest annual regret for each rule in the fit window and the holdout decade" style="width:100%;border-radius:8px;margin:16px 0;" />

```
Sector allocation as a game against nature: 9 sector funds
Fit window 2006-2015, holdout 2016-2025, annual total returns

Weights fitted on 2006-2015 (%)
                      XLB    XLE    XLF    XLI    XLK    XLP    XLU    XLV    XLY
Maximin               0.0    0.0    0.0    0.0    0.0  100.0    0.0    0.0    0.0
Minimax regret       15.2   16.9    0.0    0.0   34.8   33.0    0.0    0.0    0.0
Equal weight         11.1   11.1   11.1   11.1   11.1   11.1   11.1   11.1   11.1

                     ---- fit 2006-2015 ----   -- holdout 2016-2025 --
                     worst  max reg    CAGR     worst  max reg    CAGR
Maximin              -15.0%   37.0pp   10.90%    -8.1%   65.0pp    7.23%
Minimax regret       -32.7%   17.7pp    8.72%    -8.6%   65.1pp   14.26%
Equal weight         -35.4%   26.5pp    7.76%    -6.4%   67.5pp   12.82%
S&P 500 (SPY)        -36.8%            7.23%   -18.2%           14.59%

Where the binding years sit
Maximin            holdout worst year 2018 at -8.1%, largest regret 2022 at 65.0pp
Minimax regret     holdout worst year 2018 at -8.6%, largest regret 2022 at 65.1pp
Equal weight       holdout worst year 2018 at -6.4%, largest regret 2022 at 67.5pp
Best sector of 2022: XLE at 64.2%, worst XLY at -36.3%
Mean gap between best and worst sector: fit 33.7pp, holdout 45.8pp
```

**What this tells us**

Maximin puts the entire portfolio in consumer staples, which is the exact solution, not an artefact. Every portfolio built from these nine sectors had its worst fit-window year in 2008, so the criterion collapses to one question: which sector fell least in 2008? Staples fell 15.0%, the next best fell 23.3%, and any mixture drags that figure down. One row of the matrix fixes the whole allocation.

Minimax regret behaves differently, spreading across four sectors and weighting technology at 34.8% and staples at 33.0%, which pairs the decade's strongest sector with its steadiest. Holding a slice of the sectors most likely to run away is how a portfolio caps the gap against whichever one actually does. Largest fit-window regret falls to 17.7pp against 26.5pp for equal weight, a reduction of a third.

Neither result survives the holdout. Maximin, whose whole purpose is protecting the worst year, finishes last of the three on that measure: down 8.1% in 2018 against 6.4% for equal weight. Minimax regret keeps its ranking but almost none of its margin, beating equal weight by 2.4pp of regret rather than 8.8pp.

The 2022 column explains why. Energy gained 64.2% while consumer discretionary lost 36.3%, a spread of 100.4 percentage points inside one index in one year, and no fixed weight vector holding 17% or less of the winner keeps regret small against that. The holdout decade was wider throughout, the average gap between best and worst sector growing from 33.7pp to 45.8pp.

The cost is plainer still. Concentrating in staples returned 7.23% annually over the holdout while equal weight returned 12.82% and the index returned 14.59%. Insurance against a repeat of 2008 was paid for in every year that was not 2008.

**So what?**

Treat any worst-case optimisation over a short history as a description of the sample, not a rule for the future. Before accepting one, count how many observations bind the solution. Here the answer was one, and a criterion set by a single year is not a criterion at all, whatever the linear program reports.

Minimax regret is the more usable of the two for a mechanical reason: minimising the gap to the best performer forces weight onto high-dispersion sectors, so the optimiser diversifies as a by-product instead of collapsing to a corner. Setting a floor on every weight, or fitting on overlapping rolling windows rather than one fixed decade, pushes maximin toward the same behaviour.

Report the holdout number next to the fitted one. Maximin's fitted worst year, a loss of 15.0%, read like a guarantee and then finished last on the measure it was built to protect. The gap between those two numbers is the honest estimate of how much the criterion knows.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
