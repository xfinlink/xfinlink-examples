**How Much Leverage Maximises Long-Run Growth? Kelly Sizing in Python**

September 11, 2026 · RISK-ANALYSIS

**What's the question?**

Borrowing to hold more of a rising asset lifts return and risk together, and past some point it lowers the rate at which capital compounds. Wealth compounds multiplicatively, so what accumulates is the average log return, and the log of a levered return carries a penalty that rises with the square of the position.

John Kelly set out where the two forces balance in 1956, and Edward Thorp carried the rule into markets. For an asset whose excess return over cash is mu and whose variance is sigma squared, the growth-optimal leverage is mu divided by sigma squared. The formula is exact; its inputs are estimates from a finite past.

Two questions. What leverage would have compounded a broad equity fund fastest since 2007, and would a manager who measured that number on one stretch of history have been right about the next?

**The approach**

1. Take a broad market fund, SPY, and the nine sector funds with a continuous daily series across the window: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. The cash leg is BIL, a Treasury bill fund, which sets the financing rate.
2. Keep the days on which all eleven series have a return: 4,844 of them, from 31 May 2007 to 10 September 2026, covering 2008, 2020, 2022 and a policy rate that went from zero to above 5 percent and back.
3. Build the levered daily return as L times the fund return minus L minus one times the bill return, resetting to L each day, as a leveraged fund does.
4. Search L from 0.00 to 6.00 in steps of 0.05, recording growth and the worst peak-to-trough loss at the winning leverage, at half of it, and at the largest leverage whose worst loss stayed inside 50 percent.
5. Split the window into two halves of 2,422 days. Fit the leverage on the first, apply it to the second, and compare against the unlevered fund and against the second half's own best leverage.

Two assumptions favour leverage: borrowing at the bill rate with no spread, and no lender calling the position in. The output prices a spread separately.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CASH = "BIL"
GRID = np.round(np.arange(0.0, 6.001, 0.05), 2)

px = pd.concat([xfl.prices(t, start="2006-01-01", end="2026-09-10",
                           fields=["return_daily"], max_rows=200000)
                for t in FUNDS + [CASH]], ignore_index=True)
R = px.pivot(index="date", columns="ticker", values="return_daily").sort_index().dropna()
cash = R[CASH].values


def growth(r):                        # annualised geometric growth rate
    if np.min(r) <= -1.0:
        return -1.0
    return float(np.expm1(np.log1p(r).sum() / len(r) * 252))


def maxdd(r):
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def levered(r, c, L, spread=0.0):     # constant leverage L, reset every day
    return L * r - (L - 1.0) * (c + spread / 252.0)


def curve(r, c, spread=0.0):
    return np.array([growth(levered(r, c, L, spread)) for L in GRID])


g_cash = growth(cash)
for f in FUNDS:
    r = R[f].values
    cv = curve(r, cash)
    L = GRID[int(np.argmax(cv))]                  # growth-optimal leverage
    half = round(L / 2, 2)
    capped = GRID[np.array([maxdd(levered(r, cash, x)) for x in GRID]) >= -0.50].max()
    print(f"{f}  1.0x {growth(r) * 100:6.2f}% dd {maxdd(r) * 100:6.1f}%  "
          f"L* {L:4.2f} {cv.max() * 100:6.2f}% dd {maxdd(levered(r, cash, L)) * 100:6.1f}%  "
          f"half {growth(levered(r, cash, half)) * 100:6.2f}% "
          f"keeps {(growth(levered(r, cash, half)) - g_cash) / (cv.max() - g_cash) * 100:.0f}%  "
          f"50% cap {capped:4.2f}  kelly {(r.mean() - cash.mean()) / r.var(ddof=1):4.2f}")

mid = len(R) // 2
A, B = R.iloc[:mid], R.iloc[mid:]
for f in FUNDS:
    LA = GRID[int(np.argmax(curve(A[f].values, A[CASH].values)))]
    cvB = curve(B[f].values, B[CASH].values)
    rb, cb = B[f].values, B[CASH].values
    print(f"{f}  fitted {LA:4.2f}  best {GRID[int(np.argmax(cvB))]:4.2f}  "
          f"second half 1.0x {growth(rb) * 100:6.2f}%  fitted {growth(levered(rb, cb, LA)) * 100:6.2f}% "
          f"dd {maxdd(levered(rb, cb, LA)) * 100:6.1f}%  best {cvB.max() * 100:6.2f}%")
```

Full script with formatting and visualisation: [growth-optimal-leverage-kelly-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/portfolio-construction/growth-optimal-leverage-kelly-python.py)

**Output**

![Annualised growth rate against leverage for a broad market fund and nine sector funds, and the best leverage of 2007-2017 against the best leverage of 2017-2026](/blog-images/growth-optimal-leverage-kelly-python.png)

```
============================================================================================
GROWTH-OPTIMAL LEVERAGE ON A BROAD MARKET FUND AND NINE SECTOR FUNDS
============================================================================================
Sample     SPY and the nine sector funds with a continuous daily series across the
           window; the cash leg is BIL, a Treasury bill fund
Window     2007-05-31 to 2026-09-10, 4,844 trading days, bills compounded at 1.35% a year
Method     levered return = L x fund return - (L - 1) x bill return, reset daily;
           growth is annualised geometric; L searched from 0.00 to 6.00 in steps of 0.05
Worst single session in the sample: -20.14%   optima sitting at the edge of the search: 0

                 unlevered      growth-optimal L        half of it      drawdown held to 50%
fund    vol     growth  maxDD     L   growth  maxDD    growth  maxDD      L    growth
SPY    19.7%   10.66%  -55.2%  2.70  17.33%  -93.3%   13.06%  -68.2%   0.85    9.48%
XLB    23.8%    7.03%  -59.8%  1.45   7.65%  -75.4%    6.02%  -46.8%   0.75    6.15%
XLE    30.3%    6.52%  -71.3%  1.05   6.53%  -73.4%    5.23%  -43.7%   0.60    5.60%
XLF    30.3%    5.11%  -82.7%  0.90   5.16%  -78.6%    4.20%  -49.0%   0.45    4.20%
XLI    21.6%    9.98%  -62.3%  2.20  13.93%  -91.3%   10.59%  -66.2%   0.70    7.85%
XLK    23.5%   16.49%  -53.0%  2.95  30.05%  -94.3%   22.10%  -69.7%   0.90   15.16%
XLP    14.7%    8.67%  -32.4%  3.65  17.55%  -84.1%   13.21%  -54.2%   1.60   12.13%
XLU    19.2%    7.41%  -46.5%  2.05   9.70%  -77.0%    7.49%  -47.3%   1.05    7.62%
XLV    17.3%    9.96%  -39.2%  3.15  18.15%  -87.0%   13.70%  -57.3%   1.30   12.02%
XLY    22.8%   10.68%  -59.0%  2.15  14.69%  -89.9%   11.21%  -62.4%   0.80    9.21%

Half the growth-optimal leverage kept 72% to 75% of the excess growth rate (median 74%).
Excess return over variance, the textbook Kelly fraction: SPY 2.76 against a searched optimum of 2.70.

Leverage fitted on the first half of the window, scored on the second half
  first half 2007-05-31 to 2017-01-10, second half 2017-01-11 to 2026-09-10
fund    fitted L   best L    growth at 1.0x   at fitted L   at best L   maxDD at fitted L
SPY      1.75      3.90         15.13%        23.10%       33.81%        -53.3%
XLB      1.10      2.00          9.35%         9.82%       11.88%        -40.8%
XLE      0.75      1.35         10.41%         9.24%       10.98%        -55.0%
XLF      0.40      2.15         11.32%         6.43%       15.31%        -19.0%
XLI      1.75      2.75         12.58%        17.75%       20.45%        -63.7%
XLK      2.20      3.45         24.65%        45.14%       53.55%        -63.7%
XLP      4.60      2.80          7.78%         7.59%       11.89%        -82.5%
XLU      1.85      2.30          9.42%        12.53%       12.94%        -58.9%
XLV      3.05      3.25         10.77%        19.29%       19.37%        -69.5%
XLY      2.10      2.20         11.70%        15.99%       16.02%        -70.5%
Fitted leverage beat 1.0x in 7 of 10 funds; median gap between fitted and best 0.95x; median growth
given up against the best possible 2.38 points.

Financing spread charged on the borrowed leg, broad market fund: 0bp -> 2.70x, 17.33%   100bp -> 2.45x, 15.49%   200bp -> 2.20x, 13.96%
============================================================================================
```

**What this tells us**

The broad market fund compounded at 10.66 percent a year unlevered, with a worst loss of 55.2 percent. Growth peaked at 2.70 times exposure and 17.33 percent a year, and the drawdown at that peak was 93.3 percent: arithmetically right and financially useless, since a fall of that depth closes a margin account long before the recovery arrives. The closed form agrees with the search, giving 2.76 against 2.70.

Growth is not symmetric around its peak; it rises gently to the left and falls steeply to the right. Half the winning leverage, 1.35 times, compounded at 13.06 percent with a 68.2 percent worst loss, keeping 73 percent of the growth above bills. The pattern repeats in every fund: half the optimum kept 72 to 75 percent of the excess growth. Kelly's fraction acts as a ceiling, and standing well below it costs little.

The winning leverage belongs to the window rather than to the asset, running from 0.90 on the financial sector fund, which lost 82.7 percent in 2008, to 3.65 on consumer staples, which lost 32.4 percent. Fitted on the first half and scored on the second, it missed by a median of 0.95 turns, nine of the ten misses pointing the same way because the later decade wanted more leverage. Fitted leverage still beat unlevered exposure in seven funds, surrendering a median 2.38 points to hindsight. The one fitted above what the later decade wanted, consumer staples at 4.60 against 2.80, compounded at 7.59 percent against the unlevered fund's 7.78, through an 82.5 percent drawdown.

**So what?**

Size leverage from the loss that can be survived, not from the growth peak. Holding the worst drawdown inside 50 percent allowed 0.85 times exposure on the broad fund and 0.45 on financials, both below full investment. An equity mandate with a 50 percent loss limit has spent its risk budget before borrowing anything.

Where leverage is used, take a fraction of the estimate. Half the optimum kept roughly three quarters of the growth above cash in all ten funds and cut the broad fund's worst loss by 25 points. Price the financing first: 100 basis points over bills moved the optimum from 2.70 to 2.45 and took 1.84 points off the growth rate.

Then re-estimate on the series actually held, because 2.70 describes one fund across one window and it moved by more than a turn between that window's halves.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
