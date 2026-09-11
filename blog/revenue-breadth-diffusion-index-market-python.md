**Does Revenue Breadth Predict the Stock Market? A Quarterly Diffusion Index in Python**

September 11, 2026 · MACRO-RESEARCH

**What's the question?**

A diffusion index counts how many members of a group are moving in one direction instead of measuring how far the average moved. The ISM manufacturing survey works this way: its headline figure is the share of purchasing managers reporting better conditions, not the size of the improvement. Breadth measures are read as early warnings, on the reasoning that participation turns before magnitude does.

Company filings support the same construction without a survey. Take every member of the S&P 500, compare each fiscal quarter's revenue against the same quarter a year earlier, and count the share that grew. That gives a quarterly reading of how widely revenue growth is spread across large American companies, and the question is where the reading sits in time relative to the stock market. If corporate revenue turns before prices do, the index is a forecasting input. If prices turn first, it is a record of something the market has already settled on.

**The approach**

The universe is the S&P 500 as it actually stood at each year end from 2004 to 2025, plus the current roster: 939 entities in the union, carried by entity id rather than ticker so a recycled symbol cannot swap one company for another mid-sample. Of those, 903 carry a quarterly revenue series long enough to form year-over-year comparisons, which produces 63,364 company-quarters.

1. Pull quarterly revenue for every entity starting in 2003, one year before the sample opens, so the earliest quarter has something to compare against.
2. For each company-quarter, compare revenue with the fourth quarter back, requiring the two period ends to fall 300 to 430 days apart so that a gap in a series cannot create a comparison spanning two years.
3. The index for a calendar quarter is the share of companies whose fiscal quarter ended inside it and grew.
4. The market is SPY total return, compounded within the same calendar quarters.
5. Correlate the index, in level and in quarter-on-quarter change, against market returns from three quarters before to three quarters after.
6. Repeat with 2008Q3 to 2009Q4 and all of 2020 removed, to see how much of the relationship lives in two crises.

A fiscal quarter's reports reach the public record during the quarter that follows it, so k = +1 is the first column in which the index could inform a decision. Everything at k = 0 or earlier is hindsight.

**Code**

```python
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ids = set()
for year in range(2004, 2026):
    ids |= set(xfl.index("sp500", as_of=f"{year}-12-31")["entity_id"].dropna().astype(int))
ids |= set(xfl.index("sp500")["entity_id"].dropna().astype(int))
ids = sorted(ids)

f = pd.concat(
    [xfl.fundamentals(entity_id=ids[i:i + 60], period_type="quarterly",
                      start="2003-01-01", end="2026-09-10",
                      fields=["revenue"], max_rows=200000)
     for i in range(0, len(ids), 60)], ignore_index=True)

f = f.dropna(subset=["revenue", "period_end"]).drop_duplicates(["entity_id", "period_end"])
f = f.sort_values(["entity_id", "period_end"])
grp = f.groupby("entity_id")
f["prev"] = grp["revenue"].shift(4)
gap = (f["period_end"] - grp["period_end"].shift(4)).dt.days   # guard against a missing quarter
d = f[(f["prev"] > 0) & gap.between(300, 430)].copy()
d["grew"] = d["revenue"] > d["prev"]
d["q"] = d["period_end"].dt.to_period("Q")

D = d.groupby("q").agg(n=("grew", "size"), diffusion=("grew", "mean")).loc["2004Q1":"2026Q1"]

spy = xfl.prices("SPY", start="2003-06-01", end="2026-09-10", fields=["return_daily"])
spy["q"] = spy["date"].dt.to_period("Q")
mkt = spy.groupby("q")["return_daily"].apply(lambda x: (1 + x).prod() - 1)

t = D.join(mkt.rename("mkt"), how="inner")
t["dD"] = t["diffusion"].diff()

calm = t.copy()
for lo, hi in [("2008Q3", "2009Q4"), ("2020Q1", "2020Q4")]:
    calm = calm[(calm.index < pd.Period(lo)) | (calm.index > pd.Period(hi))]

def leadlag(frame, col):                    # corr(series at q, market return at q+k)
    out = []
    for k in range(-3, 4):
        a, b = frame[col], frame["mkt"].shift(-k)
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append((k, r, r * np.sqrt((m.sum() - 2) / (1 - r ** 2))))
    return out

print(leadlag(t, "dD"), leadlag(t, "diffusion"))
print(leadlag(calm, "dD"), leadlag(calm, "diffusion"))

fwd = t.assign(nxt=t["mkt"].shift(-1)).dropna(subset=["dD", "nxt"])
reg = stats.linregress(fwd["dD"], fwd["nxt"])
print(reg.slope, reg.slope / reg.stderr, reg.rvalue ** 2, len(fwd))
```

Full script with formatting and visualisation: [revenue-breadth-diffusion-index-market-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/macro-research/revenue-breadth-diffusion-index-market-python.py)

**Output**

![Quarterly revenue breadth across the S&P 500 against SPY quarterly returns from 2004 to 2026, and the correlation between changes in breadth and market returns at leads and lags of three quarters](/blog-images/revenue-breadth-diffusion-index-market-python.png)

```
Revenue breadth against the market, point-in-time S&P 500
Panel:   939 entities from the 2004-2025 year-end rosters plus the current
         roster; 903 carry a usable quarterly revenue series,
         63,364 company-quarters
Index:   share of companies whose fiscal quarter grew revenue against the
         same quarter a year earlier, grouped by calendar quarter of period end
Sample:  89 quarters, 2004Q1 to 2026Q1, 598 to 756 companies per quarter
Market:  SPY total return compounded within each calendar quarter

Index    mean 70.96%   sd 13.63%   low 27.53% (2009Q2)   high 88.50% (2021Q3)

Correlation with the market return k quarters away
  k                       -3      -2      -1      +0      +1      +2      +3
  change in index     +0.207  +0.226  +0.477  +0.304  -0.003  -0.106  -0.073
  t                    +1.94   +2.14   +5.03   +2.96   -0.03   -0.98   -0.67
  index level         +0.375  +0.267  +0.144  -0.118  -0.278  -0.275  -0.211
  t                    +3.70   +2.56   +1.35   -1.10   -2.68   -2.63   -1.98

Same, excluding 2008Q3-2009Q4 and 2020Q1-2020Q4 (79 quarters)
  change in index     +0.057  -0.016  +0.114  +0.141  -0.065  +0.172  -0.003
  t                    +0.49   -0.14   +1.00   +1.25   -0.57   +1.51   -0.03
  index level         +0.092  +0.036  -0.024  -0.118  -0.176  -0.202  -0.246
  t                    +0.80   +0.31   -0.21   -1.05   -1.56   -1.79   -2.19

The two turning points, quarter by quarter
  quarter      index    change    market
  2008Q3     73.68%    -4.71pt    -8.86%
  2008Q4     48.29%   -25.39pt   -21.57%
  2009Q1     29.69%   -18.60pt   -11.23%
  2009Q2     27.53%    -2.15pt   +16.28%
  2009Q3     30.34%    +2.81pt   +15.38%
  2020Q1     53.04%   -11.50pt   -19.43%
  2020Q2     34.26%   -18.77pt   +20.16%
  2020Q3     49.25%   +14.99pt    +9.04%

Next quarter's market return regressed on this quarter's change in the index
  beta -0.004   t -0.03   R2 0.0000   n 87
```

**What this tells us**

The index moves with the market's past and not with its future. A one-quarter change in breadth correlates +0.477 with the market return of the quarter before it (t = 5.03) and +0.304 with the concurrent quarter. Looking forward, the correlations are -0.003, -0.106 and -0.073 at one, two and three quarters out, none of them distinguishable from zero. Regressing the next quarter's market return on this quarter's change in breadth gives a slope of -0.004, a t-statistic of -0.03, and an R-squared of 0.0000.

The level carries the opposite sign forward. High breadth follows strong markets, at +0.375 against the return three quarters earlier, and precedes slightly weaker ones, at -0.278 one quarter ahead (t = -2.68). A wide base of revenue growth describes where the cycle already is; it is not an argument for owning more equity.

Two crises carry most of the relationship. Removing 2008Q3 to 2009Q4 and all of 2020 leaves 79 quarters, and the backward correlation falls from +0.477 to +0.114 while the forward side stays near zero.

The quarter-by-quarter record shows why. Breadth was still at 73.68% in 2008Q3, a quarter in which the market fell 8.86%, and only then dropped 25.39 points in 2008Q4 and another 18.60 in 2009Q1. Its low of 27.53% arrived in 2009Q2, a quarter in which the market gained 16.28%. The 2020 sequence repeats the shape: a low of 34.26% in 2020Q2, alongside a 20.16% quarterly gain.

**So what?**

Revenue breadth is not a timing input. Zero forward correlation and an R-squared of 0.0000 leave nothing to trade on, and the version of this idea that reads a falling diffusion index as a sell signal is reacting to a number that troughs after the market has turned.

Its value is as a state variable. A reading far below the 70.96% sample mean identifies a real contraction in corporate revenue, which is useful for sizing risk and for deciding whether a drawdown has an earnings cause. Asking which sectors keep growing when breadth is at 30% is a question this series answers; asking what the market does next is not.

One caution about reading the series live. A quarter's index is complete only once that quarter's reports are in, which happens during the quarter after it. A partial reading taken mid-quarter is a different statistic, weighted toward whichever fiscal calendars report first.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
