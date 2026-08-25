# How Much Do Profits Move When Sales Move? Operating Leverage Regression in Python

August 25, 2026 · ECONOMETRIC-RESEARCH

**What's the question?**

Operating leverage is the share of a company's cost base that does not shrink when sales fall. Rent, factory depreciation and a salaried engineering team cost the same whether revenue rises 8 percent or drops 8 percent, so a business built on those costs turns a small revenue move into a large profit move in both directions. Where costs are mostly goods bought and hours paid for, profit tracks sales close to one for one.

The quantity that captures this is the degree of operating leverage: the percentage change in operating income divided by the percentage change in revenue. A value near 1 means costs scale with sales; a value of 4 means a 10 percent revenue decline takes 40 percent of operating income with it. Textbooks compute that ratio from a single year, which one noisy year can send anywhere. Estimating it as a regression slope across a decade of filings is more stable, and it produces a standard error, so the estimate can be judged rather than trusted.

**The approach**

The sample starts from the current S&P 500 roster. Financials and Real Estate are excluded, because operating income for a bank or a landlord is not built the same way as for a manufacturer.

1. Pull eleven consecutive annual filings, FY2015 through FY2025, for revenue and operating income.
2. Require operating income to be positive and operating margin to be at least 3 percent in every one of those years. A profit base near zero or flipping sign produces percentage changes in the hundreds, which is arithmetic rather than economics.
3. Compute ten annual percentage changes in revenue and ten in operating income, then regress the profit changes on the revenue changes. The slope is the estimated degree of operating leverage.
4. Keep the p-value alongside each slope, and name a company only where its slope is distinguishable from zero at the 5 percent level.
5. Compute the annualised volatility of daily returns over 2015 to 2025 and compare it with the slope.

166 companies clear every screen and hold a continuous daily price series under a single symbol across the window. Names without one drop from the sample.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

roster = xfl.index("sp500")
tickers = sorted(roster["ticker"].dropna().unique())

frames = [xfl.fundamentals(tickers[i:i + 100], period_type="annual",
                           start="2014-06-01", end="2026-06-30",
                           fields=["revenue", "operating_income"], max_rows=20000)
          for i in range(0, len(tickers), 100)]
fund = pd.concat(frames, ignore_index=True)
fund = fund[~fund["gics_sector"].isin(["Financials", "Real Estate"])]
fund = fund[fund["fiscal_year"].between(2015, 2025)]

rows = []
for ticker, g in fund.groupby("ticker"):
    g = g.sort_values("fiscal_year")
    if len(g) != 11 or g["operating_income"].min() <= 0:
        continue
    if (g["operating_income"] / g["revenue"]).min() < 0.03:
        continue

    d_rev = g["revenue"].pct_change().dropna().to_numpy() * 100
    d_opi = g["operating_income"].pct_change().dropna().to_numpy() * 100
    fit = stats.linregress(d_rev, d_opi)

    px = xfl.prices(ticker, start="2015-01-01", end="2025-12-31",
                    fields=["return_daily"])
    px = px[px["ticker"] == ticker].dropna(subset=["return_daily"])
    if len(px) < 2500:
        continue

    rows.append({"ticker": ticker, "slope": fit.slope, "r2": fit.rvalue ** 2,
                 "pval": fit.pvalue, "opinc_sd": d_opi.std(ddof=1),
                 "vol": px["return_daily"].std(ddof=1) * np.sqrt(252) * 100})

res = pd.DataFrame(rows).sort_values("slope", ascending=False)
sig = res[res["pval"] < 0.05]
print(f"{len(sig)} of {len(res)} slopes significant; "
      f"{(sig['slope'] > 1).sum()} of those above 1.0")
```

Full script with formatting and visualisation: [operating-leverage-regression-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/econometric-research/operating-leverage-regression-python.py)

**Output**

![Ten highest and ten lowest operating leverage slopes, and operating income variability against share price variability by leverage quartile](/blog-images/operating-leverage-regression-python.png)

```
--------------------------------------------------------------------------
OPERATING LEVERAGE, FY2015-FY2025
slope = percent change in operating income per 1 percent change in revenue
--------------------------------------------------------------------------
sample: 166 current S&P 500 members outside Financials and Real Estate
        with 11 consecutive annual filings, operating income positive and
        operating margin at least 3% in every year
slopes distinguishable from zero at the 5% level: 85 of 166

HIGHEST SLOPES (5% significant only)
ticker sector                    slope    R2  op inc sd  stock vol
GLW    Information Technology     8.77  0.70     109.0%      30.0%
TEL    Information Technology     8.46  0.56     116.0%      27.0%
SBUX   Consumer Discretionary     6.79  0.68      74.7%      28.7%
ROST   Consumer Discretionary     5.50  0.91     102.4%      31.1%
FDX    Industrials                4.53  0.56      53.4%      31.9%
GWW    Industrials                4.20  0.74      23.3%      27.8%
CAT    Industrials                4.12  0.44      99.0%      30.0%
DG     Consumer Staples           3.79  0.83      23.3%      30.5%
NVDA   Information Technology     3.76  0.65     211.0%      48.7%
LH     Health Care                3.35  0.49      41.6%      26.1%

LOWEST SLOPES (5% significant only)
ticker sector                    slope    R2  op inc sd  stock vol
SNPS   Information Technology    -5.35  0.48      23.9%      33.0%
DVA    Health Care               -2.61  0.54      29.4%      31.7%
CPRT   Industrials                0.05  0.49      10.2%      26.5%
CI     Health Care                0.44  0.87      30.9%      30.4%
RSG    Industrials                0.74  0.63       8.5%      18.6%
AEE    Utilities                  0.79  0.45      11.5%      21.6%
GD     Industrials                0.82  0.42       8.8%      22.1%
HPQ    Information Technology     0.88  0.45      25.2%      34.4%
ROP    Information Technology     0.93  0.81      12.3%      22.6%
PH     Industrials                0.93  0.46      13.2%      30.9%

median slope, all 166 companies:  1.14
median slope, 85 significant:      1.65
significant slopes above 1.0: 72 of 85   below 0: 2 of 85

Spearman rank correlation with the slope
  volatility of annual operating-income growth: rho = +0.324, p = 2.0e-05
  annualised stock volatility:                  rho = +0.312, p = 4.3e-05

AVERAGES BY OPERATING-LEVERAGE QUARTILE
quartile       n  mean slope   op inc sd   stock vol
Q1 lowest     42       -0.51       24.7%       26.2%
Q2            41        0.86       21.1%       27.0%
Q3            41        1.52       19.6%       27.9%
Q4 highest    42        3.37       48.3%       31.5%

MEDIAN SLOPE BY SECTOR
Consumer Discretionary    10    1.80
Health Care               26    1.80
Communication Services     6    1.60
Information Technology    33    1.23
Industrials               44    1.23
Materials                  6    1.11
Consumer Staples          19    0.78
Energy                     4    0.74
Utilities                 18    0.36
```

**What this tells us**

Half the regressions carry no usable signal. Only 85 of 166 slopes are distinguishable from zero at the 5 percent level. Ten annual observations will not identify an elasticity for a company whose revenue grows at a steady 6 percent every year, because there is almost no variation in the regressor to work with.

Where the slope is identified, it points one way. 72 of the 85 significant slopes exceed 1.0 and only 2 are negative, with a median of 1.65: a 1 percent revenue move typically carries operating income 1.65 percent. At Corning's 8.77 and TE Connectivity's 8.46, a 10 percent revenue move implies an operating income move near 88 and 85 percent, on fits with R² of 0.70 and 0.56. Corning's revenue rose from $9.1bn to $15.6bn across the window while operating income went from $1.3bn down to $509m and then to $2.3bn. Sector medians follow the same economics: Utilities last at 0.36, the expected result for regulated businesses earning a return on a rate base rather than on volume, against 1.80 for Consumer Discretionary and Health Care.

A negative slope is a modelling artefact rather than a business property. Synopsys grew revenue every single year of the window, from $2.24bn to $7.05bn, so the regressor barely varies while the profit path is driven by acquisition costs unrelated to that year's sales. The fit runs a downward line through a nearly vertical scatter.

The link to risk is real but narrower than expected. Operating income variability roughly doubles across the quartiles, from 24.7 percent in the lowest-leverage group to 48.3 percent in the highest, while share price volatility rises only from 26.2 to 31.5 percent, a factor of 1.2. Operating leverage shows up forcefully in the earnings line and faintly in the share price.

**So what?**

The practical use is scenario work rather than screening. A demand forecast expressed in revenue becomes an earnings forecast through this slope, and defaulting to 1.0 understates the downside for most large companies. For Caterpillar at 4.12 or FedEx at 4.53, a 10 percent revenue shortfall is a 40 percent operating income shortfall, which separates a disappointing year from a covenant conversation.

The gap between the earnings response and the price response is the part worth acting on. When profits swing twice as hard in the top quartile but the stock is only a fifth more volatile, trailing volatility is a poor proxy for fundamental risk in the tail. Position sizing on volatility alone will overweight names whose profits sit one bad quarter of volume away from halving.

Estimate the slope before relying on it, and check the p-value before quoting it. Half of these regressions identify nothing, and a degree of operating leverage quoted for a company with a decade of smooth revenue growth has no content behind it.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
