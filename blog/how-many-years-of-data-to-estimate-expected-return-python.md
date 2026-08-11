**How Many Years of Data Do You Need to Estimate Expected Return? Standard Errors in Python**

August 11, 2026 · ECONOMETRIC-RESEARCH

**What's the question?**

Every allocation decision beyond risk control needs an expected return, and the usual source of that number is the historical average. An average is an estimate, and estimates carry standard errors: the standard deviation of the estimate itself, with two either side giving roughly a 95% confidence interval.

For a mean return that standard error has an awkward property, set out by Robert Merton in 1980. It depends on how long the sample runs in calendar time and on nothing else. An asset with annual volatility sigma observed over T years yields a mean estimate whose standard error is sigma divided by the square root of T, so sampling the same thirty years daily rather than monthly multiplies the observation count by twenty-one and leaves the precision of the mean where it was. Volatility does not behave that way; the standard error of an estimated standard deviation shrinks with the square root of the observation count.

Risk can be measured from a short history. Expected return cannot. What follows puts a size on that gap.

**The approach**

Twelve exchange-traded funds cover US large cap, technology, small cap, developed markets outside the US, emerging markets, long and intermediate Treasuries, investment-grade corporates, gold, REITs, utilities and consumer staples. Their launch dates differ, supplying the range of sample lengths the exercise needs.

1. Pull daily total returns, dividends included, for each fund over its full history to 7 August 2026.
2. Keep each fund's longest unbroken run of observations, where a break of more than a month starts a new run.
3. Annualise: mean daily return times 252, daily standard deviation times the square root of 252.
4. Divide annual volatility by the square root of the number of years for the standard error of the mean, then take the t-statistic and 95% interval that follow.
5. Invert the t-statistic for the history a two-standard-error result requires, which is four divided by the squared Sharpe ratio.
6. Repeat the SPY calculation on weekly and monthly returns.

Each mean is tested against zero rather than against cash, the easier of the two tests.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "XLK", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ", "XLU", "XLP"]
TD = 252

px = xfl.prices(TICKERS, start="1990-01-01", end="2026-08-07",
                fields=["return_daily"], max_rows=200000)
px = (px.dropna(subset=["return_daily"])
        .sort_values(["ticker", "date"]).reset_index(drop=True))

# each fund contributes its longest unbroken run of daily observations
block = (px.groupby("ticker")["date"].diff().dt.days > 31).cumsum()
px = px[block == block.groupby(px["ticker"]).transform("last")]

rows = []
for tk in TICKERS:
    r = px.loc[px["ticker"] == tk, "return_daily"].to_numpy()
    years = len(r) / TD
    mean = r.mean() * TD
    vol = r.std(ddof=1) * np.sqrt(TD)
    se_mean = vol / np.sqrt(years)              # sigma / sqrt(calendar time)
    se_vol = vol / np.sqrt(2 * len(r))          # falls with the observation count
    rows.append({"ticker": tk, "years": years, "mean": mean * 100, "vol": vol * 100,
                 "se_mean": se_mean * 100, "se_vol": se_vol * 100,
                 "t": mean / se_mean, "need": 4.0 / (mean / vol) ** 2})

tab = pd.DataFrame(rows).sort_values("need")

# one history, three sampling frequencies
spy = px[px["ticker"] == "SPY"].set_index("date")["return_daily"]
for label, rule, per in [("daily", None, TD), ("weekly", "W-FRI", 52), ("monthly", "ME", 12)]:
    s = spy if rule is None else (1.0 + spy).resample(rule).prod().dropna() - 1.0
    m, v = s.mean() * per, s.std(ddof=1) * np.sqrt(per)
    print(f"{label:8} obs {len(s):5d}  se(mean) {v / np.sqrt(len(s) / per) * 100:5.2f}%"
          f"  se(vol) {v / np.sqrt(2 * len(s)) * 100:5.2f}%")

print(tab.round(2).to_string(index=False))
```

Full script with formatting and visualisation: [how-many-years-of-data-to-estimate-expected-return-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/econometric-research/how-many-years-of-data-to-estimate-expected-return-python.py)

**Output**

```
Daily total returns, 1996-01-02 to 2026-08-07

                        yrs    mean    vol     se      95% interval     t  yrs for t=2
GLD Gold               21.7  11.79%  18.2%  3.91%     4.1% to 19.5%  3.01          10
SPY US large cap       30.6  11.80%  19.2%  3.47%     5.0% to 18.6%  3.40          11
LQD IG corporates      24.0   4.63%   8.3%  1.70%      1.3% to 8.0%  2.73          13
XLK Technology         27.6  13.38%  26.0%  4.96%     3.7% to 23.1%  2.70          15
XLP Consumer staples   27.6   7.72%  15.2%  2.90%     2.0% to 13.4%  2.66          16
EEM Emerging markets   23.3  13.19%  27.1%  5.62%     2.2% to 24.2%  2.35          17
IEF Mid Treasuries     24.0   3.29%   6.8%  1.39%      0.6% to 6.0%  2.37          17
XLU Utilities          27.6   9.17%  19.2%  3.67%     2.0% to 16.4%  2.50          18
IWM US small cap       26.1  11.36%  23.9%  4.67%     2.2% to 20.5%  2.43          18
EFA Developed ex-US    24.9   8.59%  20.8%  4.16%     0.4% to 16.8%  2.06          23
VNQ US REITs           21.8  11.17%  28.3%  6.07%    -0.7% to 23.1%  1.84          26
TLT Long Treasuries    24.0   4.45%  14.3%  2.92%    -1.3% to 10.2%  1.53          41

Funds whose mean return clears two standard errors: 10 of 12

SPY, one history sampled three ways
              obs    mean    vol  se(mean)  se(vol)      t
daily        7699  11.80%  19.2%     3.47%    0.15%   3.40
weekly       1597  11.47%  17.6%     3.17%    0.31%   3.62
monthly       368  11.12%  15.2%     2.74%    0.56%   4.06

SPY after 30.6 years: mean 11.80% +/- 6.80% at 95% confidence
Volatility estimate: 19.2% +/- 0.30% on the same data
A three-year record at that volatility carries a standard error of 11.1% a year
```

**What this tells us**

Thirty and a half years of daily data on the largest equity fund in the world locate its average return between 5.0% and 18.6% a year. The same 7,699 rows put volatility at 19.2% plus or minus 0.30 points, a relative precision of 1.6% against 58% for the mean. Volatility is built from squared deviations, so every observation informs it, while the mean is close to the total move divided by elapsed time.

The bottom of the table is where the practical damage sits. Long Treasuries returned 4.45% a year over 24 years with a t-statistic of 1.53 and an interval from -1.3% to 10.2%, so 24 years of daily data cannot establish that they earned anything at all. REITs fail at 21.8 years despite an 11.17% average, because 28.3% volatility swamps it. The last column converts each Sharpe ratio into the history it implies: 10 years for gold, 11 for US large cap, 15 for technology, 41 for long Treasuries. Two standard errors is the loosest bar in common use, and nothing here clears it in under a decade.

Sampling frequency changes almost nothing about the mean. Moving from daily to monthly cuts the observation count by 21 times while the standard error improves only from 3.47% to 2.74%, and even that gain traces to mild negative autocorrelation rather than sample size, visible in volatility falling from 19.2% to 15.2%. Over the same step the volatility estimate loses precision by a factor of 3.6, from 0.15 to 0.56 points; the observation count alone implies 4.6, and the lower monthly volatility accounts for the difference.

**So what?**

Treat a historical average as an interval rather than an input. An optimiser fed 11.80% for equities builds a different portfolio from one fed 5.0%, and the data supports both; rerunning an allocation at the lower bound of each interval exposes how much structure rests on precision that was never there.

Rank assets on the years-needed column before ranking them on the mean. Anything requiring 41 years to show a non-zero return does not belong in a return-driven tilt, and earns its place through correlation with everything else. That is the honest case for long Treasuries, and it survives without a return forecast.

The same arithmetic settles most track-record arguments: three years at equity-like volatility carries a standard error of 11.1% a year, so a manager ahead of a benchmark by 4% a year over that span has produced a t-statistic near 0.4. Volatility and correlation are estimable from a few years of daily data, which is why volatility targeting holds up where return-forecast-driven allocation does not. Build the parts the data can pin down, and size the rest to the interval.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
