# Why Beta Differs Between Data Sources

Beta is not a property of a stock. It is the output of a regression, and that regression has four settings: how far back the sample reaches, how often returns are sampled inside it, which index stands in for "the market", and whether the raw slope is shrunk toward one before publication. Change any setting and the number changes. Coca-Cola measures 0.35 on five years of monthly returns against SPY and −0.28 on one year of daily returns, on the same prices, over windows ending 30 June 2026. Neither figure is a mistake. If a published beta does not come with its window, frequency and benchmark attached, it cannot be reproduced, and the safest fix is to compute the number yourself from returns you control.

## What beta actually measures

Beta is the slope of a stock's returns regressed on the market's returns: the covariance between the two divided by the variance of the market. A beta of 1.3 says that when the index moves one per cent, this stock has historically moved about 1.3 per cent in the same direction, on average, at whatever sampling frequency was used.

That last clause carries more weight than it looks. Beta describes co-movement, not performance. Exxon Mobil returned 25.2% over the year to 30 June 2026 against 20.9% for SPY, and its daily beta over that same year was −0.36. The stock beat the index while moving against it on a typical day, because the days it gained were disproportionately days the index fell. A negative beta is not a broken calculation; it is a statement about timing, and it says nothing about the direction of the total return.

## The four settings that change the number

<div style="overflow-x:auto;margin:18px 0;">
<table style="width:100%;border-collapse:collapse;font-size:14px;line-height:1.55;">
<thead>
<tr style="border-bottom:1px solid #2e2e2e;">
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">Setting</th>
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">Common choices</th>
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">What it does to the number</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Estimation window</td><td style="padding:8px 10px;">1, 2, 3 or 5 years</td><td style="padding:8px 10px;">Long windows are stable but describe a company that may no longer exist in that form. Short windows track the present and jump around.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Return frequency</td><td style="padding:8px 10px;">daily, weekly, monthly</td><td style="padding:8px 10px;">Daily sampling picks up microstructure noise and non-synchronous trading, which usually pulls beta toward zero for less liquid names. Monthly sampling gives 60 observations in five years, so the standard error is wide.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Benchmark</td><td style="padding:8px 10px;">S&amp;P 500, Nasdaq-100, a sector index, a home-country index</td><td style="padding:8px 10px;">Beta is defined against whatever you regress on. A semiconductor name has a lower beta against a semiconductor index than against a broad market index.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Adjustment</td><td style="padding:8px 10px;">raw, or shrunk toward 1</td><td style="padding:8px 10px;">The Blume adjustment publishes 0.67 × raw + 0.33, on the finding that betas drift toward one over time. It compresses every extreme reading, in both directions.</td></tr>
</tbody>
</table>
</div>

The adjustment step is the one that surprises people, because it happens silently. A provider that shrinks its betas will report a lower figure than you compute for a high-beta stock and a higher figure for a defensive one, even when the window, frequency and benchmark all match. Our write-up on [adjusted beta](https://xfinlink.com/blog/merrill-lynch-adjusted-beta-python) works through the arithmetic.

## How much does the number actually move?

Six large caps, four conventions, all measured against SPY on windows ending 30 June 2026:

<div style="overflow-x:auto;margin:18px 0;">
<table style="width:100%;border-collapse:collapse;font-size:14px;line-height:1.55;">
<thead>
<tr style="border-bottom:1px solid #2e2e2e;">
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">Stock</th>
<th style="text-align:right;padding:8px 10px;color:#fafafa;font-weight:500;">5y monthly</th>
<th style="text-align:right;padding:8px 10px;color:#fafafa;font-weight:500;">3y weekly</th>
<th style="text-align:right;padding:8px 10px;color:#fafafa;font-weight:500;">1y daily</th>
<th style="text-align:right;padding:8px 10px;color:#fafafa;font-weight:500;">5y monthly, Blume</th>
<th style="text-align:right;padding:8px 10px;color:#fafafa;font-weight:500;">Spread</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">AAPL</td><td style="text-align:right;padding:8px 10px;">1.09</td><td style="text-align:right;padding:8px 10px;">1.11</td><td style="text-align:right;padding:8px 10px;">0.87</td><td style="text-align:right;padding:8px 10px;">1.06</td><td style="text-align:right;padding:8px 10px;">0.24</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">KO</td><td style="text-align:right;padding:8px 10px;">0.35</td><td style="text-align:right;padding:8px 10px;">0.12</td><td style="text-align:right;padding:8px 10px;">−0.28</td><td style="text-align:right;padding:8px 10px;">0.56</td><td style="text-align:right;padding:8px 10px;">0.84</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">NVDA</td><td style="text-align:right;padding:8px 10px;">2.20</td><td style="text-align:right;padding:8px 10px;">2.14</td><td style="text-align:right;padding:8px 10px;">1.85</td><td style="text-align:right;padding:8px 10px;">1.80</td><td style="text-align:right;padding:8px 10px;">0.40</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">JPM</td><td style="text-align:right;padding:8px 10px;">0.97</td><td style="text-align:right;padding:8px 10px;">1.02</td><td style="text-align:right;padding:8px 10px;">0.82</td><td style="text-align:right;padding:8px 10px;">0.98</td><td style="text-align:right;padding:8px 10px;">0.20</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">XOM</td><td style="text-align:right;padding:8px 10px;">0.16</td><td style="text-align:right;padding:8px 10px;">0.06</td><td style="text-align:right;padding:8px 10px;">−0.36</td><td style="text-align:right;padding:8px 10px;">0.44</td><td style="text-align:right;padding:8px 10px;">0.80</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">PG</td><td style="text-align:right;padding:8px 10px;">0.38</td><td style="text-align:right;padding:8px 10px;">0.21</td><td style="text-align:right;padding:8px 10px;">−0.06</td><td style="text-align:right;padding:8px 10px;">0.58</td><td style="text-align:right;padding:8px 10px;">0.65</td></tr>
</tbody>
</table>
</div>

The median spread across the four conventions is 0.52. For Apple, JPMorgan and Nvidia the convention barely matters in relative terms; the stocks are clearly high-beta or clearly market-like whichever way the estimate is run. The defensive names are where the choice decides the answer. Coca-Cola reads as a low-beta stock on monthly data and as an inverse-correlation stock on daily data, and a hedge sized off the wrong one of those two would be sized wrong by the full 0.63.

## Which convention should you use?

Match the sampling frequency to the horizon you actually hold over. A pairs trade rebalanced weekly should not be hedged with a beta estimated from monthly returns, because the monthly figure says nothing about how the two legs move over a week. Portfolio-level risk measured quarterly is better served by monthly or weekly returns than by daily ones, which carry noise that washes out over a quarter.

Window length is a bias-variance trade. Five years of monthly data gives 60 observations, enough that the estimate is not dominated by a handful of days, and stale enough that a company which changed materially three years ago is still being described by its old self. One year of daily data gives roughly 250 observations and describes the company as it trades now. Where a business has been reshaped by an acquisition or a spin-off, the shorter window is usually the honest one.

Use the adjustment only if the number is a forecast of next period's beta rather than a description of the last one. For risk attribution over a period that has already happened, the raw slope is what you want. The [persistence of beta over time](https://xfinlink.com/blog/does-past-beta-predict-future-beta-python) is worth reading before deciding.

## Computing beta yourself in Python

Twelve lines, using split-adjusted closes and a benchmark of your choosing:

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

px = xfl.prices(["KO", "SPY"], start="2021-06-30", end="2026-06-30",
                fields=["adj_close"])
px["date"] = pd.to_datetime(px["date"])
wide = px.pivot_table(index="date", columns="ticker", values="adj_close")

monthly = wide.resample("ME").last().pct_change().dropna()
cov = monthly.cov()
print(cov.loc["KO", "SPY"] / monthly["SPY"].var())
```

Swapping `"ME"` for `"W"`, or handing `prices()` a different benchmark ticker, gives the other conventions from the table above. The [full CAPM regression](https://xfinlink.com/blog/capm-alpha-beta-regression-python) adds the intercept and the standard errors, which tell you how much confidence the slope deserves. Field names and the parameters `prices()` accepts are in the [docs](https://xfinlink.com/docs), and the free tier covers this workload.

## What a published beta field gives you

Some APIs serve beta as a stored field rather than leaving it to the caller. Alpha Vantage's `OVERVIEW` endpoint returns one for each company: a call for IBM on 20 August 2026 returned `"Beta": "0.705"`, alongside 54 other fields, with no window, frequency or benchmark in the response. For sorting a watchlist into rough risk buckets that is enough, and it saves you a price download.

It stops being enough the moment the number carries money. A hedge ratio, a cost-of-equity input in a discounted cash flow, or a risk model that nets betas across a book all need an estimate whose construction you can state and repeat. That means starting from returns. xfinlink serves the split-adjusted price history the calculation needs across the full [S&P 500 and beyond](https://xfinlink.com/pricing), so the window, the frequency and the benchmark stay decisions you make rather than assumptions you inherit.

## FAQ

**Can beta really be negative?**
Yes. It means the stock has moved opposite the index more often than with it over the sample. Coca-Cola and Exxon Mobil both show negative one-year daily betas against SPY in the window above, while both delivered positive returns over that year.

**Why does my beta not match the figure on a finance site?**
Almost always because the site used a different window, frequency, benchmark or adjustment. Unless the provider states all four, the figure cannot be reproduced, and matching it is not a useful goal.

**Does a longer estimation window give a more accurate beta?**
It gives a more stable one, which is not the same thing. Stability is an advantage when the underlying business has not changed and a liability when it has.

**Which benchmark should I regress against?**
The one that represents the capital you would otherwise hold. For a US large-cap book that is usually a broad US index; for a sector-neutral strategy it is the sector index, because the broad-market beta will attribute sector moves to the stock.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
