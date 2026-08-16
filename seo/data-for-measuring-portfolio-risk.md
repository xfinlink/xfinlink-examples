# What Data Do You Need to Measure Portfolio Risk?

August 16, 2026 · GUIDES

Measuring portfolio risk takes three things: a total-return price series for every holding (dividends and coupons reinvested, not just price), enough history to include at least one real crisis, and the holdings aligned to common trading dates so they can be compared. With those, the three standard risk measures follow directly: volatility from the standard deviation of returns, maximum drawdown from the running peak of cumulative return, and correlation from the return series of each pair. Price-only data understates the risk of income-heavy assets like bonds, and a short history hides the losses that matter most. The rest of this guide explains what each measure needs and how to assemble the data in Python.

## What are the core risk measures?

Three numbers cover most of what an investor needs to know about how a portfolio can hurt.

Volatility is the standard deviation of returns, usually annualised. It describes the everyday size of the moves. A portfolio at 18 percent annualised volatility will, in a typical year, swing within a wider band than one at 9 percent.

Maximum drawdown is the largest peak-to-trough drop in cumulative value. It describes the worst loss an investor actually had to sit through, which is the number that decides whether a plan gets abandoned at the bottom. Two portfolios with the same volatility can have very different drawdowns.

Correlation measures whether holdings move together. It runs from -1 to +1, and it is what makes a collection of assets less risky than the sum of its parts. Diversification is entirely a story about correlation, and correlation is not fixed: it tends to rise when markets fall, which is when diversification is needed most. A [conditional-correlation study of the sector ETFs](https://xfinlink.com/blog/sector-correlations-calm-vs-stress-python) shows the average pair moving from 0.40 in calm markets to 0.73 in stress.

## What data does each measure need?

Every one of these measures is built from a return series. The table below maps the measure to what it requires.

| Risk measure | What it needs | Data field |
| --- | --- | --- |
| Volatility | Daily returns for each holding | `return_daily` |
| Maximum drawdown | Cumulative total return over time | `return_daily`, compounded |
| Correlation | Aligned daily returns for every pair | `return_daily`, common dates |

The common thread is total return, not price. A stock's price return ignores dividends; a bond's price return ignores coupons, which are most of a bond's return. Using price-only data makes an income-heavy asset look worse than it is and distorts every risk number that follows. Xfinlink's `return_daily` field is a total-return series, so dividends and coupons are already reinvested, which keeps the comparison fair to bonds and dividend payers.

## Why does the length of history matter?

A risk estimate is only as honest as the worst period inside its window. Volatility measured over a calm two-year stretch will look reassuring and mean very little. Drawdown is worse: if the window contains no crisis, the maximum drawdown is simply the deepest dip that happened to occur, which understates what the portfolio can do.

The practical rule is to include at least one genuine stress event: the 2008 financial crisis, the 2020 pandemic crash, or the 2022 selloff in which stocks and bonds fell together. The 2022 episode is the one most risk models miss, because it breaks the assumption that bonds cushion equity losses. A [study of the 60/40 portfolio](https://xfinlink.com/blog/does-a-60-40-portfolio-cut-drawdowns-python) puts numbers on it: the mix cut the worst drawdown from 51.9 percent to 34.2 percent across 2008 to 2025, yet in 2022 stocks fell 18.2 percent and bonds 13.0 percent at the same time.

Free data tiers often cap history at a rolling year, which is too short for any of this. Xfinlink's free tier covers a one-year rolling window, and the paid plans unlock daily prices back to 1996, which is long enough to include every crisis named above. The [pricing page](https://xfinlink.com/pricing) lists the windows for each plan.

## A minimal example in Python

The following pulls total returns for three assets with different risk profiles, an equity index (SPY), long Treasuries (TLT), and gold (GLD), then computes all three measures.

```python
import numpy as np
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

px = xfl.prices(["SPY", "TLT", "GLD"], start="2015-01-01", end="2025-12-31",
                fields=["return_daily"])
r = px.pivot_table(index="date", columns="ticker", values="return_daily").dropna()

ann_vol = r.std() * np.sqrt(252) * 100          # annualised volatility, %
cum = (1 + r).cumprod()
max_dd = (cum / cum.cummax() - 1).min() * 100    # worst peak-to-trough, %

print(ann_vol.round(1))
print(max_dd.round(1))
print(r.corr().round(2))
```

The output shows why total return and a long window matter. The first block is annualised volatility, the second is maximum drawdown, and the third is the correlation matrix, all in percent except the correlations:

```
ticker
GLD    14.7
SPY    17.8
TLT    15.0
dtype: float64
ticker
GLD   -22.0
SPY   -33.7
TLT   -48.4
dtype: float64
ticker   GLD   SPY   TLT
ticker
GLD     1.00  0.04  0.28
SPY     0.04  1.00 -0.18
TLT     0.28 -0.18  1.00
```

Long Treasuries carry lower day-to-day volatility than stocks yet suffered a deeper maximum drawdown, 48.4 percent against 33.7 percent, because the 2022 rate rise hit them hard. A measure that stopped at volatility would have called TLT the safer asset. The negative SPY-to-TLT correlation of -0.18 is the reason a stock-bond mix diversifies at all, and gold's near-zero correlation to stocks (0.04) is why it is held as a separate diversifier. The full field list is in the [documentation](https://xfinlink.com/docs).

## Free tools versus a data API

For a quick look at a single stock, a free library is enough. yfinance, an open-source Python package that pulls from Yahoo Finance, is well suited to a one-off script (as of August 2026 it is free and open-source, and its own documentation notes it is intended for research and educational use and is not affiliated with or vetted by Yahoo). The friction appears when the work turns into a repeated process: aligning many tickers, reaching back through several crises, and trusting that a bond's return includes its coupons.

That is the point at which a data API earns its place. A single call returns aligned total-return series for a basket of holdings over decades of history, which is exactly the shape the risk calculations expect. The measures themselves are a few lines of NumPy; the data assembly is the part that a good source removes.

## FAQ

**What is the difference between volatility and drawdown?**
Volatility is the everyday size of returns, measured as their standard deviation. Drawdown is the largest cumulative loss from a prior peak. Volatility describes the ride; drawdown describes the worst moment of it.

**Do I need total-return data or are prices enough?**
Total return, for anything that pays income. Price-only data omits dividends and bond coupons and will understate the return, and therefore misstate the risk, of income-heavy holdings.

**How much history is enough to measure risk?**
Enough to include at least one real crisis. A window with no stress event produces a drawdown figure that reflects luck rather than risk. Data back to 2008 covers three distinct crises.

**Can I measure correlation from daily or monthly returns?**
Either works, but the frequency should match the decision. Daily returns give more observations and capture short-term co-movement; monthly returns are steadier and better matched to a long-horizon allocation.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
