**Total Return vs Price Return: What Backtests Miss**

August 15, 2026 · GUIDES

Price return measures the change in a stock's price and nothing else. Total return adds what the company handed to holders along the way: cash dividends, reinvested at the price on each ex-date, and the value of any shares distributed in a spin-off. For a stock that pays little, the two barely differ. For an income stock they can disagree on the sign: over the ten years to 21 May 2026, AT&T's price fell 33.9% while its total return was positive 45.3%. Any backtest, screen, or performance table built on a price series is measuring a different quantity from what an investor actually earned, and the difference is largest exactly where the strategy is aimed at income.

## What is the difference, exactly?

Price return over a window is the ending price divided by the starting price, minus one, with both prices restated to a consistent share basis so that stock splits do not create phantom losses.

Total return adds an assumption: each cash distribution immediately buys more shares at the ex-date price. The share count grows through the window, so the final value reflects both the price path and the shares accumulated along the way. Nobody reinvests dividends at that exact moment with no fees, and that is fine. The convention exists so that two securities with different payout policies can be compared on one number, and it is what index providers and fund reporting already use.

Both are correct measures of different things. Confusing them is what causes trouble.

## How large is the gap in practice?

Ten years of daily bars for nine US names, from 23 May 2016 to 21 May 2026, 2,514 trading days each. The gap column is the difference in percentage points.

```
        price ret  total ret  gap (pp)  price CAGR  total CAGR
SPY        261.9%     320.2%      58.3      13.74%      15.45%
AAPL      1165.1%    1280.7%     115.5      28.91%      30.04%
JNJ        106.6%     160.6%      54.0       7.53%      10.06%
KO          84.6%     143.7%      59.0       6.33%       9.32%
PG          78.8%     125.7%      46.9       5.99%       8.49%
XOM         73.3%     154.7%      81.4       5.66%       9.81%
MO          16.6%     110.7%      94.1       1.54%       7.74%
T          -33.9%      45.3%      79.1      -4.05%       3.81%
AMZN       670.6%     670.6%      -0.0      22.67%      22.67%
```

Altria is the clearest case. Its price rose 16.6% across a decade, which reads as a decade wasted; its total return was 110.7%, which is a respectable result for a stock that went nowhere. AT&T flips sign outright. Even the S&P 500 tracker gives up 1.71 percentage points a year when dividends are dropped, and compounded over ten years that is 58 points of cumulative return.

Amazon paid nothing across the window and its gap is exactly zero. That is the arithmetic check on the whole exercise: where there is no cash to reinvest, the two definitions must coincide, and they do.

AT&T's gap is not all dividends. The company completed the distribution of its WarnerMedia business at the close of business on 8 April 2022, and each distributed share converted into 0.241917 shares of Warner Bros. Discovery in the merger step that followed (AT&T investor relations, checked August 2026). On the next session the traded price fell from 24.14 to 19.63, down 18.7%, while `return_daily` records positive 6.15% for that day because the distributed shares count as value received. A price column alone books that Monday as a collapse.

One caution on reading the gap column. Apple's 115-point gap is the largest in the table despite a dividend yield below 1%, because a small yield compounding on a position that grew twelvefold produces a large absolute number. Annualised figures are the honest comparison: Apple gives up 1.13 points a year, Altria gives up 6.20.

## Does an adjusted close solve this?

Only if the adjustment includes dividends, and the field name will not tell you whether it does.

Two different adjustments travel under similar labels. Split adjustment restates the share basis so a 4-for-1 split does not look like a 75% crash; it changes nothing about dividends, and the values are stable once applied. Dividend adjustment rewrites the entire price history every time a dividend goes ex, which is why a value you stored last quarter no longer matches the same value today. Our guide on [split adjustment](https://xfinlink.com/blog/split-adjustment-explained) covers the first case in detail.

xfinlink keeps the two apart rather than folding them into one column. `adj_close` is split-only, so it stays stable across reloads, and total return lives in `return_daily`, documented on [the docs page](https://xfinlink.com/docs) as the field to use for performance and backtests. Both arrive in the same call, so the choice between price return and total return is a column name rather than a second data source.

Other tools resolve it differently. In yfinance, `Ticker.history()` takes `auto_adjust`, documented in the source as "Adjust all OHLC automatically?" with "Default: True", and the adjustment multiplies open, high, low and close by the ratio of Yahoo's adjusted close to the raw close, then drops the raw columns (verified against the yfinance source on GitHub, August 2026). The default frame therefore holds an adjusted series in a column named `Close`, and recovering the traded price needs `auto_adjust=False`. Alpha Vantage splits the question across endpoints: its plain daily endpoint returns "raw (as-traded)" open, high, low, close and volume, while the adjusted endpoint that carries adjusted close plus historical split and dividend events is documented as "a premium API function" (alphavantage.co/documentation, checked August 2026).

None of these designs is wrong. What matters is knowing which one is in front of you before a number goes into a report, because the answer moves ten-year results by tens of percentage points.

## How do you compute total return in Python?

Compound the daily total returns. Two lines after the data arrives.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices("T", start="2016-05-21", end="2026-05-21",
                fields=["adj_close", "return_daily"]).sort_values("date")

price_return = df["adj_close"].iloc[-1] / df["adj_close"].iloc[0] - 1
total_return = (1 + df["return_daily"].iloc[1:]).prod() - 1

print(f"price {price_return:.1%}  total {total_return:.1%}")
```

```
price -33.9%  total 45.3%
```

The first daily return is dropped because it measures the move into the starting day, which sits outside the window the price return covers. Skipping that detail overstates total return by one day and quietly breaks the comparison.

## When is price return the right measure?

When the question is about the traded price itself.

Signals defined on price levels belong on a price series: a 200-day moving average, a breakout above a prior high, a drawdown measured from a peak. A stop-loss triggers on the price a broker sees, not on a dividend-reinvested index, and backfilling distributions into that series moves every historical threshold. Chart-based rules should be generated on price and evaluated on total return.

Headline comparisons need matching conventions on both sides. Quoting a strategy's total return against an index level quoted as price return manufactures an advantage of roughly the index's dividend yield each year, which for the S&P 500 tracker above was 1.71 points annually.

The working rule is short. Generate signals on whichever series the rule actually observes, then measure every result, benchmark included, on total return. Our guide to [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting) covers the rest of the inputs a credible backtest needs, and the [decomposition of dividend yield into payout and price](https://xfinlink.com/blog/dividend-yield-payout-price-decomposition-python) shows where the income half of the return comes from.

## FAQ

**Does adjusted close include dividends?**

It depends on the provider, and it has to be checked rather than assumed. In xfinlink, `adj_close` is adjusted for splits only and `return_daily` carries the dividend-inclusive return, so the two never get mixed up in one column.

**Can total return be positive when price return is negative?**

Yes, and AT&T above is the example: down 33.9% on price, up 45.3% on total return over the same decade. Any stock with a high enough payout and a long enough window can produce this.

**Do total return figures account for tax?**

No. The standard convention reinvests the gross distribution, ignoring withholding tax and brokerage costs. Investors in a taxable account earn less than the quoted total return, and the shortfall depends on jurisdiction and account type, so it belongs in a separate step rather than in the return series.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
