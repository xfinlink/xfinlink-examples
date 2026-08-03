# Real-Time vs End-of-Day Market Data: Which Do You Need?

Real-time market data delivers trades and quotes as the exchange publishes them. Delayed data delivers identical content behind a fixed interval, fifteen minutes at both vendors checked for this guide. End-of-day data delivers one row per security per session, published once the market has closed. The tier drives the bill far more than it drives the quality of any number: as of 3 August 2026, Massive lists its real-time stock plan at $199 a month and the 15-minute delayed plan directly below it at $29. Work that acts during the session needs the fast tier. Backtests, screens, valuation models and dashboards that refresh each morning do not, and paying for latency they never use is the most common way to overspend on market data.

## What do the three tiers actually mean?

Three labels, one underlying feed.

Real-time carries each trade and quote continuously while the session runs. Delayed carries the same content on a lag: Alpha Vantage's documentation states that setting `entitlement=delayed` "will return 15-minute delayed intraday time series", and Massive labels two of its stock plans "15-minute Delayed Data" (both read 3 August 2026). End-of-day carries a single row per security per session, holding open, high, low, close and volume alongside the dividend and split events that keep a series comparable across years.

The closing price does not differ between them. A close on an end-of-day file is the number the real-time feed printed on the final trade of the session, and the split adjustment applied to it later is a separate matter from delivery speed. What changes across the tiers is when the number arrives, and what a vendor charges to hand it over that early.

## Why does real-time cost more than end-of-day?

The ladder is visible on the price pages themselves. Massive sells four stock plans, and latency is the axis that separates them: end-of-day free, 15-minute delayed at $29 and $79, real-time at $199. Alpha Vantage sells the entitlement rather than the plan, exposing an `entitlement` parameter on its intraday, daily-adjusted and quote endpoints where `realtime` and `delayed` are the two values; premium access starts at $49.99 a month for 75 requests a minute.

| Source and plan | What arrives | Monthly price | Depth or limit |
|---|---|---|---|
| Twelve Data Basic | "Real-time US equities and ETFs" | Free | 8 API credits/min, 800/day |
| Massive Stocks Basic | "End of Day Data" | $0 | 2 years, 5 calls/min |
| Massive Stocks Starter | "15-minute Delayed Data" | $29 | 5 years, unlimited calls |
| Massive Stocks Advanced | "Real-time Data" | $199 | 20+ years, unlimited calls |
| Alpha Vantage premium | `realtime` or `delayed` entitlement | From $49.99 | 75 requests/min at entry |
| xfinlink Free | End-of-day US equities and ETFs | $0 | 12 months, 100 requests/day |
| xfinlink Pro | End-of-day US equities and ETFs | $29 | Prices from 1996, 10,000 requests/day |

Every figure was read off the provider's own pages on 3 August 2026: [massive.com/pricing](https://massive.com/pricing), the [Alpha Vantage premium page](https://www.alphavantage.co/premium/) and [documentation](https://www.alphavantage.co/documentation/), [twelvedata.com/pricing](https://twelvedata.com/pricing), and the xfinlink [pricing page](https://xfinlink.com/pricing).

That ladder is not universal, and the exception is worth knowing. Twelve Data advertises "Real-time US equities and ETFs" on its free Basic plan at 8 API credits a minute. For a watchlist of a dozen symbols that is a genuinely free real-time quote. For a screen across several hundred names it is a budget that runs out long before the universe does, which is the shape of the tradeoff at the free end of this market generally.

## Which jobs genuinely need real-time?

Anything that acts before the session closes. Order routing, live risk limits, market making, an alert that has to fire while the move is still happening. There is also a category of measurement that only the fast tier can serve, which is the path a price took during the day rather than where it finished.

Apple on 31 July 2026 makes the point cleanly. The stock closed at $308.91, down 7.4 per cent after September-quarter guidance came in below expectations, having fallen "as much as 9.7 per cent intraday" according to Yahoo Finance's report that day. An end-of-day row records the close and the volume behind it. The 9.7 per cent low is not in that row and cannot be recovered from it. A strategy that would have been stopped out on the way down needs the feed that saw the way down.

## What runs perfectly well on end-of-day prices?

Anything whose decision arrives after the close, which is most research. One row per session is the native resolution of a daily strategy, a monthly rebalance, a factor sort or a valuation screen. Pulling a week of Apple takes one call:

```python
import xfinlink as xfl
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices("AAPL", period="1w", fields=["close", "volume", "return_daily"])
print(df[["ticker", "date", "close", "volume", "return_daily"]].to_string(index=False))
```

```
ticker       date  close    volume  return_daily
  AAPL 2026-07-27 336.91  45246885      0.011681
  AAPL 2026-07-28 340.08  50765695      0.009409
  AAPL 2026-07-29 338.19  48852885     -0.005558
  AAPL 2026-07-30 333.43  55501839     -0.014075
  AAPL 2026-07-31 308.91 127398021     -0.073539
```

The close matches the reported figure to the cent, and the volume column carries the confirmation: 127 million shares against 45 to 56 million on each of the four preceding sessions. A daily file describes the event with the two numbers that a daily strategy would have traded on.

Freshness on this tier is a question of when the file lands rather than how many milliseconds old a quote is. This guide was written on Monday 3 August 2026 and the most recent row above is Friday 31 July, the last session that has happened. For a research loop, "current through the last close" is the definition of current.

Two other guides cover what else that loop needs: [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting) on splits, dividends and point-in-time universes, and [what API to use for a stock screener](https://xfinlink.com/blog/what-api-to-use-for-a-stock-screener) on getting a whole universe back per request.

## What should you check before paying for the faster tier?

How many of your decisions get made while the market is open. If the honest answer is none, latency is not the constraint; history depth and universe coverage are, and those are priced on a different axis.

What you may do with the data afterwards. The yfinance documentation states the project is "not affiliated, endorsed, or vetted by Yahoo, Inc.", that it is "intended for research and educational purposes", and that "the Yahoo! finance API is intended for personal use only", which settles the question for anything client-facing before coverage even comes up. Commercial vendors answer it in writing instead; xfinlink prices redistribution with end-user display rights at $249 a month.

How far back the plan reaches, because vendors bundle depth with speed. Massive's real-time plan carries 20+ years while the delayed plan two steps below carries 5, so a buyer who needs the history and not the speed pays $170 a month for the wrong half of the upgrade. The $29 xfinlink Pro plan reaches daily prices back to 1996 and statements back to 1950 without touching the latency question at all.

How much data one request returns. A per-symbol quote endpoint burns a request budget faster than the headline number suggests; the call above returned a full week as a pandas DataFrame with the permanent entity id, the company name and the sector already attached, and the same call shape returns 250 rows for a year.

For work that reads the close, the sensible buying order is depth first, universe second, speed last. A free xfinlink key is enough to check the shape of the data against your own pipeline before any of that gets decided: see the [docs](https://xfinlink.com/docs) for the per-endpoint limits and [pricing](https://xfinlink.com/pricing) for the plan table. The [free stock market data APIs](https://xfinlink.com/blog/free-stock-market-data-apis) guide compares what the free tiers across this market actually include.

## FAQ

**Is a 15-minute delayed price different from the closing price?**
No. During the session a delayed feed shows a price fifteen minutes old; after the close the two agree, because the final print is the final print. Delay affects intraday decisions only.

**Can a daily strategy be backtested on end-of-day data alone?**
Yes, provided the strategy trades at prices the file contains, such as the close or the next open. A rule that assumes a fill at an intraday level the daily row never held is untestable on daily data, and reporting it as tested is worse than not testing it.

**What does an xfinlink price row contain?**
Daily open, high, low, close, split-adjusted close, total return, volume, plus dividend and split events, with a permanent entity id and the company's sector on every row. Data is built from SEC EDGAR public filings and market data, covering US-listed equities and ETFs.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
