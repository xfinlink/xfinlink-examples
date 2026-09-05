# Financial Data API Rate Limits: How Much Do You Need?

Rate limits stop a job for one of two reasons: the daily allowance runs out before the universe is covered, or the per-minute ceiling stretches the run past the window available for it. Sizing the requirement takes one line of arithmetic. Universe size, multiplied by the number of endpoints needed, multiplied by how often the data refreshes. Compare that number against the plan, expressed in the plan's own unit. The headline "requests per day" figure means very little until you know what the provider counts as a request, because that definition changes from vendor to vendor.

## What Does One Request Actually Count?

Four different units are in circulation, and pricing pages rarely put them side by side.

Some providers count HTTP calls, so one call spends one unit no matter how many rows come back. Others count symbols: a call naming three tickers spends three units. A third group publishes a per-minute ceiling and no daily cap at all, making wall-clock time the constraint rather than volume. Tiingo adds a fourth unit on top of requests, capping how many distinct symbols an account may touch in a month.

Two plans advertising the same number are therefore not offering the same thing. A plan allowing 500 calls a day where each call carries 100 tickers moves 50,000 ticker-pulls. A plan allowing 5,000 calls a day at one ticker per call moves 5,000. Ten times fewer, from a number ten times larger.

| Provider | Free tier | Entry paid plan | What the limit counts |
|---|---|---|---|
| xfinlink | 100 requests/day, max 40 per hour | $29/mo, 10,000 requests/day | One request per ticker; a 100-ticker call spends 100 |
| Alpha Vantage | "25 API requests per day" | "75 requests/min + premium support: $49.99/month" | HTTP calls; `TIME_SERIES_DAILY` takes a single `symbol` |
| Twelve Data | 8 API credits/min, "800 / day" | $29/mo, 55 credits/min, no daily cap | Credits per symbol: "(1 credit) * (3 symbols) = 3 credits" |
| EODHD | "20/day" | $19.99/mo, "100'000/day" and "1000/minute" | HTTP calls; "1 call per request (any length of price history)" |
| Massive (formerly Polygon.io) | "5 API Calls / Minute" | $29/mo, "Unlimited API Calls" | HTTP calls |
| Tiingo | 1,000 requests/day, 50/hour, 500 unique symbols/month | $30/mo, 100,000/day, 10,000/hour | Requests, plus a separate monthly unique-symbol cap |
| yfinance | No published limit | — | Undocumented |

Quoted figures were read from each provider's own pricing or documentation pages on 5 September 2026. yfinance publishes no rate limit at all; its documentation states the library is "not affiliated, endorsed, or vetted by Yahoo, Inc." and describes the Yahoo Finance API it calls as "intended for personal use only", so whatever ceiling exists is undocumented and can change without notice.

## How Do You Size Your Own Requirement?

Work in ticker-pulls, since that unit converts cleanly into every other one.

```
ticker-pulls per day = universe size × endpoints needed × refreshes per day
HTTP calls per day   = ceil(universe ÷ tickers per call) × endpoints × refreshes
rows per pull        = universe size × trading days of history
```

Three numbers, one of which most people get wrong: the universe. Guessing "about 500" for an index screener is close enough for a rough estimate and wrong enough to matter at the edges, so take it from the data.

```python
import math
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

universe = len(xfl.index("sp500"))

endpoints = 2            # one prices pull, one metrics pull
refreshes_per_day = 1    # once after the close
history_years = 1
trading_days = 252
tickers_per_call = 100   # the plan's batch cap

ticker_pulls = universe * endpoints * refreshes_per_day
rows = universe * history_years * trading_days
http_calls = math.ceil(universe / tickers_per_call) * endpoints * refreshes_per_day

print(f"universe            {universe}")
print(f"ticker-pulls / day  {ticker_pulls:,}")
print(f"rows / day          {rows:,}")
print(f"HTTP calls / day    {http_calls}")
```

```
universe            504
ticker-pulls / day  1,008
rows / day          127,008
HTTP calls / day    12
```

## What Do Real Jobs Cost?

**A single-name dashboard.** One ticker, prices and fundamentals, refreshed four times during the session: 8 ticker-pulls a day. Every free tier in the table clears this, some by a wide margin; Tiingo's Starter plan allows 1,000 requests a day, more headroom than most for a one-symbol script that polls aggressively. At this size throughput is not the deciding factor, so choose on history depth and data quality instead.

**A daily screener over an index.** The arithmetic above: 1,008 ticker-pulls, 127,008 rows, 12 HTTP calls if the plan carries 100 tickers per call. On xfinlink Pro that consumes roughly a tenth of the 10,000 daily requests and about 2.5% of the 5M daily row budget. On a provider that accepts one symbol per call, the same job becomes 1,008 separate HTTP calls. Against Alpha Vantage's free 25 requests a day it never finishes; against a 75-requests-per-minute paid plan it finishes in just over thirteen minutes of continuous calling.

**A watchlist refreshed intraday.** Fifty names, one endpoint, every fifteen minutes across a 6.5-hour session: 1,300 ticker-pulls a day. That exceeds the index screener while covering a tenth of the names. Refresh frequency, not universe size, drives the cost here, and it is the input most often left out of the estimate.

**A ten-year backtest on a point-in-time universe.** Any name that was in the index at any point, not the ones that survived to today. Taking mid-year snapshots of the S&P 500 from 2016 to 2026 gives 684 distinct companies, against 504 in the index on any single day; a daily-resolution roster raises the figure further. At 2,520 trading days each, that is roughly 1.7M rows and 684 ticker-pulls, which fits inside one day on a plan with a 5M row budget. The request count barely registers. The row count is the constraint, and it is the number to check first for any historical pull.

## Per Minute or Per Day: Which One Blocks the Job?

A daily cap decides whether the job finishes. A per-minute cap decides how long it takes.

Massive's free Stocks Basic plan allows five API calls a minute with no daily ceiling, so wall-clock time rather than volume decides when a job finishes. How long depends on how much ground a single call covers: its daily market summary endpoint returns every US ticker for one date in a single request, and its documentation on 5 September 2026 lists that endpoint as included on all stocks plans. A once-a-day index snapshot therefore costs one call there, while work that has to run symbol by symbol proceeds at three hundred an hour. Alpha Vantage's free 25 requests a day is a different kind of wall, and no amount of patience gets past it. Treat a per-minute number as a schedule question and a per-day number as a feasibility question.

Hourly throttles sit awkwardly between the two. xfinlink applies one only on the Free tier, which allows 100 requests a day with no more than 40 in any hour; paid plans have no hourly ceiling, so a batch job can spend its entire daily budget in a single run rather than pacing itself across the day.

## Why Rows and Pages Matter for Historical Pulls

Two limits rarely appear in vendor headline numbers and both bite on long history.

The first is a daily row budget. xfinlink publishes 500K rows a day on Free, 5M on Pro, 25M on Max and 250M on Redistribution. Ten years of daily bars for a few hundred names is between one and two million rows, so a historical extraction that costs almost nothing in requests can still be sized by rows.

The second is pagination. Paged results are capped at a maximum number of pages per call: 10 on Free, 50 on Pro, 100 on Max, 200 on Redistribution. A request for a page beyond the cap is rejected with an explicit error naming the ceiling, so a long paged extraction is worth sizing against the page cap alongside the row budget.

## What Happens When You Hit the Limit?

The standard answer across the industry is HTTP 429. xfinlink returns it with `retry_after_seconds` in the response body, and the Python client surfaces that wait time in the error message rather than a bare status code. Any unattended job should read that field and sleep, rather than retrying immediately into the same wall.

## FAQ

**Does batching tickers into one call reduce the quota consumed?**

It depends on the metering unit, which is the part of a pricing page worth reading closely. On xfinlink and on Twelve Data the count is per symbol, so a 100-ticker call spends 100 units exactly as 100 separate calls would; what batching saves is round trips and connection overhead, not budget. On providers that count HTTP calls, batching does reduce consumption, but only where the endpoint accepts multiple symbols in the first place, and many historical endpoints do not.

**Which number should be compared first when choosing a plan?**

Convert everything into ticker-pulls per day and compare that. It is the only unit that survives translation between providers, and it is the unit xfinlink meters in directly, which removes a conversion step. For historical work, check the daily row budget immediately afterwards.

**What throughput does a free tier realistically support?**

Enough for one symbol, or a handful, refreshed a few times a day. The free tiers in the table above that publish a daily cap sit between 20 and 1,000 requests a day, and none of those will carry a screener that pulls several hundred names one at a time. The exception is a market-wide endpoint returning every ticker in one response, which collapses that job into a single call. Treat a free key as a way to validate the data before paying rather than as a production budget.

Full limits for every tier are listed on the [pricing page](https://xfinlink.com/pricing) and in the [API documentation](https://xfinlink.com/docs). For the cost side of the same decision, see [what a financial data API costs](https://xfinlink.com/blog/what-does-a-financial-data-api-cost) and [how to choose a financial data API](https://xfinlink.com/blog/how-to-choose-a-financial-data-api). For sizing a historical job in particular, [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting) covers what the row count has to include.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
