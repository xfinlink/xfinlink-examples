# What API to Use for a Stock Screener

A stock screener asks a cross-sectional question: one date, several hundred companies, a few numbers each. Most market data APIs answer the opposite question, one company across many dates, and the mismatch shows up as a rate-limit bill rather than an error message. Four properties decide whether an API can run a screen at all: how many companies come back per request, where the universe of candidates comes from, whether the returned numbers are comparable in time, and what one full pass costs against the daily budget. Field coverage decides less than any of them, since almost every provider carries revenue and net income somewhere.

## Why does a screener need different data than a chart?

Charting one stock is a deep, narrow query: 250 rows, one symbol, one call. Screening is shallow and wide: one row each for 500 symbols, all as of the same moment. An endpoint keyed on a single symbol serves the first shape well and turns the second into 500 requests, which is why a daily budget of 25 calls can be generous for a dashboard and useless for a screen.

## How many companies come back per request?

Every figure below was read off the provider's own documentation on 1 August 2026. Terms change often, so confirm before committing to one.

| Source | Companies per request | Free plan |
|---|---|---|
| yfinance | `screen()` runs Yahoo's own screener; `size` defaults to 100, "maximum 250 (Yahoo)" | No key, no account |
| Alpha Vantage | Realtime Bulk Quotes "accepts up to 100 symbols per API request"; the documentation labels it a premium endpoint | "25 API requests per day" |
| Twelve Data | Comma-separated symbols in one call, but "each symbol consumes one API credit" | 8 API credits per minute, 800 a day |
| Massive (polygon.io redirects here) | Daily Market Summary returns OHLC, volume and VWAP for "all U.S. stocks on a specified trading date" in a single request | $0 Stocks Basic: "5 API Calls / Minute", "2 Years Historical Data", "End of Day Data", "Individual use" |
| xfinlink | 1 ticker on Free, 100 on Pro, 500 on Max | 100 requests a day, 12 months of history |

Sources in row order: the [yfinance screen reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.screen.html); the [Alpha Vantage documentation](https://www.alphavantage.co/documentation/) and [premium page](https://www.alphavantage.co/premium/); the Twelve Data [batch requests article](https://support.twelvedata.com/en/articles/5203360-batch-api-requests) and [pricing page](https://twelvedata.com/pricing); [massive.com/pricing](https://massive.com/pricing) and the [Daily Market Summary reference](https://massive.com/docs/rest/stocks/aggregates/daily-market-summary), reached because polygon.io returns a 301 redirect to massive.com; the xfinlink [docs](https://xfinlink.com/docs) and [pricing page](https://xfinlink.com/pricing).

Two of those deserve credit for solving the width problem directly. Massive's Daily Market Summary returns one trading day of prices for the entire US market in a single request, and the endpoint documentation lists it as included in all Stocks plans, the $0 tier among them; a pure price or volume screen needs nothing more than that endpoint and a loop over dates. yfinance goes further and hands over a finished screen, because Yahoo already built one, which suits a quick look at what is moving today.

Both stop where a fundamental screen starts. A ranking built on margins, returns on capital or growth needs financial statements attached to the same companies, for the whole candidate list rather than for the 250 rows a screener page will show. Twelve Data's batching is worth reading carefully for the same reason: one request, yes, but the credit meter still counts every symbol, so it buys latency rather than budget. Paid xfinlink plans cap a request at 100 or 500 tickers against daily budgets of 10,000 and 50,000, which makes a 500-name statement screen five calls.

## Where does the universe come from?

Someone has to decide which companies are candidates before any ranking happens, and that decision quietly determines the result. A screen over "large US technology companies" is really a screen over whatever list produced that phrase.

Index membership is the usual answer, and taking it from a current index page introduces an error that only appears later. Today's S&P 500 list contains the companies that survived to today. Screen the past with it and every bankruptcy, buyout and demotion has already been removed from the sample, which flatters any historical result built on top. The mechanics and a measured example sit in the guide on [survivorship bias in backtesting](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting).

`xfl.index()` returns membership for the S&P 500, Nasdaq-100, Dow Jones Industrial Average and Russell 2000, either as it stands now or as it stood on any past date through the `as_of` parameter. Each row carries an `entity_id` alongside the ticker, and that identifier is the join key: it stays with a company through renames and symbol changes, which is what keeps a screen from silently merging two businesses that happened to share a string. Details are in the [docs](https://xfinlink.com/docs).

## Do the numbers line up in time?

This is the failure that produces a plausible-looking ranking that means nothing. "Latest annual figures" is not one date. Eleven large companies, one request:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

names = ["AAPL", "MSFT", "NVDA", "JNJ", "MRK", "KO", "PEP",
         "XOM", "CVX", "HD", "WMT"]

df = xfl.metrics(names, period_type="annual", period="1y",
                 fields=["net_margin", "roe"])

out = df.sort_values("net_margin", ascending=False)
print(out[["entity_id", "ticker", "entity_name", "period_end",
           "net_margin", "roe"]].to_string(index=False))
```

```
 entity_id ticker       entity_name period_end  net_margin      roe
     29109   NVDA       NVIDIA CORP 2026-01-25    0.556025 0.763333
      8611   MSFT    MICROSOFT CORP 2026-06-30    0.403054 0.302335
      4072    JNJ JOHNSON & JOHNSON 2025-12-28    0.284565 0.328706
      4847    MRK    MERCK & CO INC 2025-12-31    0.280783 0.346995
      1675     KO      COCA COLA CO 2025-12-31    0.273399 0.407442
         1   AAPL         Apple Inc 2025-09-27    0.269151 1.519130
      5775    PEP       PEPSICO INC 2025-12-27    0.087730 0.403803
      2735    XOM  EXXON MOBIL CORP 2025-12-31    0.086817 0.111201
      3616     HD    HOME DEPOT INC 2026-02-01    0.085959 1.104815
      1553    CVX      CHEVRON CORP 2025-12-31    0.066686 0.065964
      7963    WMT       WALMART INC 2026-01-31    0.030992 0.219772
```

Eleven companies, one call, eight distinct fiscal year ends spanning nine months. Apple closed its year in September, Microsoft in June, Home Depot at the start of February. Ranking these against each other means ranking Microsoft's year through June 2026 against Chevron's year through December 2025, so in any period where conditions moved between those dates the ordering partly measures the calendar. The `period_end` column is what makes that visible and fixable, either by filtering to a common window or by holding the fiscal offset constant within a peer group.

The `roe` column carries a second warning worth reading before trusting a quality ranking. Apple at 1.52 and Home Depot at 1.10 are not four times better run than Merck at 0.35; they are companies whose sustained buybacks have shrunk the book equity sitting in the denominator. Return on equity rewards a small balance sheet, so a screen that ranks on it alone sorts partly on capital structure rather than on the business. Combining several factors is the standard correction, and a worked version is in the note on [building a multi-factor stock screen](https://xfinlink.com/blog/three-factor-stock-screen-python).

## Which one fits which screen?

For a price or volume screen over the whole US market, a grouped end-of-day endpoint is the right tool and Massive's free tier reaches it. For a glance at today's movers with no account at all, yfinance calls Yahoo's screener and returns up to 250 rows. Neither is built to rank several hundred companies on filed financial statements, which is what most screens turn into once the first idea survives contact with the data.

That job wants three things in one place: a universe you can pin to a date, statements normalised into the same columns for every company, and enough companies per request that a full pass is a handful of calls. xfinlink is built from SEC EDGAR public filings and market data, returns pandas DataFrames with `entity_id` on every row, and carries prices, statements and computed metrics behind the same key. A free key covers building and debugging the screen against a rolling twelve-month window; the $29 Pro plan raises the per-request cap to 100 tickers and opens the full history. What the free tiers include across providers is set out in the guide on [free stock market data APIs](https://xfinlink.com/blog/free-stock-market-data-apis).

## FAQ

**Can a stock screener run on a free API tier?**
Building and testing one, yes. A daily pass over a full index needs the wider per-request ticker caps that paid plans carry, because a screen that costs one call per company will exhaust any free budget before it finishes.

**Do I need a screener endpoint, or just the data?**
Usually just the data. A screener endpoint returns someone else's ranking rules; a data API returns the columns and lets the sort, the weights and the filters be yours. The second is more work on day one and the only option once the criteria stop matching a preset.

**Why do my screen results change when I rerun it a month later?**
Two causes dominate. Companies file new statements, which moves their numbers; and index membership changes, which moves the candidate list. Pinning the universe with a dated membership snapshot removes the second, so any change that remains is genuinely the data.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
