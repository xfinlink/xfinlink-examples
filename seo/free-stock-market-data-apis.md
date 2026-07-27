# Free Stock Market Data APIs: What You Actually Get

Every free stock market data API charges in a currency other than money, and the usual four are request budget, history depth, data freshness, and the right to pass the data on. As of 27 July 2026 a free Alpha Vantage key is capped at 25 API requests per day; Twelve Data's Basic plan allows 8 API credits per minute and 800 per day; Massive's Stocks Basic plan allows 5 calls per minute against two years of end-of-day data; an xfinlink free key allows 100 requests per day against a rolling one-year window. yfinance asks for no key and no account, and its documentation points at Yahoo's terms for what may be done with the data afterwards.

The number on the pricing page is rarely the number that decides anything. What decides it is whether the plan finishes the job you had in mind, which depends on how much data arrives per request and how far back the window reaches.

## What do the free tiers actually include?

Every figure below was read off the provider's own pages on 27 July 2026. Terms move quickly at the free end of this market, so confirm before building against one.

| Source | Key or account | Free-plan limit | First paid tier |
|---|---|---|---|
| SEC EDGAR APIs | None. The SEC states the APIs "do not require any authentication or API keys to access" | XBRL disclosures from 10-K, 10-Q, 8-K, 20-F, 40-F and 6-K filings | None |
| yfinance | None | No published plan or rate limit; the docs state the project is "not affiliated, endorsed, or vetted by Yahoo, Inc." | None (open-source library) |
| Alpha Vantage | Yes | 25 API requests per day | $49.99/month for 75 requests per minute |
| Massive (polygon.io redirects here) | Yes | 5 API calls per minute, 2 years of history, end-of-day data | $29/month Stocks Starter: unlimited calls, 5 years of history, 15-minute delayed data |
| Twelve Data | Yes | 8 API credits per minute, 800 per day | $79/month Grow, or $66 billed annually |
| xfinlink | Yes | 100 requests per day, one ticker per request, rolling one-year window | $29/month Pro: 10,000 requests per day, prices to 1996, statements to 1950 |

Sources in row order: the [SEC's EDGAR API page](https://www.sec.gov/search-filings/edgar-application-programming-interfaces); the [yfinance documentation](https://ranaroussi.github.io/yfinance/); the [Alpha Vantage premium page](https://www.alphavantage.co/premium/); [massive.com/pricing](https://massive.com/pricing), reached because polygon.io returns a 301 redirect to massive.com; [twelvedata.com/pricing](https://twelvedata.com/pricing); the xfinlink [pricing page](https://xfinlink.com/pricing) and [docs](https://xfinlink.com/docs).

Two entries in that table are not products. The SEC publishes filing data as an open API because it is the regulator, and yfinance is an open-source client for an endpoint that belongs to somebody else. Both are free the way a library card is free: nothing to pay, and nothing promised.

## How many requests does the work actually take?

Daily request counts are easy to compare and easy to misread, because a budget means nothing until you know how much data comes back per call. One request for a year of Apple prices:

```python
import xfinlink as xfl
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices("AAPL", period="1y", fields=["close", "adj_close", "volume"])
print(df.shape)
print(df.tail(2).to_string(index=False))
```

```
(250, 8)
 entity_id ticker entity_name            gics_sector       date  close  adj_close   volume
         1   AAPL   Apple Inc Information Technology 2026-07-23 321.66     321.66 40795222
         1   AAPL   Apple Inc Information Technology 2026-07-24 333.02     333.02 47402209
```

250 rows for one request, arriving as a pandas DataFrame with the permanent entity identifier and the sector already attached. The free plan carries one ticker per request and 100 requests a day, with a 40-per-hour burst cap, which works out to one hundred one-year daily series in a day. That is enough to build something real and debug it properly, and short of what a nightly index-wide job needs. Pro raises the per-request cap to 100 tickers and the daily budget to 10,000.

Ask the same question of any free plan you are considering: how many rows does one call return, and how many calls does one finished answer take? A quote endpoint that returns a single price per call burns an 800-request budget faster than the number suggests.

## When is one year of history not enough?

Short windows are a deliberate part of how free plans are drawn: two years on Massive's Stocks Basic plan, one rolling year on the xfinlink free plan. A one-year window suits a dashboard, a screen of current values, or a script still under construction.

It does not suit anything with the word backtest attached to it. Measuring how [gross profitability predicts forward returns](https://xfinlink.com/blog/gross-profitability-forward-returns-python) needs decades of statements, not quarters, and the same is true of [volatility drag in leveraged ETFs](https://xfinlink.com/blog/why-do-leveraged-etfs-decay-volatility-drag-python), where the effect only separates from noise across a full cycle. Massive's Stocks Starter plan costs $29 a month and carries five years. The equivalent xfinlink plan is the same $29 and reaches daily prices back to 1996 and financial statements back to 1950, which is the difference between testing an idea through one regime and testing it through several.

## Which free tier fits which job?

Coursework, or an evening spent learning pandas: yfinance and the SEC APIs both do this without an account, and neither can be exhausted by a curious afternoon. The bill arrives later, when the script turns into something other people depend on, because an unofficial endpoint can change shape without notice and raw XBRL means writing and then maintaining your own mapping from tag names to a usable table.

Anything scheduled or shipped needs a key. A key is what converts a data source into a countable relationship, with a stated limit and a party who is responsible for the columns. Once that threshold is crossed the choice narrows to coverage. Options chains, intraday bars, forex and macro series point at Alpha Vantage, whose documentation runs to nine categories including realtime options and more than fifty technical indicators. US daily equity history and filing-derived financial statements, delivered as DataFrames, are what xfinlink returns, and a free key is enough to check that the shape fits before paying for the depth. Its data is built from SEC EDGAR public filings and market data, and the same endpoints are exposed to chat models through an [MCP server](https://xfinlink.com/blog/llm-financial-data-mcp-server).

## What about the licence?

The yfinance documentation is direct about this. It states that the project is "not affiliated, endorsed, or vetted by Yahoo, Inc.", that it is "intended for research and educational purposes", that "the Yahoo! finance API is intended for personal use only", and it directs users to Yahoo's terms for rights over the downloaded data. Clear, and it rules out a category of use before the question of coverage arises. SEC filings carry no equivalent constraint, being public record served by a public agency.

For anything that leaves your own machine, whether a client report or a dataset you resell, the licence question comes before coverage and price. Commercial vendors answer it in writing. xfinlink prices redistribution explicitly, at $399 a month, rather than routing it through a support conversation; the [pricing page](https://xfinlink.com/pricing) carries the full plan table and the [docs](https://xfinlink.com/docs) carry the per-endpoint limits.

## FAQ

**Is there a free stock market data API that needs no key at all?**
Two of them. yfinance is an open-source Python client, and the SEC's own APIs state that they do not require authentication or API keys. Neither comes with a support commitment.

**Can a free tier run a screen across the whole S&P 500?**
Not in one sitting. The xfinlink free plan carries one ticker per request and 100 requests a day, so 500 names spans five days of budget; Pro raises the per-request cap to 100 tickers, which turns the same screen into five calls.

**Do free tiers cover financial statements, or only prices?**
It varies. The SEC APIs return XBRL disclosures straight from 10-K and 10-Q filings, which is the raw source with no normalisation. A free xfinlink key returns statement data as a DataFrame with consistent field names across companies, inside the rolling one-year window.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
