# Web Scraping vs a Financial Data API: What Breaks

Scraping wins when the question is small, current, and answered by a page that already exists: today's closing price for forty tickers, a company's sector label, the revenue line on a filing whose URL you already hold. It loses the moment the job requires history that is correct as of a past date, or coverage of companies that no longer trade under the symbol you know them by. Those failures are not parser bugs. They follow from what a public web page is: a rendering of the present, addressed by the present ticker symbol.

The choice, then, is not really scraping versus an API. It is whether the dataset you need exists on the pages you plan to scrape at all.

## When Is Scraping the Right Call?

For a one-off script that nobody depends on, scraping is fine and the setup cost is close to zero. A weekend project pulling the current S&P 500 roster off Wikipedia is a good example. As of September 2026 the "List of S&P 500 companies" page carries a components table with symbol, security name, GICS sector, GICS sub-industry, headquarters location, date added, CIK, and year founded, which is more structure than most free endpoints hand over, and Wikipedia text is released under CC BY-SA 4.0, which permits reuse and redistribution with attribution and share-alike terms (verified against Wikipedia's copyright policy, September 2026). Historical rosters live on a separate page, a point worth remembering later.

The SEC is the other genuinely good target. Filings are public, machine-readable, and free. The agency publishes JSON APIs at `data.sec.gov` covering submissions history, a single XBRL concept for one company, all facts for one company, and cross-sectional frames for one concept across filers, with no authentication and no key required (verified 3 September 2026 against sec.gov's EDGAR application programming interfaces page). Anyone parsing filing HTML by hand when those endpoints exist is doing avoidable work.

So the concession is real: for current, small, public, filing-derived facts, scraping and free public endpoints do the job. The interesting question is what happens at the edges of that description.

## What Breaks in a Scraper?

Markup churn is the failure everyone anticipates and the least interesting one. A quote page is rendered for readers, and its DOM carries no compatibility promise; a class name changes, a table becomes a virtualised list, and the selector that worked in April returns an empty list in May. That costs maintenance hours, not correctness, because a broken scraper usually fails loudly.

Rate limiting is the failure that costs correctness, because it fails quietly. The SEC publishes hard numbers: "our current maximum access rate is 10 requests per second", and it asks every automated client to declare a user agent in request headers, with the sample format `Sample Company Name AdminContact@<sample company domain>.com` (sec.gov webmaster FAQ, as of September 2026). The agency also states plainly that it reserves the right to block IP addresses that submit excessive requests, and that requests above the guideline "may be limited for a brief period" (sec.gov privacy and security information, as of September 2026). A scraper that quietly receives throttled or truncated responses mid-backfill leaves gaps in the middle of a dataset, and gaps in the middle are far harder to notice than a total failure.

Commercial sites are less explicit about numbers and more explicit about intent. Checked on 3 September 2026, `finance.yahoo.com/robots.txt` disallows `/xhr`, `/_remote`, `/caas/`, and `/lookup/` for all user agents, and serves a blanket `Disallow: /` to a named list of automated clients that includes Scrapy, CCBot, GPTBot, ClaudeBot, and Bytespider. Whatever a given scraper does about that file, the file states the site operator's position.

## Is Scraping Yahoo Finance Allowed?

Read the terms rather than the folklore. Yahoo's Terms of Service, section 2.4(i), prohibits users from attempting to "access or collect data ... from our Services using any automated means, devices, programs, algorithms or methodologies, including but not limited to robots, spiders, scrapers, data mining tools, or data gathering or extraction tools, for any purpose without our express, prior permission" (legal.yahoo.com, as of September 2026). Section 2.4(j) separately restricts using material from the Services to build "any database, archive, mobile application, data feed, widget or any other aggregated data source that competes with or constitutes a material substitute for the Services".

The most widely used Python wrapper says the same thing in its own words. The yfinance README states that the project "is not affiliated, endorsed, or vetted by Yahoo, Inc.", that it is "intended for research and educational purposes", and that "The Yahoo! finance API is intended for personal use only" (github.com/ranaroussi/yfinance, as of September 2026). None of that makes the library bad, and for a research notebook it remains a sensible choice. It does mean the licence question is unresolved for anything that ships to customers, which is covered in more depth in [can you use Yahoo Finance data commercially](/blog/can-you-use-yahoo-finance-data-commercially).

## What No Scraper Can Reconstruct

No amount of engineering fixes this part. A web page is keyed by the current ticker, and a ticker is not a company.

Meta Platforms changed its symbol from FB to META on 9 June 2022. Any scraper keyed on "META" reaches a page whose history may or may not be stitched, and a scraper keyed on "FB" reaches whatever now holds that symbol. A permanent entity identifier removes the question:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

info = xfl.resolve("META")["data"]["META"]["entities"][0]
print(info["name"], "| entity_id", info["entity_id"],
      "| ticker_valid_from", info["ticker_valid_from"])

df = xfl.prices(entity_id=2, start="2022-06-06", end="2022-06-13", fields=["close"])
print(df[["entity_id", "ticker", "date", "close"]].to_string(index=False))
```

```
Meta Platforms Inc | entity_id 2 | ticker_valid_from 2022-06-09

 entity_id ticker       date     close
         2     FB 2022-06-06 194.25000
         2     FB 2022-06-07 195.64999
         2     FB 2022-06-08 196.64000
         2   META 2022-06-09 184.00000
         2   META 2022-06-10 175.57001
         2   META 2022-06-13 164.25999
```

One request, one continuous series, and the ticker column records what the shares actually traded under on each date. The same identifier carries across fundamentals, metrics, insider transactions, and institutional holdings, so a merge never silently joins two different companies. The mechanics of that are set out in [entity resolution across ticker changes](/blog/entity-resolution-ticker-changes-python).

The second thing pages cannot supply is membership as of a past date. The SEC's own ticker map, `company_tickers.json`, holds three fields per record (CIK, ticker, company title) with no validity dates at all, verified 3 September 2026. Wikipedia's current-components table has the same property: it describes today. Backfilling a 2010 index roster from a 2026 page is exactly the mistake that produces [survivorship bias in a backtest](/blog/what-is-survivorship-bias-in-backtesting), and it inflates results in a direction that looks like skill. `xfl.index("sp500", as_of="2010-06-30")` returns the 500 names that stood in the index on that date, added and removed dates attached, including members such as AK Steel that left in 2011 and appear on no current page.

## Scraping vs an API: The Practical Comparison

| Requirement | Scraping public pages | SEC EDGAR APIs | xfinlink |
| --- | --- | --- | --- |
| Cost | Free plus maintenance | Free, no key | Free tier; paid from $29/month |
| Documented request limit | Set by robots.txt and terms | 10 requests/second | 100/day free, 10,000/day Pro |
| Daily price history | Whatever the page renders | Not published; EDGAR carries filings | Back to 1996 on paid plans, rolling 12 months free |
| Normalised financial statements | Parse filing HTML yourself | XBRL facts per company concept | Built from SEC EDGAR filings, one schema |
| Ticker-change history | Current page only | Current ticker map only | `ticker_valid_from` / `ticker_valid_to` |
| Index membership as of a past date | Not available | Not published | `xfl.index(..., as_of=)` |

Free-tier limits and plan detail are on [pricing](https://xfinlink.com/pricing); the full field list is in the [docs](https://xfinlink.com/docs).

## FAQ

**Is web scraping financial data legal?**
It depends on the site, and the site tells you. Yahoo's terms prohibit automated collection without prior permission (as of September 2026), while the SEC actively publishes machine-readable endpoints and asks only that clients identify themselves and stay under 10 requests per second. Read the terms of the specific source before building on it.

**Can a scraper get point-in-time data?**
Not from a page that renders the present. Point-in-time history has to be recorded as it happens and stored with validity dates, which is a property of the dataset rather than of the retrieval method.

**When is it worth paying for a data API?**
For a script that runs once, scraping is cheaper and that is a reasonable trade. The calculation changes when something depends on the data being right: a backtest whose results inform capital, a product with users, or research that has to be reproducible next year. At that point the cost of a wrong number exceeds the cost of the subscription.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
