# Financial Data for Academic Finance Research

Academic work sets a higher bar on data than most trading work does, and reproducibility is the reason. A referee has to be able to check the result, and a co-author has to be able to rerun the same code eighteen months later and land on the same table. Public sources cover a large part of what a paper needs at no cost: SEC EDGAR carries the filings and the XBRL facts inside them, the Kenneth French Data Library carries factor and portfolio returns, and FRED carries the macro series. What none of them carries is the piece most empirical equity papers turn on, which is a cross-section of companies as it stood on a past date, with a stable identifier per company across ticker changes, reconstructable months later without argument.

## What does a referee actually check?

Data objections in empirical finance cluster into four shapes, and they are worth knowing before the first regression rather than after the first rejection.

1. **Universe as of the date, not as of today.** A sample drawn from the current index membership and run backwards has already excluded every company that failed, merged, or was demoted, so the estimate that comes out is not the estimate the strategy would have produced. The [survivorship bias guide](/blog/what-is-survivorship-bias-in-backtesting) works through how large the distortion gets.
2. **Adjustment convention stated explicitly.** Split adjustment, dividend treatment, and whether a return series is total or price-only change the answer materially, and a paper that does not say which convention it used cannot be replicated even by its own authors.
3. **Identifier continuity.** Tickers are reassigned. A panel keyed on the ticker string silently splices one company's history onto another's, and the join looks perfectly healthy while it does so.
4. **A licence that permits the replication package.** Journals increasingly ask for code and data alongside the manuscript, and a source whose terms forbid redistribution makes that package impossible to supply.

## Which free sources are research-grade?

Several public sources are good enough to build a published paper on, and it is worth being precise about what each one gives.

The **SEC EDGAR APIs** are the strongest free source for anything filing-level. The SEC's own developer page, read on 6 August 2026, states that "these APIs do not require any authentication or API keys to access", and that "currently included in the APIs are the submissions history by filer and the XBRL data from financial statements". Latency is not the constraint: the same page states that "the submissions API is updated with a typical processing delay of less than a second; the xbrl APIs are updated with a typical processing delay of under a minute". The cost of EDGAR is engineering, not money, and the [EDGAR API versus fundamentals API comparison](/blog/sec-edgar-api-vs-fundamentals-api) sets out what that engineering involves.

The **Kenneth French Data Library** at Dartmouth is the default source for factor returns and is free to download. Read on 6 August 2026, it publishes the Fama/French three- and five-factor series at monthly, weekly and daily frequency, momentum and reversal factors, portfolio sorts on size, book-to-market, operating profitability and investment, industry portfolios at 5 through 49 groupings, and developed and emerging market factors by region, under the notice "Copyright Eugene F. Fama and Kenneth R. French". For a paper that needs benchmark factor returns rather than its own construction of them, there is no reason to rebuild what this library already publishes.

**FRED** covers the macro side and requires registration. Its terms of use, read on 6 August 2026, require applications to display the notice "this product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis", and place responsibility on the user for third-party content: users are "solely responsible for complying with any requirements or restrictions imposed on usage of the data series by their respective owners". Copyrighted series inside FRED are not automatically yours to republish.

**Alpha Vantage** offers a free key with, in its own words as of 6 August 2026, "the standard API usage limit (25 API requests per day)". Premium tiers on the same page start at "75 requests/min + premium support: $49.99/month". Twenty-five calls a day is a tutorial budget, not a panel budget.

| Source | Covers | Access | Practical limit for a paper |
| --- | --- | --- | --- |
| SEC EDGAR APIs | Filings, submissions history, XBRL facts | No key, no authentication | Parsing and normalisation are yours to build |
| Kenneth French Data Library | Factor returns, portfolio sorts, industry portfolios | Free download | Portfolio-level only, no company panel |
| FRED | Macro and rate series | Free, key required, attribution notice required | Series-owner restrictions pass through to you |
| Alpha Vantage | Prices and company data | Free key, 25 requests per day | Paid tiers begin at $49.99 per month |

## Can a paper be built on scraped market data?

Read the terms before the sample is built, because the answer for the most commonly scraped source is unambiguous. Yahoo's terms of service, read on 6 August 2026, prohibit users from "access or collect data, or attempt to access or collect data, from our Services using any automated means, devices, programs, algorithms or methodologies, including but not limited to robots, spiders, scrapers, data mining tools, or data gathering or extraction tools, for any purpose without our express, prior permission". They also state that "unless otherwise expressly stated, you may not access or reuse the Services, or any portion thereof, for any commercial purpose", and separately forbid using the material "to create any database, archive, mobile application, data feed, widget or any other aggregated data source that competes with or constitutes a material substitute for the Services".

Whether a funded research project counts as commercial is a question for a university's counsel. The replication-package question is simpler: a dataset assembled by automated collection against those terms cannot be redistributed alongside the manuscript. The [commercial-use guide](/blog/can-you-use-yahoo-finance-data-commercially) goes through the terms clause by clause.

## Where the free stack runs out

The gap is the company panel. EDGAR tells you what a company filed but not which companies were in an index on a given date; the French library gives portfolio returns but not the companies inside them; FRED is macro. Building a point-in-time universe from public sources means reconstructing index membership from historical announcements, then keeping identifiers stable through every rename, merger and delisting in the sample window. That reconstruction is the part of a methods section reviewers probe hardest.

The size of the problem is easy to measure. Pulling the S&P 500 as it stood at the end of 2005 returns 500 companies; the current roster returns 504; fewer than half of the 2005 names are in the index today. A study that starts from today's 504 and runs back twenty years is studying the survivors.

## What a workable research stack looks like

Factor returns from the French library, macro series from FRED, and filings from EDGAR when the unit of analysis is the filing itself. For the company panel, xfinlink returns index membership as of any past date, so the universe is a parameter rather than a project:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

roster_2005 = xfl.index("sp500", as_of="2005-12-30")
current = xfl.index("sp500")

print(len(roster_2005), len(current))
```

```
500 504
```

Entities carry a permanent identifier alongside the ticker, which is what keeps a panel intact when a company changes symbol, and the data endpoints accept that identifier directly, so a company that no longer holds its old ticker is still reachable. Financial statement data is built from SEC EDGAR public filings, with a `source` column on each row recording provenance. A free key runs one ticker per request with a twelve-month history window and 100 requests a day, enough to test whether the data fits the design before any money is spent; paid plans open the full history, with daily prices back to 1996 and financial statements back to 1950. The [docs](/docs) list the fields each endpoint returns and the [pricing page](/pricing) sets out the plan limits.

## FAQ

**Is SEC EDGAR enough on its own for an equity paper?**
For anything filing-level, yes. For a cross-sectional panel, no, because EDGAR has no concept of index membership and no answer to which companies existed and traded on a given past date.

**Which factor returns should a paper use?**
The Fama/French series from the Kenneth French Data Library, unless the contribution of the paper is a new factor construction. They are free, they are the series referees expect, and rebuilding them introduces differences that then have to be explained.

**What breaks a replication most often?**
Universe construction. Two researchers using the same method on the same period will disagree if one drew the sample from today's index membership and the other reconstructed it as of the sample dates, and nothing in either codebase will report an error.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
