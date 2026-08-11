# Alpha Vantage vs Massive vs xfinlink for Fundamentals

Three vendors, one job: turn SEC filings into a table of revenue and net income that compares cleanly across companies. All three do that job. The separation shows up elsewhere, in how far back the statements reach, how many companies a single request covers, what the licence permits, and what the monthly bill comes to. Checked against each vendor's own pages on 11 August 2026, a request for IBM returns 20 annual income statements from Alpha Vantage, records dating to 29 March 2009 from Massive, and 66 annual periods beginning in 1960 from xfinlink.

## How far back does each one go?

Alpha Vantage documents `INCOME_STATEMENT` as returning "the annual and quarterly income statements for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC", refreshed "generally... on the same day a company reports its latest earnings and financials". The demo call published in that documentation, run on 11 August 2026, returned 20 annual reports covering fiscal 2006 through fiscal 2025 and 81 quarterly reports from June 2006 to June 2026. Each annual record carried 26 keys, two of them the period end date and the reporting currency.

Massive is where `polygon.io` now points; the old domain answered with a 301 redirect to `massive.com` on 11 August 2026. It splits the statements into one endpoint each. The income statements page states that "Records date back to March 29, 2009" and accepts a `timeframe` of quarterly, annual or trailing twelve months. Its plan table marks the endpoint as not included on Basic, Starter or Developer, and available with all history on Stocks Advanced at $199 per month or on the Financials & Ratios Expansion at $29 per month.

xfinlink serves statements back to 1950 on paid plans, with the income statement, balance sheet and cash flow of a period arriving in the same row. IBM comes back with 66 annual periods:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.fundamentals("IBM", period_type="annual", start="1960-01-01", end="2025-12-31",
                      fields=["revenue", "net_income"])
print(len(df), "annual periods")
print(df[["fiscal_year", "period_end", "revenue", "net_income"]].head(3).to_string(index=False))
print(df[["fiscal_year", "period_end", "revenue", "net_income"]].tail(2).to_string(index=False))
```

```
66 annual periods
 fiscal_year period_end  revenue  net_income
        1960 1960-12-31 1436.053     168.181
        1961 1961-12-31 1694.296     207.228
        1962 1962-12-31 1925.221     241.387
 fiscal_year period_end  revenue  net_income
        2024 2024-12-31  62753.0      6023.0
        2025 2025-12-31  67535.0     10593.0
```

IBM's 1960 revenue of $1.44 billion sits in the same column, on the same scale, as its 2025 revenue of $67.5 billion. Values are in millions of dollars throughout.

## Why does 2009 keep appearing as a floor?

Because that is roughly where the structured public record starts. The SEC's Financial Statement Data Sets, which publish the numeric face-financial data extracted from filings through XBRL, cover January 2009 to March 2026 and update quarterly, and the SEC states plainly that it "cannot guarantee the accuracy of the data sets". A pipeline that begins with XBRL inherits that starting date. Alpha Vantage reaches past it for IBM, back to fiscal 2006, so the floor is not identical everywhere, but it is close enough to be the first thing to check.

Whether that matters depends entirely on the study window. A factor test that starts in 1995 and a recession comparison that needs 2001 and 2008 in the same table both stop at the wall. Older filings are still public, but they exist as documents rather than as columns, and converting them is work that has to happen once per company and once per line item. That gap between a filing and a usable table is the subject of a separate note on [the SEC EDGAR API versus a fundamentals API](https://xfinlink.com/blog/sec-edgar-api-vs-fundamentals-api).

## How many companies does one request cover?

Alpha Vantage takes one symbol per call, with `function` and `symbol` both required. Its premium page describes the standard usage limit as "25 API requests per day", while its support FAQ describes the free service as "25 API requests per minute and unlimited API requests for verified open-source or educational projects". The two pages disagree as of 11 August 2026, so confirm the number before sizing a job around it. Paid plans start at 75 requests per minute for $49.99 per month and reach 1,200 per minute for $249.99 per month.

Massive filters its statement endpoints by ticker or CIK and controls volume through rows rather than calls: the `limit` parameter "Defaults to '100' if not specified. The maximum allowed limit is '50000'". The Stocks Starter plan at $29 per month advertises unlimited API calls, though statements are not part of that tier.

xfinlink caps tickers per call by plan, at 1 on Free, 100 on Pro, 500 on Max and 5,000 on Redistribution. A small panel is a single call:

```python
panel = xfl.fundamentals(["IBM", "KO", "GE"], period_type="annual",
                         start="2023-01-01", end="2024-12-31",
                         fields=["revenue", "net_income"])
print(panel[["ticker", "fiscal_year", "revenue", "net_income"]].to_string(index=False))
```

```
ticker  fiscal_year  revenue  net_income
    GE         2023    35348        9482
    GE         2024    38702        6556
   IBM         2023    61860        7502
   IBM         2024    62753        6023
    KO         2023    45754       10714
    KO         2024    47061       10631
```

Three companies, one schema, each fiscal calendar preserved in its own `period_end`. Which period type belongs in which analysis is covered in the note on [annual versus quarterly data](https://xfinlink.com/blog/annual-vs-quarterly-financial-data).

## Side by side

| | Alpha Vantage | Massive | xfinlink |
|---|---|---|---|
| Statement history | 20 annual periods for IBM, fiscal 2006 to 2025, measured 11 Aug 2026 | "Records date back to March 29, 2009" | To 1950 on paid plans; 66 annual periods for IBM |
| Companies per request | One symbol | Filter by ticker or CIK, up to 50,000 rows | 1 Free / 100 Pro / 500 Max / 5,000 Redistribution |
| Entry price for statements | Free key, then $49.99/month | $29/month Financials & Ratios, or $199/month Stocks Advanced | Free tier on a 1-year window, then $29/month Pro |
| Licence | Terms published as a PDF | $29 and $199 plans marked "Individual use only"; business plans quoted separately | Max $79/month company-wide internal; Redistribution $249/month adds end-user display |
| Rest of the platform | Options, forex, crypto, news sentiment, earnings estimates, listing and delisting status | Real-time and delayed market data; the financials expansion sells without a stocks subscription | Prices, computed metrics, index membership, insider filings, 13F holdings |

## What does the licence allow?

Massive labels both individual fundamentals routes "Individual use only" on its pricing page, at $29 per month for Financials & Ratios and $199 per month for Stocks Advanced, and quotes business plans separately. Alpha Vantage publishes its terms as a PDF rather than a web page, which is worth reading in full before anything ships. xfinlink splits the two cases on the [pricing page](https://xfinlink.com/pricing): Max at $79 per month carries a company-wide internal licence, and Redistribution at $249 per month adds the right to display the data to end users.

yfinance deserves an honest mention here, because it costs nothing and is perfectly adequate for a weekend script. Its own documentation states that the project "is not affiliated, endorsed, or vetted by Yahoo, Inc." and that "the Yahoo! finance API is intended for personal use only", which answers the question for anything with paying customers attached to it.

## Which one fits which job?

Alpha Vantage covers the widest surface outside the statements themselves. One key also reaches options, forex, crypto, news sentiment, earnings estimates and a listing status endpoint that the documentation positions for survivorship research. For a single-company lookup, or a dashboard that wants several asset classes from one vendor, it is the shortest route.

Massive is the sensible pick when market data is the main purchase and fundamentals ride along, or when a served trailing-twelve-month timeframe saves the trouble of summing quarters. Buying the financials expansion without a stocks subscription is possible, at the same $29 as the entry stocks plan.

Everything else in fundamentals work points the other way: panels across hundreds of companies, windows that open before 2009, and statements that need to sit beside prices and index membership under one key. That is the shape xfinlink is built around, and the free tier reaches all of it on a one-year window before any decision about paying. The [docs](https://xfinlink.com/docs) list the full field set per statement.

## FAQ

**Can I get pre-2009 statements without paying anyone?**
Not as structured data from the SEC. The Financial Statement Data Sets begin in January 2009; earlier filings are public as documents, so a pre-2009 panel means either a parsing project or a vendor that has already finished one.

**Which is cheapest to start with?**
Alpha Vantage's free key and xfinlink's free tier both cost nothing, the latter giving 100 requests a day on a one-year history window. Massive's statement endpoints start at $29 per month.

**Does any of them serve trailing twelve months directly?**
Massive accepts `timeframe=trailing_twelve_months` on its statement endpoints. xfinlink returns a TTM snapshot through `xfl.metrics(ticker, period_type="ttm")`. Alpha Vantage returns annual and quarterly reports, leaving the summation to the caller.

**How current is each one after an earnings release?**
Alpha Vantage states that its statement data is "generally refreshed on the same day a company reports". Massive documents its statement endpoints as end-of-day, updated daily.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
