# SEC EDGAR API vs Fundamentals API: Which to Use

The SEC EDGAR APIs are the record itself: free, unauthenticated, and returning exactly what each company tagged in its own filing, one company per request. A fundamentals API returns the same filings after normalisation, with one revenue column, one schema, and many companies per call. Use EDGAR when the question is about a particular filing and the answer has to trace back to it. Use a fundamentals API when the question spans companies or years, because in that second case nearly all the effort sits between as-filed XBRL and a table you can compare across.

## What do the SEC EDGAR APIs actually return?

Four JSON endpoints on data.sec.gov, as of July 2026. `submissions` returns a filer's filing history; `companyconcept` returns one XBRL tag for one company; `companyfacts` returns every XBRL tag for one company; `frames` returns one tag across filers for a single reporting period. The SEC states that these "do not require any authentication or API keys to access", and describes the coverage as "the submissions history by filer and the XBRL data from financial statements (forms 10-Q, 10-K, 8-K, 20-F, 40-F, 6-K, and their variants)".

Freshness is the strongest thing about them, and no reseller will beat it. The SEC reports a typical processing delay of "less than a second" on submissions and "under a minute" on the XBRL endpoints.

Two constraints travel with that. Access is capped at 10 requests per second, and every request must carry a User-Agent header naming a company and a contact address, or the response is an "Undeclared Automated Tool" error rather than data. XBRL also has a floor. Under the phase-in, large accelerated filers using US GAAP with more than $5 billion of non-affiliate float went first, for periods ending on or after 15 June 2009; all remaining domestic and foreign filers followed for periods ending on or after 15 June 2011. Ask `companyfacts` for a 2004 income statement and there is nothing on the other side to answer with.

The bulk route has the same floor. The SEC's Financial Statement Data Sets publish the numeric face-financial data as quarterly zip files starting at 2009 Q1, updated quarterly, "presented without change from the 'as filed' financial reports submitted by each registrant". The SEC attaches its own caution: "we cannot guarantee the accuracy of the data sets."

## Why does one company's revenue sit under three different tags?

Because the accounting standard changed and the tag changed with it, and the API hands back every era exactly as it was filed.

Apple's `companyfacts` document, CIK 0000320193, downloaded on 31 July 2026, is 3.7 MB of JSON carrying 505 distinct tags: 503 under `us-gaap` and 2 under `dei`. Revenue is spread across three of them. `SalesRevenueNet` covers fiscal 2009 through 2018, `Revenues` appears for fiscal 2018, and `RevenueFromContractWithCustomerExcludingAssessedTax` runs from fiscal 2019 onward. One revenue series for one company therefore requires knowing that all three exist and deciding which one wins where they overlap.

That is the easy version of the problem. The harder version is that filers may invent elements: SEC guidance states that where no standard element matches a company's line item, "a company will create a company-specific element, called an 'extension'." Repeat across several thousand filers and the mapping from tags to comparable fields becomes the actual engineering work.

Here is the same information after that work has been done once, centrally:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.fundamentals(["AAPL", "MSFT", "WMT"], period_type="annual",
                      start="2017-01-01", end="2020-12-31",
                      fields=["revenue", "net_income"])
print(df[["ticker", "fiscal_year", "period_end", "revenue", "net_income"]].to_string(index=False))
```

```
ticker  fiscal_year period_end  revenue  net_income
  AAPL         2017 2017-09-30   229234       48351
  AAPL         2018 2018-09-29   265595       59531
  AAPL         2019 2019-09-28   260174       55256
  AAPL         2020 2020-09-26   274515       57411
  MSFT         2017 2017-06-30    96571       25489
  MSFT         2018 2018-06-30   110360       16571
  MSFT         2019 2019-06-30   125843       39240
  MSFT         2020 2020-06-30   143015       44281
   WMT         2017 2017-01-31   481317       13643
   WMT         2018 2018-01-31   495761        9862
   WMT         2019 2019-01-31   510329        6670
   WMT         2020 2020-01-31   519926       14881
```

Three companies, three fiscal calendars, one column. Apple closes its year in September, Microsoft in June, Walmart in January, and the `period_end` column carries the real date while `fiscal_year` carries the label the company used. The window above straddles Apple's revenue tag change and nothing in the caller's code has to know that. Values are in millions of dollars, and the same call reaches statements back to 1950 on a paid key.

## How do you get from a ticker to a CIK?

EDGAR is keyed on CIK, never on ticker, so any pipeline built on it starts with a mapping step. The SEC publishes one, `company_tickers.json`, with an explicit caveat on its own developer page: "We periodically update the file but do not guarantee accuracy or scope."

Read on 31 July 2026, that file holds 10,432 entries and maps GM to CIK 1467858, General Motors Co. The pre-bankruptcy General Motors is still in EDGAR under CIK 40730, where the submissions API returns the name Motors Liquidation Co, a former name of GENERAL MOTORS CORP, and an empty `tickers` array. Dell behaves the same way: DELL maps to 1571996, while Dell Inc's CIK 826083 carries no ticker at all.

The mapping file is a snapshot of who holds each ticker now, which is the correct answer to a different question than the one a historical study is asking. A 2007 study that resolves tickers through it gets 2026's companies.

```python
info = xfl.resolve("GM")
for e in info["data"]["GM"]["entities"]:
    print(e["name"], "| cik", e["cik"], "|", e["ticker_valid_from"], "->", e["ticker_valid_to"])
```

```
General Motors Corporation (pre-2009 bankruptcy) | cik 0000040730 | 1962-07-02 -> 2009-06-01
General Motors Company | cik 0001467858 | 2010-11-18 -> None
```

Both companies come back, each with the dates its claim on the ticker was valid and its own CIK. The CIK matters beyond identification: it is the key that takes you straight back to the filing on EDGAR when a number needs auditing. More on what recycled tickers do to a study is in the note on [ticker recycling](https://xfinlink.com/blog/ticker-recycling-dangers-python).

## Which one should you build on?

| | SEC EDGAR APIs | xfinlink |
|---|---|---|
| Cost and key | Free, no authentication or key | Free tier, key required; $29/month Pro |
| Statement history | XBRL from 2009 at the earliest, 2011 for smaller filers | Statements to 1950, daily prices to 1996 on paid plans |
| Response shape | Raw XBRL tags, as filed, nested JSON | Named columns, one schema, pandas DataFrame |
| Companies per request | One (except the `frames` endpoint) | 1 free, 100 Pro, 500 Max |
| Ticker handling | Current holder only, via `company_tickers.json` | All historical holders with validity dates and CIKs |
| Rate limit | 10 requests/second, User-Agent required | 100/day free, 10,000/day Pro |
| Best at | Auditing a specific filing, near-live disclosure | Panels across companies and decades |

Sources: the [SEC's EDGAR API page](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), the SEC's [accessing EDGAR data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) and [developer FAQ](https://www.sec.gov/os/webmaster-faq) pages, the [interactive data compliance guide](https://www.sec.gov/info/smallbus/secg/interactivedata-secg.htm), the [Financial Statement Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) page, and the xfinlink [pricing page](https://xfinlink.com/pricing) and [docs](https://xfinlink.com/docs). All read on 31 July 2026.

Nothing above argues against going to EDGAR. It is the record, it is free, and a number that ends up in front of a client should be checkable against the document a company actually signed. What EDGAR does not do is hand back a comparable panel, and if the plan involves several hundred companies over several decades, the tag mapping, the fiscal-calendar alignment and the entity history are the project rather than a preliminary to it. xfinlink is built from SEC EDGAR public filings and market data and does that work once, returning the CIK alongside the numbers so the trip back to the filing stays open. The columns a study needs before it can start are set out in the guide on [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting).

## FAQ

**Is the SEC EDGAR API really free?**
Yes. The SEC states that the data.sec.gov APIs "do not require any authentication or API keys to access". The cost is engineering: raw XBRL, one company per request, and a 10 requests-per-second ceiling.

**How far back does EDGAR XBRL data go?**
Fiscal periods ending on or after 15 June 2009 for the largest US GAAP filers, and 15 June 2011 for everyone else. Filings before those dates exist as documents but carry no tagged financial data.

**Do I still need EDGAR if I use a fundamentals API?**
For anything that has to be audited against the source document, yes. `xfl.resolve()` returns each entity's CIK, which is the identifier EDGAR needs, so the two fit together rather than competing.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
