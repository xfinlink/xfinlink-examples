**How Far Back Does SEC EDGAR Data Go?**

August 12, 2026 · GUIDES

EDGAR's electronic archive opens in 1993, and that single date hides three different answers. Filings phased in over four years, so the first calendar year with a broad set of annual reports is 1997, not 1993. Keyword search across the text of filings reaches back only to 2001. The structured numeric data behind the data.sec.gov APIs starts in 2009, because 2009 is when the SEC first required XBRL tagging. A fundamentals panel assembled purely from SEC electronic sources therefore begins in 2009 in practice, whatever start date the code asks for. Anything older has to come from a source that did the work of digitising it.

## What exactly starts in 1993?

The Commission adopted interim rules requiring electronic filing on 23 February 1993, and the phase-in began on 26 April 1993. Then it stopped. After a statutorily mandated test group finished in December 1993, the SEC held back further phase-in while staff evaluated system performance over a six-month test period running from January to June 1994. Final rules covering all domestic registrants arrived on 19 December 1994, and phase-in restarted on 30 January 1995 (SEC, [Overview of the EDGAR rules](https://www.sec.gov/info/edgar/regoverview.htm), read 12 August 2026).

That schedule shows up plainly in the SEC's own counts. Its published tally of electronic filings by form type, updated to June 2026, records these documents filed under form type 10-K by calendar year:

| Calendar year | 10-K filings on EDGAR |
| --- | --- |
| 1993 | 4 |
| 1994 | 1,844 |
| 1995 | 2,178 |
| 1996 | 4,251 |
| 1997 | 6,540 |

Four in 1993. Somebody who reads "EDGAR starts in 1993", sets a start date to match and then builds a cross-section will get a sample that is empty in its first year, thin for the two after it, and representative only from 1997 onward. The filings made before the phase-in went in on paper, and the SEC never went back and loaded them.

## Why does EDGAR full-text search only reach 2001?

Full-text search is a separate index layered over the archive rather than part of it, and it covers filings submitted electronically since 2001; the widest option the interface offers is labelled "All (since 2001)" ([EDGAR Full Text Search FAQ](https://www.sec.gov/edgar/search/efts-faq.html), read 12 August 2026). Filings from 1993 through 2000 are still in EDGAR and still retrievable by company and form type. They are not searchable by phrase.

The distinction bites on a specific kind of question. "What did this company file in 1998" works. "Which companies first disclosed a going-concern doubt in 1998" does not, because the only way to answer it is to read the text, and the index that would let you read all of it at once starts three years later. For pulling up an actual filing, EDGAR is the source of record and no vendor replaces it. For assembling numbers across many companies and many decades, it is the wrong shape of tool.

## Why do the structured SEC APIs start in 2009?

Because they are made of XBRL, and XBRL "was first required by the SEC in 2009" (SEC, [EDGAR Application Programming Interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), read 12 August 2026). The company-concept, company-facts and frames endpoints return tagged facts, so their coverage cannot start before the tagging obligation did. The Financial Statement Data Sets, which package the face financials into quarterly archives, begin in the same year and are described by the SEC as "extracted from corporate financial reports filed with the Commission using eXtensible Business Reporting Language (XBRL)".

One more limit is worth knowing before building anything on the submissions endpoint: it returns "at least one year's of filing or to 1,000 (whichever is more) of the most recent filings". It answers what a company filed lately, not what it filed in 1996.

## What starts when

Verified against each source's own pages on 12 August 2026.

| Source | Earliest data | What it returns |
| --- | --- | --- |
| EDGAR filing archive | 1993, thin until 1997 | Filed documents, one company at a time |
| EDGAR full-text search | 2001 | Phrase search across filing text |
| data.sec.gov XBRL APIs | 2009 | Tagged facts per company or per period |
| SEC Financial Statement Data Sets | 2009 | Quarterly archives of face-financial numbers |
| Alpha Vantage daily time series | "20+ years of historical data" | Daily OHLCV per symbol |
| xfinlink fundamentals | 1950 | Annual and quarterly statements as a DataFrame |
| xfinlink daily prices | 1996 | Daily bars, split-adjusted close, total return |

Alpha Vantage's depth statement is quoted from its [API documentation](https://www.alphavantage.co/documentation/); the xfinlink rows come from the [pricing page](https://xfinlink.com/pricing), where full history is included on every paid plan and the free tier serves a rolling one-year window.

## What breaks when a panel starts in 2009?

2009 is an unhelpful place for a financial history to begin, and not only because it is recent. It begins at the bottom, after the credit crisis had already run. A leverage screen fitted on 2009 onward has seen balance sheets recovering from a contraction and never one walking into it. The inflation of the late 1970s sits outside the window entirely, as does the 2000 unwind and the recession that followed it.

Sample length also decides what a test is allowed to claim. Our [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting) note that a ten-year window contains no full credit cycle, and the same reasoning applies harder to fundamentals, where each company contributes one observation per year rather than 250. Fifteen years of annual statements across 500 names is 7,500 company-years, which sounds like plenty until the question involves distress, and distress is concentrated in the years the window excludes.

A second problem arrives with the universe rather than the statements. Pairing a 2009-onward panel with a current index list reintroduces [survivorship bias](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting) on top of the truncation, and the two errors point the same way.

## How do you get statements from before EDGAR?

From a provider that digitised the pre-electronic record and normalised it into one schema. xfinlink serves annual and quarterly statements back to 1950, which is 43 fiscal years before EDGAR accepted its first 10-K.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["GE", "IBM", "KO", "PG", "XOM"]
df = xfl.fundamentals(tickers, period_type="annual",
                      start="1900-01-01", end="2026-12-31",
                      fields=["revenue"])

for t in tickers:
    s = df[df["ticker"] == t]
    print(f"{t}  {s['fiscal_year'].min()}-{s['fiscal_year'].max()}  "
          f"rows={len(s)}  before 2009={(s['fiscal_year'] < 2009).sum()}")

print(df[df["fiscal_year"] == 1950][["ticker", "period_end", "revenue"]]
      .sort_values("ticker").to_string(index=False))
```

Output:

```
GE  1950-2025  rows=76  before 2009=59
IBM  1950-2025  rows=76  before 2009=59
KO  1950-2025  rows=76  before 2009=59
PG  1950-2026  rows=77  before 2009=59
XOM  1950-2025  rows=76  before 2009=59

ticker period_end  revenue
    GE 1950-12-31   2232.9
   IBM 1950-12-31    214.9
    KO 1950-12-31    215.2
    PG 1950-06-30    632.9
   XOM 1950-12-31   3134.6
```

Revenue is in millions of dollars. Of the 76 annual statements each company contributes, 59 predate the XBRL requirement, which is where a panel built from data.sec.gov would have had to start. Procter & Gamble carries 77 because its fiscal year ends in June, so fiscal 2026 has already closed and filed.

The shape of the result matters as much as the depth. One call returns five companies in one DataFrame with fiscal-year labels and period-end dates already aligned, which is the part that takes the longest when the same panel is assembled from raw filings. The trade-offs between the two approaches are set out in our guide to the [SEC EDGAR API versus a fundamentals API](https://xfinlink.com/blog/sec-edgar-api-vs-fundamentals-api). Field names and parameters are in the [docs](https://xfinlink.com/docs).

## FAQ

**Can I get pre-1993 filings from EDGAR at all?**
No. Those filings were submitted on paper and were not loaded into the electronic archive afterwards. Their numbers survive only where somebody transcribed them.

**Does EDGAR full-text search cover the 1990s?**
No. The index covers filings submitted electronically since 2001. Filings from 1993 to 2000 remain retrievable by company and form type, one document at a time.

**Is a history starting in 2009 enough for a backtest?**
For anything about growth, margins or capital intensity, often yes. For anything about distress, leverage or how a strategy behaves in a credit contraction, no, because the window begins after the last one ended.

**How far back do daily prices go?**
xfinlink serves daily bars from 1996 on paid plans, and a rolling one-year window on the free tier.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
