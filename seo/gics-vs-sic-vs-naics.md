# GICS vs SIC vs NAICS: Which Industry Classification to Use

Three industry classification systems turn up in equity work, and each was built to answer a different question. GICS, maintained by MSCI and S&P Dow Jones Indices, groups companies the way investors compare them, and its owners describe it as a scheme built to give investors consistent industry definitions. SIC is the code carried in a company's EDGAR filing header, so it is the right key when the work has to line up with the filing. NAICS is the federal statistical standard, which makes it the key that joins a company to government economic data. Screens, peer sets and factor neutralisation belong to GICS. Filing-level work belongs to SIC. Anything that has to meet Census or Bureau of Labor Statistics tables belongs to NAICS.

The trap sits one level below that. Many market data APIs return a field called `sector` that follows none of the three, and the strings look close enough to a standard that nobody checks.

## Who maintains each scheme?

GICS is jointly run by two index providers. MSCI states that "MSCI and S&P Dow Jones Indices developed this classification standard to provide investors with consistent and exhaustive industry definitions", and the GICS methodology book dated August 2024 describes four levels containing "11 Sectors, 25 Industry Groups, 74 Industries, and 163 Sub-Industries". A company's full classification is "an 8-digit code with text description", and the structure is reviewed annually. Both pages were read on 5 August 2026.

SIC is the oldest of the three and no longer has a federal owner. The Bureau of Labor Statistics puts its span plainly: "For over 60 years, the Standard Industrial Classification (SIC) system served as the structure for the collection, presentation, and analysis of the U.S. economy", and NAICS "was introduced in 1997". The Office of Management and Budget finished the job in a Federal Register notice dated 21 December 2021, which states that "Statistical Policy Directive No. 9, Standard Industrial Classification of Enterprises, will be eliminated effective immediately".

The SEC did not follow. Its own code list page, read on 5 August 2026, says that "The Standard Industrial Classification Codes that appear in a company's disseminated EDGAR filings indicate the company's type of business", and that "These codes are also used in the Division of Corporation Finance as a basis for assigning review responsibility for the company's filings". A classification retired from federal statistics in 2021 still decides which SEC staff read your 10-K.

NAICS is the live government standard. The same Federal Register notice records that Mexico's INEGI, Statistics Canada and OMB "jointly developed NAICS in 1997 and continue to collaborate", that "Revisions are considered every five years in calendar years ending with 2 and 7", and that federal establishment data for reference years from 1 January 2022 should be published on 2022 NAICS codes. The next revision lands on the 2027 cycle.

| Scheme | Maintained by | Structure | Built for |
|---|---|---|---|
| GICS | MSCI and S&P Dow Jones Indices | 11 sectors, 25 industry groups, 74 industries, 163 sub-industries; 8-digit code | Comparing companies as investments |
| SIC | No current federal owner; still assigned in EDGAR | 4-digit codes on the filing header | Filing-level identification and SEC review routing |
| NAICS | INEGI, Statistics Canada and OMB, revised in years ending 2 and 7 | Hierarchical codes up to 6 digits | Publishing and joining government economic statistics |

Sources: the [MSCI GICS page](https://www.msci.com/our-solutions/indexes/gics) and [GICS methodology book](https://www.msci.com/indexes/documents/methodology/1_MSCI_Global_Industry_Classification_Standard_GICS_Methodology_20240801.pdf), the [BLS NAICS page](https://www.bls.gov/bls/naics.htm), the [Federal Register notice of 21 December 2021](https://www.govinfo.gov/content/pkg/FR-2021-12-21/html/2021-27536.htm) on govinfo, and the [SEC's SIC code list](https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list). All read on 5 August 2026.

## Why does one company carry three different sector labels?

Take IBM. Under GICS the sector is Information Technology. The SEC's submissions API, queried on 5 August 2026, returns SIC 3570, described as "Computer & office Equipment". Alpha Vantage's `OVERVIEW` endpoint, called the same day against its own demo key, returns `Sector` as `TECHNOLOGY` and `Industry` as `INFORMATION TECHNOLOGY SERVICES`, and its 40-plus field response carries no SIC or NAICS code at all. The yfinance documentation, also read on 5 August 2026, exposes `Sector` and `Industry` classes whose keys look like `technology` and `software-infrastructure`, and it names no classification standard anywhere.

Three labels for one company, and they do not resolve to each other. `TECHNOLOGY` is not a GICS sector name; GICS calls that sector Information Technology, and the difference matters the moment code tries to match strings against an index provider's sector list. Nothing here is wrong on any vendor's part. A field called `sector` is only a promise about grouping, not about which scheme did the grouping, and the question worth asking of any data source is which of the three it follows, if any.

Amazon shows the same gap in the other direction. EDGAR returns SIC 5961, "Retail-Catalog & Mail-Order Houses", read on 5 August 2026. The code is accurate to the filing header and says nothing about the cloud business, because the SIC vocabulary was fixed in an era that ended before that business began. GICS places the company in Consumer Discretionary, which is a claim about how investors should compare it rather than about what its filing cover page says.

## When is each one the right key?

Reach for SIC when the unit of analysis is the filing. Anything that has to reconcile against EDGAR, or that follows how the SEC itself routes a registrant, should use the code EDGAR carries, straight from the filing header. The guide on [SEC EDGAR API vs a fundamentals API](https://xfinlink.com/blog/sec-edgar-api-vs-fundamentals-api) covers what that endpoint returns and what it costs in engineering.

Reach for NAICS when the equity data has to meet an economic series. Employment, output, price indices and establishment counts are published on NAICS, so a study linking company results to industry conditions needs the code the statistical agency used, on the vintage it used.

Everything cross-sectional in equities points at GICS: sector weights, peer sets, sector-neutral factor sorts, dispersion between industries. Comparing companies as investments is the job MSCI states the scheme was designed for, and it is the only one of the three subject to an annual review of its own structure. Two of these schemes describe what a company files; the third describes how a portfolio is compared.

## How do you get a sector-keyed universe in Python?

The awkward part of sector work is rarely the classification itself. It is the join: a mapping table that has to be sourced, refreshed, and kept aligned with ticker changes. Data that arrives already carrying its sector removes that step.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

names = ["AAPL", "MSFT", "IBM", "KO", "DIS", "GE", "JPM", "XOM"]
df = xfl.prices(names, period="1w", fields=["close"])

latest = df.sort_values("date").groupby("ticker", as_index=False).tail(1)
print(latest[["ticker", "entity_name", "gics_sector", "date", "close"]]
      .sort_values("gics_sector").to_string(index=False))
```

```
ticker                          entity_name            gics_sector       date  close
   DIS                       DISNEY WALT CO Communication Services 2026-08-04  98.18
    KO                         COCA COLA CO       Consumer Staples 2026-08-04  86.56
   XOM                     EXXON MOBIL CORP                 Energy 2026-08-04 153.96
   JPM                  JPMORGAN CHASE & CO             Financials 2026-08-04 357.52
    GE                  GENERAL ELECTRIC CO            Industrials 2026-08-04 377.28
  MSFT                       MICROSOFT CORP Information Technology 2026-08-04 492.81
   IBM INTERNATIONAL BUSINESS MACHINES CORP Information Technology 2026-08-04 235.15
  AAPL                            Apple Inc Information Technology 2026-08-04 309.38
```

The GICS sector rides on the price row itself, next to the permanent entity identifier, so grouping by sector is a `groupby` rather than a merge. `xfl.search(gics_sector="Information Technology")` runs the same idea in reverse and returns the entities in a sector, which is how a sector-restricted screen starts. The [stock screener guide](https://xfinlink.com/blog/what-api-to-use-for-a-stock-screener) covers the rest of that loop, and the [docs](https://xfinlink.com/docs) list the fields each endpoint returns.

## FAQ

**Who owns GICS?**
MSCI and S&P Dow Jones Indices, jointly. Both publish the methodology book free, and terms for any use of the structure itself come from those two owners rather than from a data vendor.

**Why does EDGAR still use SIC when the government retired it?**
Because the SEC uses it operationally rather than statistically. The SIC code on a filing header tells the Division of Corporation Finance which staff review the filing, a job the code does whether or not any statistical agency still publishes on it.

**Which classification should a backtest use?**
GICS, for any test whose portfolios are built or neutralised by sector, since it is the one of the three built around how investors compare companies and the one whose structure is reviewed every year. Use SIC or NAICS when the test has to reconcile against filings or government series.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
