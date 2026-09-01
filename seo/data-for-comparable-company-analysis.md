# What Data You Need for Comparable Company Analysis

A comparable company analysis needs four things from a data source: a defensible way to draw the peer set, financial statements on a consistent basis, market values dated to the same day, and identifiers that survive a ticker change. Prices and earnings on their own are not enough. The arithmetic of a multiple is trivial; the work is making sure fifteen companies are measured the same way, on the same calendar, before any median is taken.

## What goes into a comps table?

Three inputs, and only one of them is a price. The first is the peer set, which decides the answer more than any other choice you make. The second is a set of multiples. The third is the fiscal period each multiple is anchored to.

Multiples come in two families. Equity multiples divide the market value of the shares by something that belongs to shareholders: price to earnings, price to book. Enterprise multiples divide the value of the whole business by something that belongs to every capital provider. Enterprise value is market capitalisation plus debt minus cash, so EV/EBITDA compares two companies without letting a difference in borrowing distort the ranking. A utility financed with 60 percent debt and a software company financed with none will look very different on P/E for reasons that have nothing to do with the operating business. On EV/EBITDA they become comparable.

That is the case for enterprise multiples in most sectors. Banks and insurers are the exception, since debt is raw material rather than financing for them; price to book and price to earnings are the working multiples there.

## How do you pick the peer set?

Two routes, and serious work uses both. Classification codes come from the filings themselves: the SIC code a company registers with the SEC, its NAICS code, and the GICS sector, industry and sub-industry assigned to it. Index membership is the second route, and it does a job the codes cannot, which is to bound the set by size and liquidity.

Each route fails on its own. GICS sub-industries are narrow by design; Caterpillar shares its sub-industry with exactly one other S&P 500 member, which is not a comps table. SIC codes go the other way. A search of SIC 4931 or 3531 across a full entity history returns every company that ever registered under the code, including Clark Equipment, Barber Greene and Koehring, which have not traded in decades.

Full historical coverage is what makes a survivorship-bias-free study possible, and it is the wrong default for a comps table valuing a company today. Intersect the classification with a current index membership list and the problem disappears in one line of code.

Size matters more than most templates admit. A $170bn utility and an $11bn utility face the same regulators and different capital markets, so a peer set that spans an order of magnitude in market value is really two peer sets averaged together.

## Which multiples earn their place?

Four cover most situations, and you rarely need all four at once.

EV/EBITDA is the default for capital-intensive businesses with real operating profit. EV/Revenue is what remains when a company has no profit to divide by, which is why it dominates in software and biotech. P/E is the multiple everyone quotes and the one most easily broken by a single non-recurring item; a company with a large asset sale or write-off in the year prints a P/E that means nothing. P/B is worth carrying for financials and for asset-heavy names where book value tracks replacement cost.

Whatever the set, compute every multiple in the table on the same names. A median EV/EBITDA drawn from twelve companies and a median P/E drawn from fifteen are not describing the same peer group.

## What breaks a comps table

Fiscal year ends are the first thing to check. Electric utilities almost all close in December, which makes them an easy sector to compare. Retailers close in January or February, Apple closes in late September, and a table that stacks each company's "latest annual" figures next to the others is comparing different economies. Either align on a common calendar quarter or state plainly which period each column represents.

Split adjustment is the second. Per-share figures in a filing are as-reported and are not restated when the company later splits its stock, while a historical price series usually is adjusted. Divide an adjusted price by an as-reported EPS and the P/E is wrong by exactly the split factor, silently, for every company that has ever split. Reading a pre-computed multiple avoids the whole class of error.

Identity is the third and the least visible. Facebook became META, Dell went private and came back, and tickers get reassigned to unrelated companies after a delisting. A peer set keyed on ticker strings loses the history of any company that changed one. Keying on a stable entity identifier instead means the series survives the rename. xfinlink resolves tickers to persistent entity ids for exactly this reason, and the same id addresses prices, statements and index membership.

## Building the table in Python

The code below resolves a target company to its classification, pulls every entity in that sector, narrows to the sub-industry and to current S&P 500 membership, then reads the multiples for the latest annual period.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TARGET = "DUK"
MULTIPLES = ["ev_ebitda", "ev_revenue", "pe_ratio", "pb_ratio"]

info = xfl.resolve(TARGET)["data"][TARGET]["entities"][0]["classifications"]

pages, off = [], 0
while True:
    page = xfl.search(gics_sector=info["gics_sector"], limit=500, offset=off)
    if page.empty:
        break
    pages.append(page)
    off += len(page)
    if len(page) < 500:
        break
sector = pd.concat(pages, ignore_index=True)

live = xfl.index("sp500")
peers = sector[(sector["gics_sub_industry"] == info["gics_sub_industry"])
               & (sector["entity_id"].isin(live["entity_id"]))]

m = xfl.metrics(entity_id=[int(i) for i in peers["entity_id"]],
                period_type="annual", period="2y",
                fields=["market_cap"] + MULTIPLES)
m = m.sort_values("period_end").groupby("entity_id").tail(1)

comps = m.dropna(subset=MULTIPLES).sort_values("ev_ebitda")
print(comps[["ticker", "period_end", "market_cap"] + MULTIPLES].to_string(index=False))
print(comps[MULTIPLES].median())
```

Output:

```
target: DUK | Utilities | Electric Utilities | SIC 4931
sector entities searched: 720   peers in the S&P 500: 17
peers reporting all 4 multiples: 15
fiscal period ends in the table: 2025-12-31

ticker                    entity_name period_end  market_cap  ev_ebitda  ev_revenue  pe_ratio  pb_ratio
   EIX           EDISON INTERNATIONAL 2025-12-31    20770.81       5.72        3.06      4.67      1.18
    ES              EVERSOURCE ENERGY 2025-12-31    26331.74       9.84        3.92     15.38      1.63
   NRG               N R G ENERGY INC 2025-12-31    20964.27      10.05        1.08     27.46     32.25
   EXC                    EXELON CORP 2025-12-31    44725.56      10.48        3.80     16.01      1.55
   PNW     PINNACLE WEST CAPITAL CORP 2025-12-31    11805.20      10.68        4.08     19.33      1.68
   DUK               DUKE ENERGY CORP 2025-12-31    93289.98      10.76        5.54     19.00      1.80
    SO                    SOUTHERN CO 2025-12-31    98506.43      12.26        5.52     22.45      2.74
  EVRG                     EVERGY INC 2025-12-31    18657.41      12.27        5.80     22.14      1.83
   AEP AMERICAN ELECTRIC POWER CO INC 2025-12-31    66217.67      12.92        5.14     18.38      2.13
    FE               FIRSTENERGY CORP 2025-12-31    26500.25      13.51        3.46     26.06      2.12
   ETR                   ENTERGY CORP 2025-12-31    48157.35      13.58        5.78     27.19      2.85
   XEL             X C E L ENERGY INC 2025-12-31    47200.34      14.47        5.47     22.13      2.00
   LNT            ALLIANT ENERGY CORP 2025-12-31    17480.19      14.95        6.41     21.65      2.38
   CEG      CONSTELLATION ENERGY CORP 2025-12-31    85728.24      16.00        3.56     37.13      5.91
   NEE             NEXTERA ENERGY INC 2025-12-31   171514.22      17.51        9.49     24.95      3.14

peer median vs DUK
  ev_ebitda   median   12.27   DUK   10.76   premium  -12.3%
  ev_revenue  median    5.14   DUK    5.54   premium   +7.8%
  pe_ratio    median   22.13   DUK   19.00   premium  -14.1%
  pb_ratio    median    2.12   DUK    1.80   premium  -15.1%
```

Every period end reads 2025-12-31, which is what an easy sector looks like. Run the same code on retail or on semiconductors and the period-end line will show several dates, at which point aligning them becomes your problem to solve rather than one to ignore.

Duke trades below the peer median on three of the four multiples and above it on EV/Revenue, which is the signature of a company earning a thinner margin on each dollar of revenue than its peers do. That is a starting question, not a conclusion.

## Where the data comes from

| Source | Peer selection | Statements | Ready multiples | Cost, as of September 2026 |
| --- | --- | --- | --- | --- |
| SEC EDGAR APIs | None | Yes, from XBRL | No | Free, no key required |
| yfinance | Sector and industry labels | Yes | Some | Free, personal use |
| Alpha Vantage | None | Yes | Some | Free tier, then $49.99 to $249.99 a month |
| xfinlink | SIC, NAICS, GICS, index membership | Yes | Yes | Free tier, then $29 a month |

The SEC's own APIs are free, need no key, and are the authoritative source for anything a company filed. They return XBRL facts and submission history and nothing else: no share prices, no market values, no peer grouping. A comps table built on EDGAR alone is a comps table you still have to compute market capitalisation for, from a price source EDGAR does not have.

yfinance is a fine way to look at one company on a Sunday afternoon. Its own README states that the project "is not affiliated, endorsed, or vetted by Yahoo, Inc." and that "the Yahoo! finance API is intended for personal use only" (checked 1 September 2026), which settles the question of whether it belongs in work you publish or sell.

Alpha Vantage sells a real API with published limits: 25 requests a day on the free tier, and premium plans from $49.99 a month at 75 requests a minute up to $249.99 a month at 1,200 requests a minute (alphavantage.co/premium, checked 1 September 2026). Rate limits shaped as requests per minute suit a screen that polls a few symbols at a time.

xfinlink was built around the two problems this article keeps returning to. Peer selection runs off SIC, NAICS, GICS and index membership in the same query, so the peer set is drawn rather than typed. The multiples arrive computed, so the split-adjustment trap never opens. Historical index membership is available through an `as_of` date, which is what a comps table dated to a past quarter requires. See the [docs](/docs) and [pricing](/pricing) for the full surface.

## Questions people ask

**How many peers should a comps table have?**
Somewhere between eight and twenty. Below eight the median moves whenever one company reports; above twenty the set has usually stopped being comparable in any real sense.

**Median or mean?**
Median. Valuation multiples have a long right tail and occasional negative values, and one company with depressed earnings can pull a mean far away from anything the peer group looks like.

**Does a historical comps table need point-in-time data?**
Yes. Drawing the 2015 peer set from today's index membership keeps only the companies that survived to 2026, which biases the peer median upward. Use a dated membership snapshot instead. [What survivorship bias does to a backtest](/blog/what-is-survivorship-bias-in-backtesting) covers the same failure in a different setting.

**Why does my P/E disagree with another source?**
Usually the earnings figure, not the price: trailing twelve months against last fiscal year, or diluted against basic, or restated against as-first-reported. [Why beta differs between sources](/blog/why-beta-differs-between-sources) walks through the same class of disagreement.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
