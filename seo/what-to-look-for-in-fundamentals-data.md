# What to Look for in Fundamentals Data

Judge a fundamentals dataset on five things: whether line items are mapped to one schema across filers, whether units and share counts are stated rather than guessed at, whether banks and insurers and property trusts keep the statement items that matter to them, whether revenue can be broken down below the top line, and whether the dataset knows which company a ticker pointed to at the time. Price data can be checked by eye. A balance sheet cannot, so most of the errors in a fundamentals-driven model arrive silently and stay.

Every claim about another provider below was checked against that provider's own documentation in August 2026, and dated accordingly.

## What does "standardised" actually mean?

Public companies file in XBRL, which sounds like it settles the problem. It does not. The SEC's own API documentation is explicit that comparability applies to facts using a "non-custom taxonomy (e.g. us-gaap, ifrs-full, dei, or srt)" that apply "to the entire filing entity" (sec.gov EDGAR APIs, as of August 2026). Anything a company tags with a custom extension falls outside that guarantee, and large filers use extensions constantly.

The practical consequence is that two companies can both report cost of sales while tagging it differently, and a naive join produces a gross margin series with holes in it. A standardised dataset makes the mapping decision once and applies it everywhere. The useful follow-up question for any vendor is whether they will tell you what the mapping was. xfinlink documents its derivations in the open: gross profit, for instance, is computed uniformly as revenue minus `cost_of_revenue` where present and `cost_of_goods_sold` otherwise, because pre-2015 filings reported the concept under the second name. You may disagree with that choice, but you can see it and correct for it.

Ask to see the field reference before you buy. If nobody can tell you how a derived field was derived, you are buying somebody's undocumented opinion.

## Are the units written down anywhere?

This is the least glamorous item on the list and the one that costs the most. Revenue in millions against a market capitalisation in units, or share counts in thousands against earnings per share in dollars, produces ratios that are wrong by three orders of magnitude while looking entirely plausible on a chart.

xfinlink states it in the reference: monetary fields in millions of US dollars, per-share fields in dollars per share, share counts in millions. That single paragraph removes an entire category of bug. Whatever provider you choose, find the equivalent paragraph and read it before writing the first ratio, and see [how shares outstanding are reported](/blog/how-are-shares-outstanding-reported) for why the share-count line in particular repays attention.

## What happens when the company is a bank or a REIT?

Generic schemas assume every company has inventory, cost of goods and gross profit. Banks have none of those in any meaningful sense; they have net interest income, deposits and credit losses. Insurers earn premiums. Property trusts are judged on funds from operations, a measure that does not appear on any standard income statement.

A dataset that flattens all of them into one schema either drops those lines or forces them into fields where they do not belong. Industry-specific fields sit alongside the standard ones here, populated only where they apply:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

f = xfl.fundamentals(["JPM", "PGR", "SPG", "AAPL"], period_type="annual",
                     start="2024-01-01", end="2024-12-31",
                     fields=["revenue", "bank_net_interest_income", "bank_deposits",
                             "ins_premiums_earned", "reit_ffo", "inventory"])
print(f[["ticker", "revenue", "bank_net_interest_income", "bank_deposits",
         "ins_premiums_earned", "reit_ffo", "inventory"]].to_string(index=False))
```

```
ticker    revenue  bank_net_interest_income  bank_deposits  ins_premiums_earned  reit_ffo  inventory
  AAPL 391035.000                       NaN            NaN                  NaN       NaN     7286.0
   JPM 177556.000                   92583.0      2406032.0                  NaN       NaN    12988.0
   PGR  75372.000                       NaN            NaN              70799.0       NaN        NaN
   SPG   5963.798                       NaN            NaN                  NaN  3994.361        NaN
```

JPMorgan's $2.41 trillion of deposits and Simon Property's $3.99 billion of funds from operations are the numbers an analyst covering those companies would actually reach for. Apple has inventory and none of the rest, which is correct rather than missing. If a screen for financial-sector value is built on a schema without these fields, it is ranking banks on a ratio nobody in banking uses.

## Can you see inside the revenue line?

Consolidated revenue answers almost nothing on its own. Whether a company's growth came from one geography or one product is usually the question, and that lives in the segment note rather than on the face of the statements.

Segment data is available here through an `include=segments` flag, with a distinction that matters more than it sounds: the default returns only the segments that sum to consolidated revenue, and an "all" setting adds subtotals and overlapping members. Datasets that do not draw that line hand you a segment table where the parts add to more than the whole, and nothing marks which rows caused it.

## Does the dataset know which company you are asking about?

Tickers are reassigned. A backtest that pulls fundamentals for a symbol over twenty years may be splicing two unrelated companies together and will never say so.

The check to run is whether a provider can address a company by a stable identifier rather than only by its current symbol, and whether it can tell you every company that has used a given ticker with the dates each was valid. xfinlink exposes both: an `entity_id` that names the company directly and reaches past holders of a recycled ticker, and a resolution endpoint returning every historical holder with validity dates. Index membership carries the same discipline, with historical rosters returned as of a chosen date, which is what stops [survivorship bias](/blog/what-is-survivorship-bias-in-backtesting) from quietly inflating a backtest.

## How far back does the history go?

Depth requirements depend on the job. A factor study spanning several business cycles needs decades; a dashboard of current ratios needs the last few years. Paid xfinlink plans carry fundamentals back to 1950, daily prices to 1996 and institutional holdings to 1978, with the free tier serving a rolling one-year window.

The related question is how many companies you can ask about in one call, because that governs whether a cross-sectional screen is one request or five hundred. Per-request ticker limits here run from 1 on the free tier to 100 on Pro and 500 on Max.

## How the common sources compare

| | Line items standardised across filers | Industry-specific statements | Segment breakdowns | Historical entity resolution | Free access |
|---|---|---|---|---|---|
| SEC EDGAR XBRL API | Within non-custom taxonomies only; extensions excluded | Raw tags, mapping is yours | In the filing, not assembled | CIK is stable; no ticker history assembled | Free, no authentication (as of August 2026) |
| Alpha Vantage | Yes, own schema | Not published as separate fields | Not published | Not published | 25 requests per day (as of August 2026) |
| yfinance | Yes, own schema | Not published as separate fields | Not published | Not published | Free library, no key |
| xfinlink | Yes, derivations documented | Bank, insurance and REIT fields | `include=segments`, primary or all | `entity_id` plus dated ticker history | Free tier, 1-year rolling window |

SEC EDGAR deserves its place at the top of that table. It is the source of record, it costs nothing, it requires no key, and for a single company studied carefully it is the right answer. What it does not do is hand you a comparable panel across a few hundred filers; you build that yourself, and the mapping work is the expensive part. Alpha Vantage's fundamentals endpoints are a reasonable fit for a handful of tickers, though 25 requests a day (alphavantage.co/premium, as of August 2026) will not carry a cross-sectional screen, and their premium tiers start at $49.99 a month for 75 requests per minute. yfinance exposes `income_stmt`, `quarterly_income_stmt` and `ttm_income_stmt` on its Ticker object (ranaroussi.github.io/yfinance, as of August 2026), which is enough for a weekend script.

The choice turns on how many companies you need at once and whether you need to trust the join across them. For one company, take EDGAR. For a panel, take the dataset that has already made the mapping decisions and will show you what they were.

## Frequently asked questions

**Is XBRL data from EDGAR already standardised?**
Partly. The SEC states that facts using non-custom taxonomies are comparable across companies and over time, but that guarantee excludes custom extension tags, which large filers use routinely. The mapping work that remains is the reason vendors exist.

**Do I need annual or quarterly statements?**
Annual for anything spanning cycles, quarterly when timing matters within a year. The trade-offs are set out in [annual vs quarterly financial data](/blog/annual-vs-quarterly-financial-data).

**How do I check a provider before committing?**
Pull one company you know well and reconcile three or four line items against its latest 10-K, then pull a bank and a property trust and see whether the fields that matter to those businesses survived. Details are in the [API reference](/docs), and the [plans](/pricing) set out per-request limits.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
