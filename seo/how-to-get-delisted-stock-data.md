# How to Get Historical Data for Delisted Stocks

A delisted company still has a complete price record; reaching it means asking by a permanent company identifier rather than by ticker. Sears Holdings stopped trading on 23 October 2018, and its 3,872 daily bars are still there under entity 83480. The symbol SHLD has since passed to two other issuers, which is why the symbol alone no longer finds the company.

That distinction matters more than it sounds. A ticker is a lease, not a name.

## Why do delisted companies disappear from most datasets?

Most retail data feeds are built to answer one question: what is trading right now. When a company goes bankrupt, gets acquired, or drops below an exchange's listing standards, it stops being an answer to that question, and the row is dropped rather than retired. Nothing announces the deletion.

The consequence for research is [survivorship bias](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting), and it runs in one direction only. Companies do not leave an index because things went well. Backtest a strategy on the current S&P 500 roster and every firm that failed during the period has been quietly excluded, so the measured return belongs to a portfolio nobody could have held.

## What happens to the ticker after a company leaves?

The exchange reissues it. Symbols are scarce, four letters or fewer, and a good one gets picked up again within a few years. Four different issuers have held SHLD since 1972:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

for e in xfl.resolve("SHLD")["data"]["SHLD"]["entities"]:
    print(f"{e['entity_id']:6d}  {e['name'][:28]:28s}  "
          f"{e['ticker_valid_from']} -> {e['ticker_valid_to'] or 'current'}")
```

```
 74727  ANGELES CORP                  1972-12-14 -> 1975-01-21
 83480  SEARS HOLDINGS CORP           2005-03-28 -> 2018-10-23
 67180  VICTORY PORTFOLIOS II         2020-11-05 -> 2021-10-11
 69627  GLOBAL X FUNDS                2023-09-13 -> current
```

Three of those four are unrelated businesses, and one is an exchange-traded fund. A backtest that reads SHLD as a single continuous series is stitching a department store to a fund across a five-year gap. The validity windows are the fix: each record states the dates over which that issuer actually held the symbol, so the right one can be selected by date rather than guessed. The same trap is worked through in detail for a live example in [what ticker recycling does to a backtest](https://xfinlink.com/blog/ticker-recycling-dangers-python).

## How do you find a company that no longer trades?

Search by name, then address the result by its identifier. The identifier does not move when the ticker does, which is the whole point of having one.

```python
hits = xfl.search(q="Sears Holdings")
eid = int(hits.loc[hits["entity_name"] == "SEARS HOLDINGS CORP", "entity_id"].iloc[0])

px = xfl.prices(entity_id=eid, start="2003-01-01", end="2019-12-31", fields=["close"])
px = px.sort_values("date")
print(f"entity {eid}: {len(px)} daily bars, "
      f"{px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}")
print(f"last close:   {px['close'].iloc[-1]:.4f}")
```

```
entity 83480: 3872 daily bars, 2003-06-10 to 2018-10-23
last close:   0.3659
```

`entity_id` is accepted anywhere a ticker is, across prices, fundamentals, and metrics. For a company caught up in a merger or a bankruptcy reorganisation, `resolve()` also carries predecessor and successor links, so the chain from one legal entity to the next can be walked without hand-maintained mapping tables. The [General Motors bankruptcy](https://xfinlink.com/blog/gm-bankruptcy-entity-resolution-python) is the standard worked case, since the pre-2009 company and the post-2010 company are genuinely different registrants that share a symbol.

## What does the end of a price record look like?

Three companies, each of which stopped trading for a different reason:

| Company | Entity | Bars | First | Last | Last close |
|---|---|---|---|---|---|
| Enron Corp | 3962 | 1,519 | 1996-01-02 | 2002-01-11 | 0.6250 |
| Bear Stearns Companies | 8372 | 3,125 | 1996-01-02 | 2008-05-30 | 9.3300 |
| Sears Holdings Corp | 83480 | 3,872 | 2003-06-10 | 2018-10-23 | 0.3659 |

Each final bar lands where the corporate record says it should. Enron last traded on the NYSE on Friday 11 January 2002, and the exchange suspended trading the following Tuesday. Bear Stearns closed at 9.33 on 30 May 2008, the day JPMorgan Chase completed the acquisition at an implied 10 dollars a share. Sears filed for Chapter 11 on 15 October 2018 and moved to the over-the-counter market on 24 October, one session after the last bar above.

Those endings are the useful part. A strategy that held Bear Stearns into 2008 needs the 9.33, not a gap.

## Which sources carry delisted history?

| Source | Delisted price history | Addressed by |
|---|---|---|
| yfinance | returns an empty DataFrame for a symbol Yahoo no longer serves | ticker |
| SEC EDGAR | filings stay available permanently; no price data at all | CIK |
| xfinlink | daily bars retained on the company record, back to 1996 | entity id or ticker |

yfinance prints `No data found, symbol may be delisted` and hands back an empty frame, a behaviour its own issue tracker has carried since June 2023 (checked 14 August 2026). For a weekend script on liquid large caps that is a reasonable trade for a free library. It becomes a problem when the empty frame is indistinguishable from a genuine absence of trading, because the backtest silently continues without the name.

EDGAR is worth knowing about for a different reason: the filings of a delisted company remain public indefinitely, so the fundamental record survives even where the price record has to come from somewhere else.

Full history on xfinlink sits on the paid plans, with daily prices back to 1996; a free key returns a rolling one-year window, which is enough to test the calls but not to reconstruct a company that stopped trading in 2008. Details are on [pricing](https://xfinlink.com/pricing) and the endpoint reference is in the [docs](https://xfinlink.com/docs).

## FAQ

**Does a delisted company keep its identifier?**
Yes. The identifier belongs to the company, not to the symbol, so it keeps working after the ticker is reassigned to somebody else.

**How do I know which issuer held a ticker on a given date?**
`resolve()` returns every holder with a `ticker_valid_from` and `ticker_valid_to`, so the holder on any date can be selected directly rather than inferred.

**Is a company that moved to the over-the-counter market delisted?**
Not in the sense that matters here. It left its original exchange, and the exchange listing ends, but the company continues to exist and to file. Sears moved to the over-the-counter market as SHLDQ on 24 October 2018, the session after the last bar shown above.

**Can I get fundamentals for a company that no longer trades?**
Yes, through the same `entity_id`. Statements were filed while the company was reporting and stay attached to the company record afterwards.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
