# What Is Survivorship Bias in Backtesting?

Survivorship bias is the error of testing a strategy on the companies that were still standing at the end of the test window. Draw a universe from today's S&P 500, or from a price file that only holds symbols still trading, and every bankruptcy, buyout and quiet delisting has already been swept out of the sample before the first line of code runs. The result is a backtest that reports returns nobody could have earned. The correction is mechanical rather than clever: take index membership as it stood on the formation date, keep the names that later disappeared, and hold each one to its final traded price.

## Where does the bias get in?

Usually at the universe, and usually without anyone noticing the moment it happens. The universe gets chosen from a current list, because a current list is what is easy to obtain: the Wikipedia page of S&P 500 components, a broker export, a screener result. Every name on that list passed a test the strategy knows nothing about, which is that the company survived and stayed large enough to remain in the index.

The price source is the other common entrance. Some feeds serve whatever the symbol currently maps to and nothing else, so a query for a company acquired in 2016 returns an empty frame, an unrelated fund, or an error. Whatever the strategy does with those failures, dropping them, filling them forward, or skipping the ticker entirely, the effect is the same: the losers leave and the average improves.

A third path is narrower but nastier. Ticker symbols get reassigned. If the backtest is keyed on the symbol rather than on the company, a 2015 position can silently become a different business by 2024, and the returns stitched across the handover belong to nobody.

## How much does the index actually turn over?

Enough that the choice of universe dominates most of the parameters people spend their time tuning. On the S&P 500 roster of 29 July 2016, 142 of the 495 names are absent from the roster today, a little under 29% in ten years. The membership event log records 189 removals over the same decade, a larger number because it also counts companies that joined after 2016 and left again before now.

The distortion those removals cause has been measured directly. Running one identical revenue-growth screen over two universes that differ in nothing except point-in-time membership, [the point-in-time version compounded at 8.74% a year against 18.79% for the contaminated version](https://xfinlink.com/blog/growth-screen-survivorship-bias-python) between June 2021 and July 2026. Ten percentage points a year of pure hindsight, from one choice about which list to start from, across five years.

## How do you build a point-in-time universe?

Ask the data source for membership on the date the portfolio forms, not membership today.

```python
import xfinlink as xfl
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

past = xfl.index("sp500", as_of="2016-07-29")   # membership as it stood then
now = xfl.index("sp500")                        # membership today

gone = set(past["entity_id"]) - set(now["entity_id"])
print(len(past), len(now), len(gone))
```

Output:

```
495 499 142
```

Those 142 entity identifiers are the sample. A backtest that never sees them is not a backtest of the strategy; it is a backtest of the strategy plus a decade of hindsight about which companies would still exist. The `as_of` parameter carries no extra cost on a paid plan, and a free key allows 50 historical snapshots per day.

## What happens to the companies that leave?

Leaving the index is often not failure at all: companies get acquired, taken private, or restructured out of eligibility. Each departure still needs a rule, and the workable one is to hold the position to the last available price and then redeploy the proceeds. Prices that run past the delisting date are the part many free sources cannot supply.

| Company | Ticker | Last traded | Last close |
|---|---|---|---|
| Starwood Hotels & Resorts | HOT | 2016-09-22 | $77.05 |
| Legg Mason | LM | 2020-07-31 | $49.99 |
| First Republic Bank | FRCB | 2026-07-24 | $0.0004 |

Starwood's last trade was $77.05 on 22 September 2016, the session before Marriott completed the acquisition, so a position in it was cashed out near its peak. First Republic tells the opposite story and tells it in a way that matters for measurement: taken into FDIC receivership on 1 May 2023, the shares still change hands over the counter at four hundredths of a cent. A backtest keyed on "is this ticker still in the index" throws that away and books nothing. A backtest that keeps the position books a total loss, which is what actually happened to the shareholder.

## Does the ticker still point at the same company?

Not always, and the failure is silent.

```python
info = xfl.resolve("EMC")
for e in info["data"]["EMC"]["entities"]:
    print(e["name"], e["entity_id"], e["ticker_valid_from"], "->", e["ticker_valid_to"])
```

Output:

```
E M C CORP MA    8539   1988-03-22 -> 2016-09-06
GLOBAL X FUNDS  69408   2023-05-15 -> None
```

EMC Corp used that symbol until Dell acquired it in September 2016. Seven years later an exchange-traded fund took it over. A price request for EMC today returns the fund, which is correct behaviour and completely wrong for a study of the 2016 S&P 500. Permanent entity identifiers are what keep the two apart; the same problem shows up under [ticker changes at META, GM and DELL](https://xfinlink.com/blog/entity-resolution-ticker-changes-python).

## Where can you get the data?

Every entry below was checked against the provider's own pages or API on 27 July 2026.

| Source | What it provides for this problem |
|---|---|
| Wikipedia, "List of S&P 500 companies" | A current components table plus a table their own heading calls "Selected changes to the list of S&P 500 components" |
| Alpha Vantage `LISTING_STATUS` | CSV of active or delisted US listings, with `ipoDate` and `delistingDate`, filtered by a `date` parameter |
| Massive (`polygon.io`) all-tickers | A `date` parameter to "retrieve tickers available on that date" and an `active` flag for delisted symbols |
| yfinance | Per-symbol data from Yahoo Finance; the API reference lists no function for index constituents or membership history |
| xfinlink | `index(as_of=)` membership on any date, `index_events()` as a dated add/remove log, prices for delisted companies, and `resolve()` for reassigned symbols |

Alpha Vantage and Massive both answer the question "which securities existed and traded on this date", which is genuinely useful and covers the delisting half of the problem. Neither answers "which of them were in the index on that date", and that is the half the backtest depends on. Working the membership question out from index provider press releases is possible; it is also weeks of work for something that should be one function call. The [xfinlink docs](https://xfinlink.com/docs) cover both endpoints, and full price history for delisted companies runs back to 1996 on the $29 [Pro plan](https://xfinlink.com/pricing).

## FAQ

**Is survivorship bias the same as look-ahead bias?**
No. Look-ahead bias uses information that was not public yet, such as a fiscal-year figure on the day the year ended rather than the day it was filed. Survivorship bias uses a universe that was not knowable yet. A backtest can suffer from either alone, and many suffer from both.

**Does survivorship bias always inflate returns?**
For long-only equity strategies, almost always, because the excluded names are disproportionately the ones that went to zero. The sign can flip for strategies designed to profit from distress, where removing the failures removes the gains.

**How far back does point-in-time membership need to go?**
As far back as the backtest starts, plus whatever lookback the signal needs. Ten years of history covers roughly 29% turnover in the S&P 500, so any test spanning a decade or more is dominated by this choice rather than nudged by it.

**Can this be done on a free plan?**
Historical membership snapshots, yes: a free key allows 50 as-of calls per day. Price histories for delisted companies from the 1990s and 2000s need a paid plan, since the free tier serves a rolling one-year window.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
