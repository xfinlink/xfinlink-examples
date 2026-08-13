# How to Get Historical Market Cap Data in Python

Historical market capitalisation is not something you can read off a price chart, because market cap is price multiplied by the share count **on that same day**, and the share count changes. The reliable way to get it in Python is to pull a daily series where both legs are stored per date, rather than reconstructing it from a current share count. With xfinlink that is a single call: `xfl.prices(ticker, fields=["market_cap"])` returns one market cap value per trading day, computed from the price and the shares outstanding as they stood on that day.

Most free tools do not offer this. They give you a current market cap and a price history, and leave you to combine them, which is exactly the combination that produces wrong answers.

## Why is historical market cap harder to get than price?

Price is a single observed number. Market cap is a product of two series that both move, and they move for different reasons.

Share counts change through buybacks, issuance, acquisitions paid in stock, and splits. Buybacks and issuance move the count by a few percent a year. Splits move it by multiples, and they are the reason most reconstructions fail badly rather than slightly.

There is a second complication. Share counts are reported in filings, so they update quarterly rather than daily, and the figure a company files is as-reported for that period. Any daily market cap series therefore has to carry the correct as-filed count forward between filings, matched to the right date. That bookkeeping is the actual work, and it is why price history alone will not substitute.

## What goes wrong if you multiply price by today's share count?

The error is not small. Nvidia split its stock twice between 2019 and 2025, four-for-one and then ten-for-one, a cumulative factor of forty.

Take the last trading day of 2019. Nvidia closed at $235.30 with 0.61 billion shares outstanding, so its market cap was about $144 billion. Multiply that same 2019 closing price by the 24.49 billion shares outstanding at the end of 2025 and you get $5,762 billion, which would have made Nvidia the largest company in the world at the end of 2019 by a wide margin.

```
Market cap on 2019-12-31
  correct (that day's price x that day's share count): $144bn
  naive   (that day's price x today's share count):    $5,762bn
  overstatement: 40.0x
```

A forty-fold error is obvious enough to catch. The dangerous version is subtler: a company that bought back 15% of its shares over a decade produces a market cap series that is quietly wrong by 15%, in a direction that correlates with the very buyback behaviour a study might be testing. Nothing in the output flags it.

The same trap appears with per-share fundamentals, and it is worth reading our note on [split adjustment](https://xfinlink.com/blog/split-adjustment-explained) before mixing adjusted prices with as-filed share counts.

## How do you pull a daily market cap series?

Request `market_cap` as a field on the price endpoint. Each row is one ticker on one trading day, with the share count that applied.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices(["NVDA", "AAPL"], start="2019-12-31", end="2025-12-31",
                fields=["close", "shares_outstanding", "market_cap"])
```

Year-end values from that series:

```
Year-end market capitalisation, USD billions
  date                NVDA        AAPL
  2020                 323       2,232
  2021                 735       2,902
  2022                 360       2,058
  2023               1,223       2,977
  2024               3,289       3,785
  2025               4,567       4,109
```

Because `shares_outstanding` comes back alongside, the calculation is auditable rather than a black box. If a figure looks wrong, both inputs are on the row.

For period-end snapshots aligned to fiscal reporting rather than the calendar, `xfl.metrics(fields=["market_cap"])` gives one value per fiscal period, which is the right shape for joining against income statement or balance sheet items. Full parameters are in the [docs](https://xfinlink.com/docs).

## What are the options?

| Approach | Historical series | Handles splits | Notes |
|---|---|---|---|
| Price history × current share count | Reconstructed | No | Wrong by the cumulative split factor; 40x for Nvidia since 2019 |
| Price history × quarterly share counts, joined manually | Yes, with work | If you handle it | Requires matching filing dates to trading dates yourself |
| yfinance | No | Not applicable | `Ticker.info["marketCap"]` returns a single current value; `Ticker.history()` returns prices. A request for a historical series has sat on the project's issue tracker as issue #1008 (checked 13 August 2026) |
| xfinlink `prices(fields=["market_cap"])` | Yes, daily | Yes | Share count stored per date and returned alongside |

yfinance is a reasonable choice when you need today's market cap for a handful of tickers in a script you will run once. It is a poor fit when the market cap has to be correct on a date in the past, which is the case for any backtest, any event study, and any size-sorted portfolio.

## Which share count should the calculation use?

Basic shares outstanding, as filed, is the standard input for market capitalisation, and it is what the `market_cap` field uses: shares outstanding multiplied by the closing price, both at the scale in force on that date.

Diluted share counts include options and convertible instruments that have not been exercised. They are the right denominator for per-share earnings and the wrong one for market value, since the market is not currently paying for shares that do not yet exist. Where the distinction matters for your work, both counts are available on the fundamentals endpoint, and our guide to [how shares outstanding are reported](https://xfinlink.com/blog/how-are-shares-outstanding-reported) covers why providers disagree on the number.

## Does the universe matter as much as the number?

For a single company, no. For a cross-sectional study, it matters more than the market cap itself.

If you rank companies by size at a date in the past, you need the companies that existed at that date, not the ones that exist now. Pulling market caps for today's index members and reading them backwards produces a sample already filtered by survival, which is a larger error than most share-count problems. Our analysis of [index turnover](https://xfinlink.com/blog/sp500-index-turnover-survivorship-python) puts a number on it: only 238 of the 481 companies tracked from the S&P 500 of 2005 were still members twenty years later.

The fix is to build the universe from a point-in-time constituent list, then pull market caps for those companies on that date. Background on why this matters is in our guide to [survivorship bias in backtesting](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting).

## FAQ

**Can I calculate historical market cap myself from free data?**
Yes, if you can obtain a share count history with filing dates and match it to trading dates. The arithmetic is easy; sourcing a clean as-filed share count series with the right effective dates is the part that takes time.

**Why does my market cap differ from what a website shows?**
The most common causes are a different share count basis (basic against diluted), a different as-of date for the count, and the treatment of multiple share classes. Companies with dual-class structures need every class summed, and providers vary on whether they do that.

**How far back does the daily series go?**
Daily prices with market cap run from 1996. Every paid plan carries the full history, and the free tier covers a rolling one-year window; details are on the [pricing page](https://xfinlink.com/pricing).

**Does the series adjust for splits retroactively?**
No, and that is deliberate. The share count on each row is what the company had outstanding on that date, so the market cap is correct as of that date rather than restated onto today's share basis. That is what you want for any historical comparison.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
