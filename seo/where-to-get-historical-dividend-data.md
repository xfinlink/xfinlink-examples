# Where to Get Historical Dividend Data for Stocks

Four sources carry historical dividends, and they are not interchangeable. A company's own investor relations page is authoritative but covers one company. SEC filings report the total cash a company paid over a period, not the amount per share attached to any particular date. Market data APIs carry a per-share cash amount stamped on the ex-dividend date, which is the form every yield calculation and total-return series actually needs. Commercial terminals carry all of it and cost accordingly. For anything programmatic, the API route is the only one that scales, and the field to look for is a per-share cash amount on the ex-dividend date.

## What does a single dividend record contain?

A dividend has four dates and one amount, and confusing them produces wrong numbers rather than approximate ones.

The **declaration date** is when the board announces the payment. The **ex-dividend date** is the first session on which a buyer does not receive it; the price drops by roughly the dividend amount that morning, which is why this date and no other belongs in a return calculation. The **record date** determines who is on the register. The **payment date** is when cash actually arrives, often several weeks later.

Most datasets key on the ex-dividend date because that is the date the market prices. Some carry all four. A source that gives you only a payment date and an amount cannot be joined cleanly to a price series without introducing a timing error of a few weeks, which matters for anything measured at daily or monthly frequency.

The amount itself has its own trap: a regular quarterly payment and a one-off special distribution both arrive as "a dividend" in most feeds, with no flag separating them. A single large special can double a company's apparent trailing yield and then vanish the next year, which reads as a dividend cut when nothing was cut.

## Why do two sources report different dividend yields for the same stock?

Usually because they are computing different things under the same label.

Trailing yield sums the cash paid over the last twelve months. Indicated yield takes the most recent regular payment and multiplies by the payment frequency. Forward yield uses an estimate of the next twelve months. For a company that just raised its dividend, these three can sit two or three percentage points apart, and none of them is wrong.

The second cause is adjustment. If the price used in the denominator is a dividend-adjusted price rather than the traded price, the yield is not the yield an investor would receive. Split adjustment has a similar effect on the numerator: a per-share dividend paid before a 4-for-1 split needs restating onto the current share basis before it can be compared to a recent one, and not every source does that consistently.

The third cause is simply the treatment of specials, described above.

## What are the options in Python?

| Source | What you get | History | Cost, as of August 2026 |
| --- | --- | --- | --- |
| yfinance | `Ticker.dividends`, `splits`, `actions`, `capital_gains` | Varies by ticker | Free library. Its README states it is "intended for research and educational purposes" and that "the Yahoo! finance API is intended for personal use only" |
| Alpha Vantage | `DIVIDENDS` endpoint returning `declaration_date`, `ex_dividend_date`, `record_date`, `payment_date`, `amount` | Full history per symbol | Free tier limited to 25 API requests per day; premium plans start at $49.99 a month for 75 requests a minute |
| xfinlink | `dividend` on the ex-date row of the price frame, `return_daily` as total return, and dividend metrics through `xfl.metrics()` | Daily prices back to 1996 on paid plans; free key covers a rolling one-year window | Free tier, paid plans for full history |

Alpha Vantage wins a point here worth stating plainly: it returns all four dates on every record, which xfinlink does not. If your work depends on the gap between record date and payment date, that endpoint is the better tool, and it is cheap to test against with the documented demo key.

The tradeoff runs the other way once the unit of work stops being one symbol. Dividends are almost never the end goal; they are an input to a yield screen, a payout study, or a total-return series, all of which need prices for the same companies over the same window. A dividend feed that arrives separately from the price series has to be joined to it, and that join is where ticker changes and share-class mix-ups do their damage.

## How do you pull a dividend history and compute a yield?

The `dividend` column arrives inside the price frame, on the ex-date row, in the same units as the traded close beside it.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices("KO", start="2015-01-01", end="2026-08-01",
                fields=["close", "adj_close", "return_daily", "dividend"])
df["date"] = pd.to_datetime(df["date"])

cash = df[df["dividend"].fillna(0) > 0][["date", "dividend", "close"]]
last12 = cash[cash["date"] >= cash["date"].max() - pd.Timedelta(days=365)]["dividend"].sum()
last_close = df.iloc[-1]["close"]
print(f"trailing 12-month cash {last12:.2f}, yield {100 * last12 / last_close:.2f}%")

price_ret = df.iloc[-1]["adj_close"] / df.iloc[0]["adj_close"] - 1
total_ret = (1 + df["return_daily"].fillna(0)).prod() - 1
print(f"price return {100 * price_ret:.1f}%, total return {100 * total_ret:.1f}%")
```

Real output:

```
ex-dividend dates on file: 46 from 2015-03-12 to 2026-06-15
      date  dividend  close
2025-06-13      0.51  71.02
2025-09-15      0.51  66.21
2025-12-01      0.51  71.95
2026-03-13      0.53  77.34
2026-06-15      0.53  80.91

trailing 12-month cash dividend: 2.08
last close 87.59 on 2026-07-31  ->  yield 2.37%

2015-01-02 to 2026-07-31
  price return  (adj_close):    107.9%
  total return  (return_daily): 185.1%
  dividends contributed:         77.2 points
```

Two things in that output are worth reading closely. The quarterly rate steps from 0.51 to 0.53 at the March 2026 ex-date, which is why the trailing sum of 2.08 sits between the 2025 annual total of 2.04 and the 2.12 the new rate implies. And over eleven and a half years, dividends account for 77 of Coca-Cola's 185 points of total return. Any study that measures performance on price alone throws away about forty percent of what the shareholder earned.

## What quietly breaks a dividend backtest?

Two things, and neither announces itself.

The first is survivorship. Build a universe from today's list of dividend payers, then run it back ten years, and you have selected on the outcome: companies that cut their dividend and fell out of the index are absent from the sample, so the strategy looks safer than it was. The fix is a point-in-time roster, read as it stood on the formation date. `xfl.index("sp500", as_of="2015-12-31")` returns the 500 companies that were in the index that day, including the ones that later left.

The second is identity. Tickers get reassigned, and a company that changes its symbol will silently split into two partial histories if the join key is a string of letters. Keying on `entity_id` instead of the ticker keeps a company's dividends and prices attached to the company rather than to the letters it happened to trade under. Our write-up on [ticker recycling](https://xfinlink.com/blog/ticker-recycling-dangers-python) has the measured version of that problem, and [survivorship bias in backtesting](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting) covers the first.

For a worked example that uses both, see [does a high dividend yield predict a dividend cut](https://xfinlink.com/blog/high-dividend-yield-cut-risk-python), which rebuilds eleven years of dividend rates from ex-date cash across point-in-time index members. Full field definitions are in the [docs](https://xfinlink.com/docs), and plan limits are on the [pricing page](https://xfinlink.com/pricing).

## FAQ

**Does adjusted close already include dividends?**
Not in every dataset, and the difference is large. In xfinlink, `adj_close` is adjusted for splits only, so it is a clean price series; `return_daily` is the total return including dividends. Check which convention your source uses before comparing performance numbers across providers.

**How do I separate regular dividends from specials?**
No feed flags them reliably, so infer it: take the year's payments, compute the median, and discard anything far above it. A payment more than about one and a half times the year's median is almost always a special.

**Can I get dividend data for delisted companies?**
Only if the source keeps the delisted company's price history at all, which many do not. This is the same survivorship problem in a different guise, and it is worth testing on a known case before trusting a backtest built on the answer.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
