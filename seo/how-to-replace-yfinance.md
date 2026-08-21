# How to Replace yfinance in a Python Script

Swapping the import takes a minute. Getting the same numbers back takes an afternoon, because the two libraries disagree about what a column called `close` contains. As of August 2026, the yfinance documentation gives `yf.download()` the defaults `auto_adjust=True` and `multi_level_index=True`, so a multi-ticker call returns adjusted OHLC under a MultiIndex. `xfl.prices()` returns one long DataFrame in which `close` is the raw as-traded price, `adj_close` is split-adjusted, and `return_daily` is total return including dividends. Fix the column meanings first and the frame shape second; the rest of the migration is renaming.

## What changes inside the DataFrame?

Three things move, and only one of them is cosmetic.

Adjustment stops being a hidden default. The yfinance reference documents `auto_adjust` as "Adjust all OHLC automatically? Default is True", which means the price columns of a plain `yf.download()` are already modified before the script sees them. xfinlink splits that decision across separate columns rather than a flag: `close` is never modified, `adj_close` carries the split adjustment, and `return_daily` carries the total return that a performance calculation actually needs. A script that swaps the call, keeps reading `close`, and computes a percentage change will now step across every split in the sample.

The frame shape changes from wide to long. One `xfl.prices(["AAPL", "MSFT"])` call returns rows, not nested columns, with `ticker` and `entity_id` as identifier columns beside the date. Anything written against a MultiIndex needs a `pivot`, which is one line, and anything written as a groupby gets simpler.

Corporate actions arrive in the same frame instead of separate attributes. yfinance documents `actions=False` as the `download()` default and exposes `Ticker.dividends`, `Ticker.splits` and `Ticker.actions` separately; xfinlink returns `dividend` and `split_ratio` as fields of the price frame, on the ex-date and split date respectively.

## Which call replaces which?

Every yfinance name below is a documented attribute or function in the yfinance reference as of August 2026.

| yfinance | xfinlink | Note |
| --- | --- | --- |
| `yf.download(tickers, ...)`, `Ticker.history()` | `xfl.prices(tickers, start=, end=, interval=, fields=)` | Daily and coarser; `interval` accepts `1d` through `1y` |
| `Ticker.dividends`, `Ticker.splits`, `Ticker.actions` | `fields=["dividend", "split_ratio"]` on `xfl.prices()` | Same frame as the prices |
| `Ticker.income_stmt`, `Ticker.balance_sheet`, `Ticker.cashflow`, `quarterly_*` | `xfl.fundamentals(ticker, period_type="annual"\|"quarterly")` | Built from SEC filings, annual back to 1950 on paid plans |
| `Ticker.info` | `xfl.metrics(ticker, fields=[...])` | Returns a typed DataFrame; categories include valuation, profitability, leverage, growth |
| `Ticker.insider_transactions` | `xfl.insiders(ticker, ...)` | Form 3, 4 and 5 transactions, one row each |
| `Ticker.institutional_holders` | `xfl.holdings(ticker, quarter=...)` | Form 13F, one row per manager, security and quarter |
| No equivalent | `xfl.index("sp500", as_of="2015-06-30")` | Point-in-time constituents for sp500, ndx100, djia, russell2000 |
| No equivalent | `xfl.resolve("META")`, `xfl.prices(entity_id=...)` | Entity identity across ticker changes and reuse |

## What does yfinance still cover that xfinlink does not?

Intraday bars, for one. The yfinance documentation lists intervals down to `1m` with the caveat that "Intraday data cannot extend last 60 days", which is enough for a short-horizon study and something xfinlink does not serve at all; daily is the finest interval on this side. Options chains through `Ticker.option_chain`, analyst estimates and revisions, fund holdings through `funds_data`, and a news feed are all documented yfinance attributes with no counterpart here.

There is also nothing to sign up for. The quick start in the yfinance documentation constructs `yf.Ticker("MSFT")` with no authentication step, while xfinlink wants a key on every call. For a weekend script that pulls one chart, that difference is real.

Those are different jobs from the one a daily research pipeline does. Nothing forces an all-or-nothing move either: an options call can stay where it is while the daily history moves.

## How do you check the migration worked?

Two checks catch most of the damage, and a third confirms the frame shape. Run the first pair on a known split and on a large dividend payer, because those are the places where a column swap changes the answer without changing the shape of the output.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

px = xfl.prices("NVDA", start="2024-06-05", end="2024-06-12",
                fields=["close", "adj_close", "split_ratio"]).sort_values("date")
px["split_ratio"] = px["split_ratio"].fillna(1.0)
print(px[["date", "close", "adj_close", "split_ratio"]].to_string(index=False))

vz = xfl.prices("VZ", start="2021-08-02", end="2026-07-31",
                fields=["adj_close", "return_daily"]).sort_values("date")
price_return = vz["adj_close"].iloc[-1] / vz["adj_close"].iloc[0] - 1
total_return = (1 + vz["return_daily"].iloc[1:]).prod() - 1
print(f"\nVZ 2021-08-02 to 2026-07-31: price {price_return:.1%}, total {total_return:.1%}")

wide = (xfl.prices(["AAPL", "MSFT", "NVDA"], start="2024-06-05", end="2024-06-12",
                   fields=["adj_close"])
        .pivot(index="date", columns="ticker", values="adj_close"))
print("\n" + wide.to_string())
```

```
      date      close  adj_close  split_ratio
2024-06-05 1224.40002 122.440002          1.0
2024-06-06 1209.97998 120.997998          1.0
2024-06-07 1208.88000 120.888000          1.0
2024-06-10  121.79000 121.790000         10.0
2024-06-11  120.91000 120.910000          1.0
2024-06-12  125.20000 125.200000          1.0

VZ 2021-08-02 to 2026-07-31: price -16.1%, total 2.7%

ticker           AAPL       MSFT        NVDA
date                                        
2024-06-05  195.87000  424.01001  122.440002
2024-06-06  194.48000  424.51999  120.997998
2024-06-07  196.89000  423.85001  120.888000
2024-06-10  193.12000  427.87000  121.790000
2024-06-11  207.14999  432.67999  120.910000
2024-06-12  213.07001  441.06000  125.200000
```

NVIDIA split 10-for-1 on 10 June 2024. The `close` column drops from 1208.88 to 121.79 across that date because that is what the stock traded at; `adj_close` runs 120.888 to 121.790 without a step. If a migrated script shows a 90% single-day loss anywhere in 2024, it is reading `close` where it used to read an adjusted series.

Verizon is the second check. Over five years to 31 July 2026 the price fell 16.1% while the total return was positive 2.7%, so the sign of the answer depends entirely on which column the script reads. Dividend-paying names are where a quiet column mismatch turns into a wrong conclusion rather than a small error. More on that distinction in [split adjustment explained](https://xfinlink.com/blog/split-adjustment-explained) and [where to get historical dividend data](https://xfinlink.com/blog/where-to-get-historical-dividend-data).

## What can the script do after the swap?

Ticker strings stop being the primary key. `xfl.resolve()` returns a stable `entity_id` for each company that has used a ticker, along with its SEC CIK and its FIGI, and the price, statement, metric, insider and holdings functions each accept `entity_id=` in place of a ticker. That is what makes a company reachable after it renames itself, and what stops a study from picking up a different issuer that inherited the same three letters later. Delisted names stay reachable the same way, which is the usual reason a yfinance-era backtest reads too optimistic; [survivorship bias in backtesting](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting) covers that case in full.

Universes gain a date. `xfl.index("sp500", as_of="2015-06-30")` returns the roster as it stood on that day rather than today's members mapped backwards, so a screen run over history is not quietly restricted to companies that survived to the present.

Statements, insider transactions and 13F holdings sit behind the same identifier as the prices, so a pipeline that already resolved a company does not need a second identity system to join filings to bars. An [MCP server](https://xfinlink.com/docs) exposes the same functions to a language model if the eventual consumer is an assistant rather than a notebook.

Start on the free key, which covers a rolling one-year window at 100 requests a day and one ticker per call. Paid plans lift both the history floor and the per-call ticker cap; the [pricing page](https://xfinlink.com/pricing) has the current numbers.

## FAQ

**Can both libraries live in the same project?**
Yes. They are ordinary Python packages with no shared state. Keeping yfinance for an options chain while daily history comes from `xfl.prices()` is a normal setup.

**What replaces `auto_adjust=True`?**
Two columns instead of one flag: `adj_close` for a split-adjusted price series, `return_daily` for total return including dividends. Compound `return_daily` rather than differencing `adj_close` whenever dividends matter.

**Is Yahoo Finance data usable in a commercial product?**
The yfinance README states that the library "is intended for research and educational purposes" and that "the Yahoo! finance API is intended for personal use only", and it points readers to Yahoo's terms. The longer answer is in [can you use Yahoo Finance data commercially](https://xfinlink.com/blog/can-you-use-yahoo-finance-data-commercially).

**How far back does the history go?**
Daily prices reach 1996, financial statements 1950, and institutional holdings 1978 on paid plans. A free key sees a rolling twelve months of each.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
