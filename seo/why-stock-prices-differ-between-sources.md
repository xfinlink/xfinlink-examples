# Why Do Stock Prices Differ Between Data Sources?

Two sources give two closing prices for the same stock on the same day for four reasons, and they are worth checking in this order: one number is adjusted and the other is as-traded; the two sources are describing different companies that shared a ticker; the adjusted series has been rebased since the last download; or the rows are not the same trading day. The first explanation covers most disagreements, and it is the fastest to rule out because the ratio between the two numbers is usually a clean split factor.

## Is one price adjusted and the other raw?

A closing price is a fact. An adjusted closing price is a calculation applied to that fact, and every vendor documents its own default. Apple's 4-for-1 split on 31 August 2020 shows the size of the effect:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.prices("AAPL", start="2020-08-26", end="2020-09-02",
                fields=["close", "adj_close", "split_ratio"])
print(df[["date", "close", "adj_close", "split_ratio"]].to_string(index=False))
```

```
      date     close  adj_close  split_ratio
2020-08-26 506.09000 126.522500          NaN
2020-08-27 500.04001 125.010002          NaN
2020-08-28 499.23001 124.807502          NaN
2020-08-31 129.03999 129.039990          4.0
2020-09-01 134.17999 134.179990          NaN
2020-09-02 131.39999 131.399990          NaN
```

Either column is defensible. What breaks a comparison is reading 499.23 from one source and 124.81 from another and concluding that one of them is wrong. Divide the larger by the smaller: a result of exactly 4.0, or 2.0, or 1.5, or any product of past split ratios, means the two sources are simply looking at different columns. The mechanics of that adjustment, including what a reverse split does to a screen, are covered in [split adjustment explained](/blog/split-adjustment-explained).

Defaults differ, so the same three lines of code against two APIs can return two different quantities without either being an error. Every row below was checked against the vendor's own documentation on 2 September 2026.

| Source | What the default daily close means | Getting the other column |
| --- | --- | --- |
| yfinance `download()` | `auto_adjust` defaults to `True`, so the OHLC columns come back adjusted ([docs](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)) | Pass `auto_adjust=False` for as-traded values |
| Alpha Vantage `TIME_SERIES_DAILY` | Raw as-traded open, high, low, close and volume ([docs](https://www.alphavantage.co/documentation/)) | Adjusted close comes from `TIME_SERIES_DAILY_ADJUSTED`, which the same page lists as a premium function |
| Massive custom bars | `adjusted` defaults to `true` and covers splits only ([docs](https://massive.com/docs/rest/stocks/aggregates/custom-bars)) | Set `adjusted=false` for unadjusted bars |
| xfinlink `xfl.prices()` | `close` is always the raw as-traded price; `adj_close` sits beside it in the same row, split-adjusted ([docs](/docs)) | Both columns arrive together, with `split_ratio` and `dividend` on the same rows |

yfinance also ships a `repair` flag, documented on that same page on 2 September 2026 as detecting currency unit 100x mixups and attempting a repair, which is a genuinely useful thing to have in a free library. For a reconciliation, though, the property that matters is whether both quantities reach you in one response, because that is what lets you prove which column the other source used instead of guessing.

## Do both sources mean the same company?

This one produces disagreements that no adjustment ratio explains, and it is the failure that survives every sanity check. Ticker symbols get reassigned. GM belonged to the General Motors Corporation that filed for bankruptcy in 2009, and it belongs to the General Motors Company that listed in 2010; the two are separate legal entities with separate filings and separate price histories.

```python
info = xfl.resolve("GM")
for e in info["data"]["GM"]["entities"]:
    print(e["entity_id"], "|", e["name"], "|",
          e["ticker_valid_from"], "->", e["ticker_valid_to"])
```

```
4 | General Motors Corporation (pre-2009 bankruptcy) | 1962-07-02 -> 2009-06-01
5 | General Motors Company | 2010-11-18 -> None
```

Ask for GM in June 2008 and the answer should be the company that actually traded then:

```python
df = xfl.prices("GM", start="2008-06-02", end="2008-06-06", fields=["close"])
print(df[["date", "entity_id", "entity_name", "close"]].to_string(index=False))
```

```
      date  entity_id                                      entity_name  close
2008-06-02          4  General Motors Corporation (pre-2009 bankruptcy)  17.44
2008-06-03          4  General Motors Corporation (pre-2009 bankruptcy)  17.58
2008-06-04          4  General Motors Corporation (pre-2009 bankruptcy)  17.01
2008-06-05          4  General Motors Corporation (pre-2009 bankruptcy)  17.05
2008-06-06          4  General Motors Corporation (pre-2009 bankruptcy)  16.22
```

A source keyed on ticker strings alone has three ways to answer that request: the old company, the new one, or an empty result. All three appear in practice, which is why a ticker-level disagreement in the pre-2010 window is often not a price disagreement at all. The row above carries `entity_id`, so the question of which company is being priced has an answer printed next to the number. The related problem of which identifier to key a database on is covered in [ticker vs CIK vs FIGI](/blog/ticker-vs-cik-vs-figi), and the case of companies that stopped trading entirely in [historical data for delisted stocks](/blog/how-to-get-delisted-stock-data).

## Did the number change since the last download?

An adjusted price series is defined relative to the present, so its history moves whenever a new adjustment event occurs. A dividend-adjusted series is rebased on every ex-dividend date, which for a quarterly payer means four rewrites of the entire history each year. A split-only adjusted series moves only when the company splits its stock, which is rare for most names and never for many.

That difference decides whether two downloads taken a month apart can be compared at all. xfinlink stores `adj_close` split-adjusted and leaves cash dividends in their own `dividend` column, so pulling the same date twice returns the same number, and the raw `close` beside it never moves at all. When a reconciliation shows a small percentage gap that is not a split factor and not a company mismatch, an intervening dividend in the other source's adjustment basis is the usual culprit, and comparing raw closes settles it immediately.

## Are the two rows the same trading day?

Check the boring things last, and check them properly. Compare the whole overlapping window rather than the most recent row, because the newest row is where freshness differences and late corrections live: one source may have closed its file for the day and the other not. Confirm that both series exclude the same non-trading days. Confirm that a date in the other source is a trading date rather than a timestamp converted from another timezone, which shifts an entire series by one row and produces a disagreement on every single day.

A useful habit: merge the two series on date, take the ratio rather than the difference, and look at its distribution. A constant ratio is an adjustment basis mismatch. A ratio of 1.0 everywhere except a handful of days is a data gap or a correction. A ratio that wanders is two different companies, or two different securities of the same issuer.

## FAQ

**Which price should be stored in a database?**
The raw close, plus the split ratios and dividends needed to derive anything else. Storing only an adjusted column means the stored history changes meaning every time a new adjustment event occurs, and nothing already saved can be reconciled against a filing or a broker statement.

**Is a difference of a few cents worth chasing?**
Usually not on a single day, but it is worth knowing the cause before it is ignored, because a systematic few cents on every row is a convention difference and a random few cents is rounding or a different trade set. The first affects a backtest; the second rarely does.

**Can two sources disagree even when both are correct?**
Yes, and it is common. Different adjustment defaults, different treatment of a recycled ticker, and different file-closing times each produce two defensible numbers. The reconciliation is a matter of establishing which convention each source used, which is why sources that return the raw and the adjusted values together are easier to audit. The [xfinlink docs](/docs) list the price fields and their conventions, and the [free tier](/pricing) covers a rolling twelve months of history, which is enough to run the comparison against whatever is in use today.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
