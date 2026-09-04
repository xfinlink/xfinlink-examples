# What Is Look-Ahead Bias in Backtesting?

Look-ahead bias is what a backtest does when it acts on information that was not public on the day it trades. The usual version is arithmetical rather than dramatic: a company's fiscal year ends, the strategy reads that year's revenue with the period-end date attached, and the report those figures come from does not reach the market for another month. Microsoft's fiscal 2026 ended on 30 June 2026, and its annual report was filed on 29 July 2026 (verified against data.sec.gov, 4 September 2026). The shares rose 15.5 percent the following session. A backtest that ranked Microsoft on fiscal 2026 revenue at the end of June collected that move on the strength of a document nobody outside the company had read.

The repair is a discipline rather than a purchase. Admit no figure into the simulation before the date it could have been read, and pick the universe as it stood on the trade date rather than as it stands now.

## How Does Look-Ahead Bias Get Into a Backtest?

Nearly always through a join. Statement rows are keyed by period end, because period end is when the accounting stops, while publication is a separate event some weeks later that most data formats give you no reason to think about. Merge a fundamentals table onto a price table on the period-end date and the leak is in, on every row, quietly.

The universe is the other route. Screening today's index members across a ten-year window imports the knowledge of which companies survived long enough to be members today. That is [survivorship bias](/blog/what-is-survivorship-bias-in-backtesting) rather than look-ahead bias in the strict sense, but the two travel together and both push results the same way.

Neither failure announces itself. A leaking backtest runs clean, draws a plausible equity curve, and passes every test you write for it, because the arithmetic is not what went wrong. The information set is.

## When Did the Number Actually Become Public?

The SEC sets the outer bound, and it varies with the size of the filer.

| Filer status | Annual report (Form 10-K) | Quarterly report (Form 10-Q) |
|---|---|---|
| Large accelerated filer (public float of $700 million or more) | 60 days after fiscal year end | 40 days after quarter end |
| Accelerated filer | 75 days | 40 days |
| All other filers | 90 days | 45 days |

The 60-day and 75-day annual deadlines and the 40-day quarterly deadline come from the SEC's rule on accelerated filer definitions and periodic report deadlines; the 90-day and 45-day figures are the deadlines retained for filers outside those categories (both verified against sec.gov, 4 September 2026).

Those are limits rather than schedules. The six annual reports used later in this guide were filed between 22 and 41 days after the period ended: Oracle at 22 days, Microsoft at 29, Nvidia at 31, Apple at 34, Costco at 38, Walmart at 41 (data.sec.gov submissions API, verified 4 September 2026). Headline revenue and earnings usually reach the market earlier still, in an earnings release furnished on Form 8-K, which is why Microsoft's jump landed on 30 July rather than in August.

Those publication dates are public, if you would rather lag to the day than to the bound. The SEC's submissions API returns `filingDate` and `acceptanceDateTime` for every accession, at no cost and with no key. Alpha Vantage returns `reportedDate` and `reportTime` on the quarterly entries of its EARNINGS endpoint, while its INCOME_STATEMENT response carries `fiscalDateEnding` and no publication date (verified against alphavantage.co demo responses, 4 September 2026). yfinance documents `Ticker.get_earnings_dates(limit=12, offset=0)` (ranaroussi.github.io/yfinance, as of September 2026). Any of them beats assuming the figure was known on the day the quarter closed.

Lagging to the statutory deadline instead needs nothing beyond the period end, and it is the version that cannot leak: on that rule no figure enters the backtest before the market held it, at the cost of a few days of live edge. Academic practice is more cautious again. The Fama-French portfolios for "July of year t to June of t+1" use book equity for fiscal year t-1 (Kenneth French's data library, verified 4 September 2026), a gap of at least six months.

## How Much Price Action Sits Inside That Window?

Enough to decide the outcome of a backtest. The code below takes six large companies with six different fiscal calendars, marks the earliest date each annual figure was certainly public under the 60-day rule, and measures what the share price did in between.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["MSFT", "ORCL", "NVDA", "WMT", "AAPL", "COST"]

f = xfl.fundamentals(tickers, period_type="annual", fields=["revenue"], start="2025-01-01")
f["period_end"] = pd.to_datetime(f["period_end"])
f = f.sort_values("period_end").groupby("ticker").tail(1)
f["usable_from"] = f["period_end"] + pd.Timedelta(days=60)  # Form 10-K deadline

p = xfl.prices(tickers, start="2025-08-01", end="2026-09-03", fields=["adj_close"])
p["date"] = pd.to_datetime(p["date"])

rows = []
for r in f.itertuples():
    s = p[p["ticker"] == r.ticker].sort_values("date")
    at_end = s[s["date"] >= r.period_end].iloc[0]["adj_close"]
    at_pub = s[s["date"] >= r.usable_from].iloc[0]["adj_close"]
    rows.append({"ticker": r.ticker, "fiscal_year": r.fiscal_year,
                 "period_end": r.period_end.date(), "revenue_musd": r.revenue,
                 "usable_from": r.usable_from.date(),
                 "price_move_pct": round(100 * (at_pub / at_end - 1), 1)})

print(pd.DataFrame(rows).sort_values("period_end").to_string(index=False))
```

```
ticker  fiscal_year period_end  revenue_musd usable_from  price_move_pct
  COST         2025 2025-08-31        275235  2025-10-30            -2.0
  AAPL         2025 2025-09-27        416161  2025-11-26             9.1
  NVDA         2026 2026-01-25        215938  2026-03-26            -8.2
   WMT         2026 2026-01-31        706413  2026-04-01             0.5
  ORCL         2026 2026-05-31         67357  2026-07-30           -48.6
  MSFT         2026 2026-06-30        331839  2026-08-29            36.0
```

Costco and Walmart barely moved. Oracle lost close to half its value, Microsoft gained more than a third, and both of those windows contain the release that explains the move.

Widening the same measurement to the whole index gives the shape of the distribution. Of the 504 companies in the S&P 500 today, 498 have an annual period whose 60-day window has already closed, and 495 of those have a trading session at both ends of it. Across those 495, the median absolute price move over the window is 11.1 percent, the middle half of the sample falls between 5.1 and 19.4 percent, and 53.7 percent of companies move by more than a tenth. Some of that is ordinary market movement rather than a reaction to the accounts, and the release of the accounts sits inside the same window for almost every one of them.

## What Else Leaks?

Restated figures are the subtler case. A number published in February can be corrected a year later, and a backtest reading the corrected version assumes knowledge of an amendment that did not exist at the trade date. A generous lag absorbs part of that, and anchoring the join on `period_end` rather than on the fiscal-year label keeps a company's own re-labelling of its years out of the merge, a point covered in [annual vs quarterly financial data](/blog/annual-vs-quarterly-financial-data).

Price levels are the other. Split-adjusted history is rescaled by splits that had not happened at the simulated trade date, so any rule that reads a price level rather than a return sees a series no contemporary investor saw. Returns are unaffected by the rescaling, which is why most strategies never notice; a filter such as "trades below $10" is affected, and the mechanics are set out in [split adjustment explained](/blog/split-adjustment-explained).

## What Does a Look-Ahead-Safe Setup Need?

Two things, and the second is the one people usually skip.

A period end you can lag from. Every xfinlink fundamentals row carries `period_end`, `fiscal_year` and `fiscal_period`, so the safe date is one addition, as in the code above. For the six filings in that table, the `period_end` served matched the report date the SEC holds for the same filing exactly.

A universe as of the trade date. `xfl.index("sp500", as_of="2016-06-30")` returns the 500 companies that stood in the index that day, 146 of which are not members now, with the date each one entered and the date it left attached. Screening a 2016 strategy against the 2026 membership list instead is the mistake that flatters a backtest most and is hardest to see afterwards.

Field lists and parameters are in the [docs](https://xfinlink.com/docs); the free tier covers a rolling year of history and full history comes with the paid plans on the [pricing](https://xfinlink.com/pricing) page.

## FAQ

**Is look-ahead bias the same as survivorship bias?** No, though they usually turn up together. One is a fact arriving too early; the other is a list of companies chosen with hindsight. Lagging every figure correctly still leaves the second problem untouched if the universe came from today's index membership.

**How many days should I lag fundamentals?** Ninety days after fiscal year end covers every filer status for an annual report, and 45 days covers every filer status for a quarter. Tighten it only where you hold the actual publication date for that filing.

**Does look-ahead bias only affect fundamentals?** No. Anything revised after the fact carries it, including restated statements, index membership lists rebuilt from a current page, and analyst estimate histories that have been overwritten. Prices are the least affected, because a closing price is final on the day it is set.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
