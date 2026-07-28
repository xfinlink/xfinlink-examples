**When Do Corporate Insiders Actually Trade? Form 4 Timing Analysis in Python**

July 28, 2026 · INSIDER-FLOWS

**What's the question?**

Corporate insiders do not pick their trading dates freely. Almost every large US company closes an internal trading window some weeks before the fiscal quarter ends and reopens it a day or two after results are published. While management knows the quarter's numbers and the market does not, any trade invites a claim of dealing on material non-public information. Rule 10b5-1 supplies the standard exemption: an insider who adopts a written plan while the window is open, fixing amounts and dates in advance, may keep trading after it shuts.

Those policies are private, but the outcome is public. Form 4 requires an insider to report a transaction within two business days, including the date it was placed, so the policy can be recovered from behaviour. If windows work the way company handbooks describe, the distribution of open-market transactions across the quarter should be far from flat.

**The approach**

The sample is the 499 current S&P 500 constituents and every open-market buy and sell dated between 1 July 2021 and 30 June 2026. Grants, option exercises, tax withholding and gifts are excluded, since none is a decision about when to be in the market.

1. Pull Form 4 transactions in batches of 25 tickers, then quarterly period ends for the same batches.
2. Anchor each transaction to the most recent fiscal quarter end of its own filer, not the calendar quarter. A January-ending retailer sits five weeks out of phase with a December filer, and a shared calendar would blur the effect being measured.
3. Record the days between that quarter end and the transaction, keeping the following 91 days.
4. Take the median gap between period end and filing date as the mark for when results reach the market, using Q1 through Q3, since the fourth quarter is reported on the annual 10-K under its own deadline.
5. Compare transactions per day across three stretches of the quarter, separately for buys and sells.

**Code**

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END, BATCH = "2021-07-01", "2026-06-30", 25

tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique().tolist())
batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]

trades, quarters = [], []
for b in batches:
    trades.append(xfl.insiders(b, start=START, end=END,
                               transaction_type=["open_market_buy", "open_market_sell"],
                               fields=["ticker", "transaction_date", "transaction_type",
                                       "insider_role", "insider_name"],
                               max_rows=200000))
    q = xfl.fundamentals(b, start="2021-01-01", end="2026-07-28",
                         period_type="quarterly", max_rows=200000)
    quarters.append(q[["ticker", "period_end", "fiscal_period", "filing_date"]])

trades = pd.concat(trades, ignore_index=True)
quarters = pd.concat(quarters, ignore_index=True)

# typical interim reporting lag (the fourth quarter is reported on the annual 10-K)
q13 = quarters[quarters["fiscal_period"].isin(["Q1", "Q2", "Q3"])].copy()
q13["lag"] = (pd.to_datetime(q13["filing_date"]).dt.tz_localize(None)
              - pd.to_datetime(q13["period_end"])).dt.days
report_day = int(q13.loc[q13["lag"].between(1, 120), "lag"].median())

qe = (quarters[["ticker", "period_end"]].drop_duplicates()
      .assign(period_end=lambda d: pd.to_datetime(d["period_end"]))
      .sort_values("period_end"))
trades["transaction_date"] = pd.to_datetime(trades["transaction_date"]).dt.tz_localize(None)

m = pd.merge_asof(trades.sort_values("transaction_date"), qe,
                  left_on="transaction_date", right_on="period_end",
                  by="ticker", direction="backward")
m["day"] = (m["transaction_date"] - m["period_end"]).dt.days
m = m[m["day"].between(0, 91)]

sells = m[m["transaction_type"] == "open_market_sell"]
buys = m[m["transaction_type"] == "open_market_buy"]

for lo, hi in [(0, report_day - 1), (report_day, report_day + 29), (report_day + 30, 91)]:
    n = hi - lo + 1
    print(f"days {lo:>2}-{hi:<2}  "
          f"sells/day {sells['day'].between(lo, hi).sum() / n:7.1f}  "
          f"buys/day {buys['day'].between(lo, hi).sum() / n:6.1f}")
```

Full script with formatting and visualisation: [when-do-insiders-trade-form-4-timing-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/insider-transactions/when-do-insiders-trade-form-4-timing-python.py)

**Output**

![Open-market insider sells and buys by day of the fiscal quarter](/blog-images/when-do-insiders-trade-form-4-timing-python.png)

```
S&P 500 open-market insider transactions, 2021-07-01 to 2026-06-30
Universe: 499 tickers   Transactions placed in a mapped quarter: 74,511
Median 10-Q reporting lag: 32 days after fiscal quarter end (n=7,629, IQR 27-36)

Days after quarter end                   Sells     /day    Buys    /day
 0-31  quarter closed, report not out   17,158    536.2   1,376    43.0
32-61  first month after the report     33,891   1129.7   1,636    54.5
62-91  run-up to the next quarter end   19,177    639.2   1,273    42.4

Busiest selling day: day 51 (1,626 transactions)
Quietest stretch: days 80-91 at 253.1 sells/day, 22% of the post-report rate
Median transaction day: sells 46, buys 43
Sells per buy: 16.4
Officers: 48,632 sells, median day 47, 75.7% on or after day 32
Directors: 11,446 sells, median day 43, 78.6% on or after day 32

Sells on days 0-2: 2,508 (3.6% of sells) from 363 insiders; top 3 account for 409
```

**What this tells us**

Selling runs at 536 transactions per day while the quarter is closed and results are unpublished, then jumps to 1,130 per day in the month after the typical filing date, 2.1 times higher. The busiest day of the cycle is day 51, roughly three weeks after a median filer reports.

Across days 80 to 91, the fortnight in which the next period is closing, selling falls to 253 per day, 22% of the post-report rate. Nothing in market microstructure explains a collapse of that size on those dates. It matches policies that shut the window before the period ends, when management can already estimate the quarter it is about to report.

Buying traces the same shape with much weaker amplitude: 43.0, 54.5 and 42.4 transactions per day across the three stretches, a peak-to-trough ratio of 1.3 against 2.1 for sales. Purchases are rare, at one buy for every 16.4 sells, and mostly discretionary. Routine selling is what pre-arranged plans automate, so it inherits the calendar those plans are built around.

The spike at the start of the quarter belongs to that automation. Days 0 to 2 hold 3.6% of all sales and 3.8% of purchases land on day 0 alone, concentrated in 363 insiders of whom three account for 409 of the 2,508 early sales. Plans dated to the first trading day of a quarter execute whether or not the window is open. Role barely matters: officers place 75.7% of sales on or after day 32, directors 78.6%.

**So what?**

Any signal built on insider transactions inherits this calendar, and the naive versions confuse it with information. A screen that flags a company because insider selling rose sharply this week will fire on the mechanical post-earnings surge for hundreds of names at once; a screen hunting unusual quiet will fire on the fortnight before period close.

The correction is to standardise against day of quarter first. Compute the expected count for each day from 0 to 91, from the curve above or from the company's own history, and score observed activity against that baseline instead of a flat prior. The anchor has to be the filer's own fiscal quarter end.

Two refinements repay the effort: treat the day 0 to 2 cluster as its own series, since pre-arranged plans dominate it and carry no view on price, then weight what remains by how unusual it is. A purchase on day 85, when the aggregate rate sits at its floor, required an exception to the policy that stops everyone else.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
