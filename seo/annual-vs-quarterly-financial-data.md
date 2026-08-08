**Annual vs Quarterly Financial Data: Which to Use**

August 8, 2026 · GUIDES

Use annual data when the question is about a company's economics, and quarterly when the question is about timing. Valuation multiples, margin trends, and returns on capital are annual questions: a full year cancels seasonality and matches how companies actually report their audited position. Anything that depends on when information became known, including event studies, earnings-driven signals, and backtests with a rebalancing date, needs quarterly data because annual figures arrive once and sit stale for twelve months. Most people reach for quarterly by default because it looks like more information. It is more observations of a noisier series, which is a different thing.

## What is the actual difference?

An annual figure covers a fiscal year and comes from a 10-K. A quarterly figure covers roughly thirteen weeks and comes from a 10-Q, except the fourth quarter, which most companies do not file separately and which has to be derived by subtracting the first three quarters from the annual total.

That derivation is the first practical difference. If a provider does not compute the fourth quarter, the series has a hole every fourth period. If it computes it by subtraction, any adjustment the company booked at year end lands entirely in that quarter.

The second difference is the audit. Annual statements are audited; quarterly statements are reviewed, which is a lighter standard. Figures that move between the last quarterly filing and the annual one are common and are not errors.

## When is annual data enough?

For most cross-sectional work, annual data is not a compromise but the correct choice.

Ratios built on a full year are comparable across companies with different seasonal shapes. A retailer earning most of its profit in the December quarter and a software company earning evenly across the year cannot be ranked on a single quarter's margin without the ranking being mostly a statement about the calendar.

Long-horizon studies also fit annual data naturally. Our analysis of [whether high returns on capital persist](https://xfinlink.com/blog/does-high-roic-persist-fade-python) tracks quintiles of the S&P 500 over nine years, and quarterly figures would have added noise without changing a single conclusion, because the question is about competitive position rather than about any particular quarter.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

annual = xfl.fundamentals("AAPL", period_type="annual",
                          start="2024-01-01", end="2025-12-31",
                          fields=["revenue"])
```

## When do you need quarterly?

Quarterly data earns its place when the answer depends on a date.

Backtests are the clearest case. A strategy that rebalances in March using annual figures needs to know what was actually filed by March, and an annual series carries no filing date to check against. Reading a fiscal-year figure into a date before the company published it is look-ahead bias, and it inflates results in a way that survives every other sanity check. The related trap of dropping companies that disappeared is covered in our guide to [survivorship bias in backtesting](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting).

Quarterly data is also the only way to see the shape of a year. Apple's fiscal 2025 is a single revenue number in annual form; in quarterly form the December quarter is 30% of it.

```
Apple, quarterly revenue ($m), fiscal year ends late September
  FY2024 Q2  ends 2024-03-30  revenue   90,753  net income  23,636
  FY2024 Q3  ends 2024-06-29  revenue   85,777  net income  21,448
  FY2024 Q4  ends 2024-09-28  revenue   94,930  net income  14,736
  FY2025 Q1  ends 2024-12-28  revenue  124,300  net income  36,330
  FY2025 Q2  ends 2025-03-29  revenue   95,359  net income  24,780
  FY2025 Q3  ends 2025-06-28  revenue   94,036  net income  23,434
  FY2025 Q4  ends 2025-09-27  revenue  102,466  net income  27,466
  FY2026 Q1  ends 2025-12-27  revenue  143,756  net income  42,097
```

The September 2024 quarter shows why the distinction matters. Revenue of 94,930 sits in line with the quarters around it while net income drops to 14,736, a third below the quarter before it, because of a one-off charge. An annual figure absorbs that into a twelve-month total and a reader would never see it.

## Why do fiscal years not line up?

A fiscal year is whatever the company says it is, and comparing "2025" across companies is comparing different periods.

```
  AAPL  FY2025 ends 27 September 2025
  MSFT  FY2025 ends 30 June 2025
  WMT   FY2025 ends 31 January 2025
  NVDA  FY2025 ends 26 January 2025
  COST  FY2025 ends 31 August 2025
```

Walmart's fiscal 2025 ended nearly eight months before Apple's did. Any screen that groups on a fiscal-year label is silently mixing periods up to eight months apart, which matters a great deal in a year when conditions changed, such as 2020 or 2022.

Two habits fix this. Group on the period end date rather than the fiscal-year label when the comparison is macroeconomic, and check that fiscal year ends are exposed by whatever source you use before assuming a shared calendar. xfinlink returns `fiscal_year`, `fiscal_period`, `period_end` and `filing_date` on every fundamentals row, so the alignment question can be answered without a second lookup.

## How do you build a trailing twelve month figure?

Trailing twelve months combines the two: current as of the last filing, and a full year long, so seasonality cancels. Sum the four most recent quarters.

```python
q = xfl.fundamentals("AAPL", period_type="quarterly",
                     start="2023-01-01", end="2025-12-31",
                     fields=["revenue", "net_income"])
ttm = q.sort_values("period_end").tail(4)["revenue"].sum()
```

```
  trailing four quarters: 435,617
  most recent annual:     416,161
```

The gap between the two figures, 4.7%, is one quarter of growth that the annual number has not caught up to. Any multiple computed on revenue is therefore 4.7% higher on the annual figure than on the trailing twelve months, for the same company on the same day.

| | Annual | Quarterly | Trailing twelve months |
|---|---|---|---|
| Period covered | Fiscal year | About 13 weeks | Last four quarters |
| Updates | Once a year | Four times a year | Four times a year |
| Seasonality | Cancelled | Present | Cancelled |
| Audited | Yes | Reviewed only | Mixed |
| Best for | Ratios, long-horizon studies | Event studies, backtest timing | Current valuation multiples |

## Frequently asked questions

**Can annual figures be rebuilt from quarterly ones?**
For flow items such as revenue, net income and cash flow, yes: four quarters sum to the year. For balance-sheet items such as debt, cash and total assets, no, because each is a snapshot at a single date rather than an amount accumulated over the period. Take the year-end value instead of summing. The [field reference](https://xfinlink.com/docs) marks which is which.

**Should quarterly figures be annualised by multiplying by four?**
Only for a company with no seasonality, which is rare. Multiplying Apple's December quarter by four overstates its annual revenue by about 20%. Use a trailing twelve month sum instead.

**Why does a quarterly figure disagree with the same quarter shown in the annual report?**
Quarterly statements are reviewed rather than audited, and year-end adjustments are booked against the fourth quarter. A restated comparative in a later filing is the normal reason the two disagree, not an error in either.

**Which should a screener use?**
Trailing twelve months for anything divided by price, and annual for balance-sheet quality measures, which change slowly and are cleanest at year end. Related reading on a common reporting trap: [how shares outstanding are reported](https://xfinlink.com/blog/how-are-shares-outstanding-reported).

Both period types come from the same SEC filings and the same endpoint in xfinlink, switched with one parameter, and the [pricing page](https://xfinlink.com/pricing) sets out how much history each plan covers. The decision is about the question being asked, not about what a data source makes convenient.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
