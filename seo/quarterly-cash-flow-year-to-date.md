# Why Is Quarterly Cash Flow Year-to-Date in SEC Filings?

September 12, 2026 · GUIDES

Quarterly cash flow arrives year-to-date because the SEC requires it that way. Rule 210.10-01(c)(3) of Regulation S-X states that interim statements of cash flows "shall be provided for the period between the end of the preceding fiscal year and the end of the most recent fiscal quarter" (verified against the eCFR, September 2026). A second-quarter 10-Q therefore reports six months of cash flow, and a third-quarter 10-Q reports nine. To recover a single quarter, subtract the previous cumulative figure from the current one. The fourth quarter takes the annual total from the 10-K minus the nine-month figure, because no filing reports it on its own.

## What Does a 10-Q Actually Report?

Two statements inside one document follow different rules, which is why this catches people who have already handled quarterly revenue without trouble. Paragraph (c)(2) of the same rule requires interim statements of comprehensive income "for the most recent fiscal quarter" as well as for the year-to-date period. Paragraph (c)(3), covering cash flows, asks only for the cumulative period. Revenue and net income leave a 10-Q in both shapes; operating cash flow leaves in one.

Apple's fiscal 2025 shows the pattern without ambiguity. Every fact below carries the same start date, 29 September 2024, and a later end date:

```
2024-09-29 to 2024-12-28    3 months     29,935m   10-Q
2024-09-29 to 2025-03-29    6 months     53,887m   10-Q
2024-09-29 to 2025-06-28    9 months     81,754m   10-Q
2024-09-29 to 2025-09-27   12 months    111,482m   10-K
```

Read those four values as quarters and Apple appears to have raised its quarterly operating cash flow from $29.9bn to $111.5bn inside a year. The true quarters are the gaps between consecutive rows: 29,935, then 23,952, then 27,867, then 29,728.

## Why Is There No Fourth-Quarter Cash Flow Statement?

Companies file three 10-Qs and one 10-K a year. The 10-K covers twelve months and never isolates the final three. A fourth quarter is a derived figure by construction rather than a reported one, which is also why fourth-quarter figures from different vendors disagree more often than the other three.

Apple's tagged history makes the point concrete. Of every operating cash flow fact the company has filed under `NetCashProvidedByUsedInOperatingActivities`, sixteen carry a three-month duration, and all sixteen are first quarters, running from fiscal 2009 through fiscal 2026 (verified against data.sec.gov, September 2026). The other durations are six, nine and twelve months.

Anyone building a quarterly series from EDGAR therefore writes two pieces of logic: the subtraction itself, and a reset at the fiscal year boundary. The reset is where implementations usually break, because subtracting a fiscal first quarter from the prior year's annual total produces a number that is wrong and looks ordinary.

## How Do You Get Standalone Quarters Instead?

The code below pulls the same fiscal year twice. Once from the SEC as filed, once from xfinlink, where quarterly cash flow rows are already de-cumulated to single-quarter values at ingestion. Walmart is included as a second case because its fiscal year ends on 31 January, so its quarters and the calendar quarters never line up.

```python
import json
import urllib.request

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TAG = "NetCashProvidedByUsedInOperatingActivities"
HEADERS = {"User-Agent": "your name your.email@example.com"}  # SEC requires this


def as_filed(cik, fiscal_start):
    """Operating cash flow exactly as the company tagged it with the SEC."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{TAG}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    facts = json.load(urllib.request.urlopen(req))["units"]["USD"]
    return sorted({(f["start"], f["end"], f["val"], f["form"])
                   for f in facts if f["start"] == fiscal_start})


cases = [
    ("Apple fiscal 2025", "AAPL", "0000320193", "2024-09-29", "2024-10-01", "2025-10-01"),
    ("Walmart fiscal 2026", "WMT", "0000104169", "2025-02-01", "2025-02-05", "2026-02-05"),
]

for label, ticker, cik, fiscal_start, start, end in cases:
    print(label)
    print("  as filed with the SEC")
    for s, e, val, form in as_filed(cik, fiscal_start):
        months = round((pd.Timestamp(e) - pd.Timestamp(s)).days / 30.44)
        print(f"    {s} to {e}  {months:2d} months  {val / 1e6:>9,.0f}m  {form}")

    q = xfl.fundamentals(ticker, period_type="quarterly", start=start, end=end,
                         fields=["operating_cash_flow", "capital_expenditures",
                                 "free_cash_flow"])
    a = xfl.fundamentals(ticker, period_type="annual", start=start, end=end,
                         fields=["operating_cash_flow"])
    print("  standalone quarters from xfinlink")
    for _, r in q.iterrows():
        print(f"    {r.fiscal_period} to {r.period_end.date()}  "
              f"ocf {r.operating_cash_flow:>8,.0f}m  "
              f"capex {r.capital_expenditures:>6,.0f}m  "
              f"fcf {r.free_cash_flow:>8,.0f}m")
    print(f"    quarters sum to {q['operating_cash_flow'].sum():>9,.0f}m")
    print(f"    annual row      {a['operating_cash_flow'].iloc[0]:>9,.0f}m\n")
```

Output:

```
Apple fiscal 2025
  as filed with the SEC
    2024-09-29 to 2024-12-28   3 months     29,935m  10-Q
    2024-09-29 to 2025-03-29   6 months     53,887m  10-Q
    2024-09-29 to 2025-06-28   9 months     81,754m  10-Q
    2024-09-29 to 2025-09-27  12 months    111,482m  10-K
  standalone quarters from xfinlink
    Q1 to 2024-12-28  ocf   29,935m  capex  2,940m  fcf   26,995m
    Q2 to 2025-03-29  ocf   23,952m  capex  3,071m  fcf   20,881m
    Q3 to 2025-06-28  ocf   27,867m  capex  3,462m  fcf   24,405m
    Q4 to 2025-09-27  ocf   29,728m  capex  3,242m  fcf   26,486m
    quarters sum to   111,482m
    annual row        111,482m

Walmart fiscal 2026
  as filed with the SEC
    2025-02-01 to 2025-04-30   3 months      5,411m  10-Q
    2025-02-01 to 2025-07-31   6 months     18,352m  10-Q
    2025-02-01 to 2025-10-31   9 months     27,452m  10-Q
    2025-02-01 to 2026-01-31  12 months     41,565m  10-K
  standalone quarters from xfinlink
    Q1 to 2025-04-30  ocf    5,411m  capex  4,986m  fcf      425m
    Q2 to 2025-07-31  ocf   12,941m  capex  6,423m  fcf    6,518m
    Q3 to 2025-10-31  ocf    9,100m  capex  7,218m  fcf    1,882m
    Q4 to 2026-01-31  ocf   14,113m  capex  8,015m  fcf    6,098m
    quarters sum to    41,565m
    annual row         41,565m
```

Both reconciliations close to the dollar, and capital expenditure sits on the same row as the cash flow it is subtracted from, so free cash flow can be checked rather than trusted. Field definitions are in the [documentation](https://xfinlink.com/docs); the free tier covers a rolling one-year window and paid plans carry full history, which is set out on the [pricing page](https://xfinlink.com/pricing).

## Which Sources Return Standalone Quarters?

| Source | What the quarterly cash flow looks like | Fourth quarter |
| --- | --- | --- |
| SEC XBRL company concept API | As filed, so year-to-date durations of 3, 6, 9 and 12 months | Absent; derive it from the 10-K |
| Alpha Vantage `CASH_FLOW` | `annualReports` and `quarterlyReports`; IBM's four fiscal 2025 quarters sum to its annual figure | Present in that response |
| yfinance | `Ticker.quarterly_cashflow` returns a DataFrame; the reference gives no further detail | Check it yourself |
| xfinlink | Single-quarter values, de-cumulated at ingestion | Present, and it foots to the annual row |

Alpha Vantage deserves credit here. Its demo response for IBM returned 20 annual and 81 quarterly cash flow reports, and the four fiscal 2025 quarterly `operatingCashflow` values add up exactly to the annual figure alongside them, so those quarters are standalone rather than cumulative (verified against the Alpha Vantage demo endpoint, September 2026). The constraint is throughput rather than shape: a free Alpha Vantage key is capped at 25 API requests per day, stated on the company's own premium page as of September 2026, which is roughly twelve companies if each needs a cash flow call and a price call.

## What Breaks When the Quarters Stay Cumulative?

Walmart is the clearest illustration, because retail cash flow is violently seasonal. Its true fiscal 2026 quarters run 5,411, 12,941, 9,100 and 14,113. A pipeline that reads the year-to-date figures as quarters reports the fourth quarter as 41,565, which is 2.9 times the real number, and the first quarter correctly, which is what makes the fault so hard to see.

What fails afterwards fails quietly. Seasonality disappears, because a cumulative series rises monotonically inside every fiscal year whatever the business actually did, and quarter-on-quarter growth then measures the calendar rather than the company. Coverage ratios built on a quarter, such as dividends against operating cash flow, come out understated in the first quarter and overstated by the fourth.

None of this throws an error. The series looks smooth, which is precisely the problem: a smooth, rising, four-point series per year is what a cumulative series looks like, and it is also what a healthy company looks like.

## FAQ

**Is quarterly revenue year-to-date as well?**

No. Regulation S-X requires interim income statements for the most recent fiscal quarter alongside the year-to-date period, so a 10-Q carries revenue and net income in both shapes. Only the cash flow statement is cumulative-only. That asymmetry is also why a pipeline can look correct on the income statement and be wrong on cash flow.

**How do I check whether my quarters are already standalone?**

Sum four consecutive fiscal quarters and compare the result against the annual figure for that fiscal year. Standalone quarters foot to the annual total; cumulative ones overshoot it badly, by 2.5 times for Apple's fiscal 2025 and 2.2 times for Walmart's fiscal 2026. Run the check on a company whose fiscal year does not end in December, which will also catch calendar-alignment mistakes.

**Do fiscal quarters match calendar quarters?**

Often not. Apple's fiscal 2025 ended on 27 September 2025 and Walmart's fiscal 2026 ended on 31 January 2026, so grouping either company by calendar quarter mixes periods that the company never reported together. More on that in [comparing companies with different fiscal year ends](https://xfinlink.com/blog/comparing-different-fiscal-year-ends).

## Where to Go Next

The check worth building into any pipeline is the reconciliation above: four quarters against the annual row, per company, per fiscal year. It costs one extra call and it catches cumulative quarters, a missed fiscal-year reset, and a mislabelled fourth quarter in the same test. xfinlink returns both period types from the same endpoint with the same field names, so the comparison is two lines. Related reading: [where to get free cash flow data](https://xfinlink.com/blog/where-to-get-free-cash-flow-data-python) and [annual vs quarterly financial data](https://xfinlink.com/blog/annual-vs-quarterly-financial-data).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
