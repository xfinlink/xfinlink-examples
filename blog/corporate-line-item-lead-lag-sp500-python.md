**Which Corporate Line Item Turns First? Lead-Lag Analysis of S&P 500 Fundamentals in Python**

September 17, 2026 · MACRO-RESEARCH

**What's the question?**

Every quarter, several hundred large companies report what they sold, what it cost to produce, what sits in the warehouse, what customers still owe, and what was spent on plant and equipment. That is the raw material the official macro series are built from, at company grain.

A great deal of commentary treats the inventory cycle as a leading indicator, on the reasoning that companies build stock ahead of demand and run it down ahead of a slowdown. If that is right, aggregate inventory growth is an early-warning gauge. If inventory moves after revenue instead, it confirms something already visible, and it arrives with a filing delay on top. The cross-correlation function settles the matter: it correlates two series across a range of time shifts and reports where the correlation is strongest. A peak at a positive shift means the line item moved first.

**The approach**

1. Pull quarterly figures for the current S&P 500 roster on revenue, cost of sales, selling and administrative expense, inventory, receivables, payables, and capital spending.
2. Companies close their books on their own calendars, so assign each report to the calendar quarter whose end date it sits closest to, one report per company per quarter.
3. Hold the sample constant: only companies with an unbroken record on all seven items across the 66 quarters from 2010Q1 to 2026Q2 stay in. Requiring inventory and cost of sales keeps firms that make and move physical goods, and leaves out banks and insurers, which report neither line.
4. Convert each company's series to growth against the same quarter a year earlier, then take the cross-sectional median. A dollar sum tracks the largest two or three companies rather than the typical one, and the mean of a growth ratio blows up when a denominator lands near zero.
5. Correlate each line item against revenue at shifts of minus four to plus four quarters, then repeat with 2020Q1 through 2021Q4 removed, since a shock that large can set the answer by itself.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SERIES = ["revenue", "cost_of_sales", "selling_general_admin", "inventory",
          "accounts_receivable", "accounts_payable", "capital_expenditures"]
FIELDS = ["revenue", "cost_of_revenue", "cost_of_goods_sold", "selling_general_admin",
          "inventory", "accounts_receivable", "accounts_payable", "capital_expenditures"]
QS = pd.period_range("2010Q1", "2026Q2", freq="Q")

tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique())
raw = pd.concat([xfl.fundamentals(tickers[i:i + 100], start="2008-06-01",
                                  end="2026-09-17", period_type="quarterly",
                                  fields=FIELDS, max_rows=200000)
                 for i in range(0, len(tickers), 100)], ignore_index=True)

raw["quarter"] = (raw["period_end"] + pd.Timedelta(days=45)).dt.to_period("Q") - 1
raw["cost_of_sales"] = raw["cost_of_revenue"].fillna(raw["cost_of_goods_sold"])
raw = (raw.sort_values(["ticker", "quarter", "period_end"])
          .drop_duplicates(["ticker", "quarter"], keep="last"))

names = sorted(raw["ticker"].dropna().unique())
wide = {f: raw.pivot_table(index="ticker", columns="quarter", values=f)
             .reindex(index=names, columns=QS) for f in SERIES}
keep = np.logical_and.reduce([wide[f].notna().all(axis=1).values for f in SERIES])
panel = pd.Index(names)[keep]

med = pd.DataFrame({f: (wide[f].loc[panel].pct_change(4, axis=1) * 100).median(axis=0)
                    for f in SERIES}).dropna()

rev = med["revenue"]
profile = pd.DataFrame({f: {L: med[f].shift(L).corr(rev) for L in range(-4, 5)}
                        for f in SERIES if f != "revenue"}).T
print(profile.round(2))
```

Full script with formatting and visualisation: [corporate-line-item-lead-lag-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/macro-research/corporate-line-item-lead-lag-sp500-python.py)

**Output**

![Median year-on-year growth in revenue, inventory and capital spending for 102 S&P 500 companies since 2011, with a cross-correlation table showing where each line item's correlation with revenue growth peaks](/blog-images/corporate-line-item-lead-lag-sp500-python.png)

```
Panel: 102 companies, unbroken quarterly records 2010Q1 to 2026Q2
Sectors: Health Care 26, Industrials 23, Information Technology 21,
         Consumer Discretionary 12, Consumer Staples 11, Materials 8, Real Estate 1
Growth series: 62 quarters, 2011Q1 to 2026Q2

Positive lag = turns before revenue. Negative lag = turns after revenue.
       Line item  Peak lag  Corr at peak  Corr same qtr  Peak ex-2020/21  Latest YoY %
       Inventory        -2         0.674          0.560                0           6.7
Capital spending        -1         0.786          0.697                0           8.8
    SG&A expense         0         0.835          0.835                0           8.4
   Cost of sales         0         0.919          0.919                0           7.3
     Receivables         0         0.854          0.854                0           8.6
        Payables         0         0.863          0.863                0           9.9

Cross-correlation against revenue growth, by lag in quarters:
                         -4    -3    -2    -1     0     1     2     3     4
cost_of_sales         -0.06  0.28  0.51  0.70  0.92  0.59  0.36  0.09 -0.19
selling_general_admin -0.07  0.21  0.46  0.64  0.84  0.53  0.28  0.04 -0.19
inventory              0.50  0.64  0.67  0.64  0.56  0.44  0.23  0.06 -0.14
accounts_receivable    0.05  0.32  0.44  0.67  0.85  0.63  0.40  0.06 -0.24
accounts_payable      -0.07  0.25  0.47  0.73  0.86  0.68  0.50  0.23 -0.03
capital_expenditures   0.28  0.53  0.71  0.79  0.70  0.37  0.09 -0.22 -0.44

Same profile excluding 2020Q1-2021Q4:
                         -4    -3    -2    -1     0     1     2     3     4
cost_of_sales         -0.22  0.11  0.49  0.71  0.86  0.64  0.36 -0.00 -0.28
selling_general_admin -0.10  0.06  0.29  0.53  0.78  0.68  0.52  0.28  0.03
inventory             -0.00  0.23  0.42  0.58  0.66  0.51  0.31  0.05 -0.19
accounts_receivable   -0.21  0.03  0.32  0.60  0.86  0.83  0.61  0.34  0.05
accounts_payable      -0.33 -0.05  0.27  0.58  0.82  0.72  0.55  0.31  0.04
capital_expenditures   0.18  0.42  0.59  0.68  0.69  0.42  0.25 -0.04 -0.24

Turning points (median year-on-year growth):
  Revenue           cycle peak 2021Q2 at  26.5%   trough 2024Q1 at   2.3%
  Inventory         cycle peak 2022Q1 at  20.8%   trough 2024Q1 at  -1.4%
  Capital spending  cycle peak 2021Q3 at  25.1%   trough 2024Q3 at  -3.1%

Latest quarter 2026Q2: Revenue +8.5%, Cost of sales +7.3%, SG&A expense +8.4%,
Inventory +6.7%, Receivables +8.6%, Payables +9.9%, Capital spending +8.8%
```

**What this tells us**

Not one of the six line items has its peak correlation at a positive shift. Nothing in this panel leads revenue, in the full sample or with the pandemic years removed.

Four of the six peak in the same quarter as revenue. Cost of sales at 0.92 is close to an accounting identity, and receivables at 0.85 and payables at 0.86 are tied to the quarter's billing; those four lines restate revenue in different units and carry no timing information. Inventory and capital spending do separate themselves, in the wrong direction for a forecaster: inventory peaks two quarters after revenue at 0.674 against 0.560 in the same quarter, and capital spending peaks one quarter after at 0.786.

The 2021 and 2022 episode shows the mechanism at unmissable scale. Median revenue growth topped out at 26.5% in 2021Q2 and had fallen to 6.8% by 2022Q2, while inventory growth did not peak until 2022Q1, at 20.8%, and was still running at 18.9% in 2022Q4 against revenue growth of 5.3%. Goods ordered during the boom kept arriving into a slowdown that had already started, and the overhang shows up as a lag.

The lag is not symmetric. Both series troughed together in 2024Q1, revenue at 2.3% and inventory at minus 1.4%, because a buffer is filled deliberately and drained involuntarily. Removing 2020 and 2021 moves every peak to zero, inventory included, where the profile reads 0.66 contemporaneously against 0.58 one quarter later. The measured length of the lag depends on the pandemic shock; the sign does not, since no peak moves to a positive shift in either sample.

**So what?**

Aggregate inventory growth should not be used as a recession signal. By the time it turns, revenue has been turning for two or three quarters and the filings reporting it are another six weeks old. The series is good for sizing a correction already under way: the gap between revenue growth and inventory growth measures how much stock has still to be worked off, and in 2022Q4 that gap stood at 13.6 percentage points.

The forecastable direction runs the other way. Revenue growth predicts inventory and capital spending one to two quarters ahead, which is usable for anyone modelling supplier volumes, freight demand, or capital goods orders. Revenue growth has climbed from 2.3% at the 2024Q1 trough to 8.5% in 2026Q2, with inventory at 6.7%, so the lag structure implies another quarter or two of catching up before the two meet.

For a nowcast built on filings, treat the income statement and the balance sheet as a confirmation layer with unusual breadth, not as an early signal. Anything that genuinely leads has to come from outside those lines: orders, bookings, deferred revenue, prices. Before trusting a claim that a fundamental series leads the cycle, run the nine-column correlation on it. The answer costs one line of code and is frequently the opposite of the claim.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
