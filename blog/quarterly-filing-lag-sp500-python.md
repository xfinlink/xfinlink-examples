# How Long After Quarter End Do Financials Become Public? Filing Lag Analysis in Python

September 13, 2026 · FUNDAMENTAL-ANALYSIS

**What's the question?**

A quarter ends on 30 September. The numbers describing it do not exist yet. Accountants close the books, the audit committee reviews them, lawyers argue over the wording, and only then does the Form 10-Q reach the SEC's public record. Anyone who aligns a fundamental figure to its period end has handed the strategy information that nobody could act on at the time.

That is look-ahead bias, and unlike most modelling errors it has an exact size: the number of days between the period end and the filing. Academic work sidesteps it with a blanket lag, often six months for annual accounts, which is safe and expensive: a signal built on figures half a year old has already decayed.

So the practical question is how long the wait really is. The SEC sets the outer bound: a large accelerated filer must submit Form 10-Q within 40 calendar days of the quarter closing. The floor is set by the company itself.

**The approach**

1. Take the current S&P 500 roster and carry every member by its permanent entity id, so a ticker reassigned to another company cannot substitute one business for another.
2. Pull quarterly statements for fiscal quarters ending between 1 July 2024 and 30 June 2026, keeping each row's period end and filing date.
3. Keep the first three fiscal quarters. A fourth fiscal quarter carries no Form 10-Q of its own; those figures arrive inside the annual report, which runs on a different deadline.
4. Measure the lag in calendar days as filing date minus period end, and keep lags of 1 to 90 days. A quarter also reappears as a comparative column in later filings, so a document landing more than 90 days after the period end is a restatement of the quarter rather than its first report.
5. Require each company to contribute at least four of the six fiscal quarters the window holds. Reporting speed is a habit read across several quarters, and one or two describe it too loosely to rank a company or weigh it against its sector.

That leaves 2,781 company-quarters across 482 companies. As a check, 30 rows drawn at random were compared against the SEC's own filing index: all 30 matched to the day.

**Code**

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

roster = xfl.index("sp500")
entity_ids = sorted({int(e) for e in roster["entity_id"].dropna()})

frames = []
for i in range(0, len(entity_ids), 50):
    frames.append(xfl.fundamentals(entity_id=entity_ids[i:i + 50], period_type="quarterly",
                                   start="2024-07-01", end="2026-09-13", fields=["revenue"]))
raw = pd.concat(frames, ignore_index=True)

df = raw[(raw["source"] == "filing") & raw["fiscal_period"].isin(["Q1", "Q2", "Q3"])].copy()
df = df[(df["period_end"] >= "2024-07-01") & (df["period_end"] <= "2026-06-30")]
df["lag"] = (df["filing_date"] - df["period_end"]).dt.days
df = df[(df["lag"] > 0) & (df["lag"] <= 90)]
df = df[df.groupby("entity_id")["period_end"].transform("size") >= 4]

print(df["lag"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
for d in (20, 25, 30, 35, 40, 45):
    print(d, round(100 * (df["lag"] <= d).mean(), 1))
```

Full script with formatting and visualisation: [quarterly-filing-lag-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/quarterly-filing-lag-sp500-python.py)

**Output**

```
S&P 500 Form 10-Q filing lag | fiscal Q1-Q3 ending 2024-07-01 to 2026-06-30
Companies: 482   Company-quarters: 2,781

Days from fiscal quarter end to filing
  minimum        8
  5th pct       22
  median        31
  75th pct      36
  90th pct      38
  95th pct      39
  99th pct      41
  maximum       56

Share of quarters on file by day N
  day 20     3.2%
  day 25    19.6%
  day 30    40.6%
  day 35    68.7%
  day 40    98.4%
  day 45    99.9%
  day 50   100.0%

Filed in the last five days before the 40-day deadline (36-40): 29.7%
Filed after day 40: 44 quarters (1.6%) from 31 companies
  tail by day: 41d x22  42d x12  43d x7  45d x1  49d x1  56d x1

Fastest five companies (median days)
  DAL    DELTA AIR LINES INC              9.5
  ORCL   ORACLE CORP                     10.5
  OMC    OMNICOM GROUP INC               16.0
  UAL    U A L CORP                      16.0
  FAST   FASTENAL CO                     16.0

Slowest five companies
  SMCI   SUPER MICRO COMPUTER INC        41.0
  SPG    SIMON PROPERTY GROUP INC        40.0
  RL     RALPH LAUREN CORP               40.0
  TPR    TAPESTRY INC                    40.0
  CPAY   CORPAY INC                      40.0

Median company lag by sector
  Industrials               29.0  (78 companies)
  Information Technology    31.0  (72 companies)
  Real Estate               31.0  (31 companies)
  Consumer Discretionary    31.2  (42 companies)
  Consumer Staples          32.0  (33 companies)
  Financials                32.5  (74 companies)
  Communication Services    33.0  (19 companies)
  Health Care               34.0  (57 companies)
  Utilities                 34.5  (31 companies)
  Energy                    35.2  (22 companies)
  Materials                 35.5  (23 companies)
```

**What this tells us**

The median quarter reaches the public record 31 days after it ends, with 40.6% of filings in by day 30 and 68.7% by day 35. Three weeks after a quarter closes only 3.2% of the index has filed, so a screen run on 21 October is still reading June figures for 97 companies in 100.

The deadline does most of the shaping. Filings pile up against it: 29.7% of quarters arrive in the final five days of the window, and by day 40 the share on file jumps to 98.4%. Past that point the distribution nearly stops. Of the 44 quarters filed later, 41 land at 41, 42 or 43 days, almost all of them cases where the 40th day fell on a weekend or a closed federal calendar and the filing rolled to the next business day. Three sit further out: Sandisk at 45 days, Archer-Daniels-Midland at 49 and Super Micro Computer at 56. The last two were each preceded by a Form NT 10-Q, the notice a company files when its accounts will not be ready in time.

Speed is a company habit rather than a sector one. Delta Air Lines takes 9.5 days and files the 10-Q on the same day it issues the earnings release; Oracle takes 10.5. Sector medians cover a narrow range, 29.0 days for Industrials to 35.5 for Materials, while the gap between the fastest and slowest individual companies is over 30 days.

**So what?**

Use the filing date, not the period end. Every fundamental row carries one, and joining a price series on `filing_date` removes the bias at its source without discarding a single observation. This matters most for event studies and quarterly rebalances, where a one-month shift moves the entry point past a full earnings reaction.

If a single fixed lag is easier to implement, 45 calendar days covers 99.9% of quarterly filings and 40 days covers 98.4%. A six-month convention borrowed from annual-data research throws away five months of freshness for no gain.

Treat the tail as a signal of its own. Slipping past the deadline is rare enough to be informative, and the two companies at the far end of this distribution had told the regulator in advance that their books were not closed.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
