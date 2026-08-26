**Comparing Companies With Different Fiscal Year Ends**

August 26, 2026 · GUIDES

Align on `period_end`, the calendar date the accounts actually closed, and never on the fiscal year label. A quarter of the S&P 500 closes its books somewhere other than December, and a label like "fiscal 2026" is a naming convention each company chooses for itself. Walmart and Microsoft both call their most recent year fiscal 2026, and those two years share seven months. Group on the label and you are comparing periods that barely overlap; group on `period_end` and you are comparing the same stretch of economic time.

**Why do fiscal year ends differ at all?**

A fiscal year is supposed to end when the business is quiet, so that counting inventory and closing the books does not collide with the busiest weeks of the year. Retailers therefore close in late January or early February, after the holiday season and the returns that follow it. Technology companies are spread across the calendar for their own reasons: Microsoft closes in June, Apple in September, and Nvidia and Salesforce in late January alongside the retailers.

The result across the current index:

```
S&P 500 companies: 504
December year end: 379
Other month:       125 (25%)

Most common non-December year ends
  January    23
  June       22
  September  22
  October    10
```

125 companies is too many to treat as edge cases, and they are not confined to one corner of the market. Information Technology supplies 42 of them, more than any other sector, and the 28 companies closing in January or February split across consumer discretionary, technology, staples and industrials. Any screen covering the whole index will include them.

**What breaks when you compare on the fiscal year label?**

The label answers a question about a company's own calendar, not about the world's. Six companies, all reporting their latest completed year:

```
The same label covers different twelve months
ticker   fiscal_year    period_end    twelve months to
WMT             2026    2026-01-31 Feb 2025 - Jan 2026
HD              2026    2026-02-01 Feb 2025 - Feb 2026
NKE             2026    2026-05-31 Jun 2025 - May 2026
MSFT            2026    2026-06-30 Jul 2025 - Jun 2026
COST            2025    2025-08-31 Sep 2024 - Aug 2025
AAPL            2025    2025-09-27 Sep 2024 - Sep 2025

Walmart and Microsoft both label it fiscal 2026. The two windows share 7 months.
```

Two things go wrong here. Walmart's fiscal 2026 and Microsoft's fiscal 2026 describe periods offset by five months, so a "fiscal 2026 revenue growth" table silently mixes them. And Costco's most recent year carries a 2025 label while overlapping Walmart's 2026 window by seven months, against five months of overlap with the Walmart year labelled 2025. The labels sort the two companies into buckets that misstate which periods actually line up.

The damage compounds when the comparison is about something macroeconomic. A study of how tariffs, wage inflation or interest rates hit margins needs every company measured over the same months. Grouping on the label spreads the sample across period ends that can sit eleven months apart, which blurs whatever the study was trying to see.

**How do you align them?**

Two approaches, depending on what the analysis needs.

**Trailing alignment** is the right default for a cross-section. Pick a cut date, then take each company's most recent annual filing whose `period_end` falls on or before it. Every company contributes its latest complete year as of that date, which is what an investor standing on that date could actually have known.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ids = xfl.index("sp500")["entity_id"].dropna().astype(int).tolist()
f = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                                start="2024-09-01", fields=["revenue"])
               for i in range(0, len(ids), 100)], ignore_index=True)

CUT = "2026-06-30"
aligned = f[pd.to_datetime(f["period_end"]) <= CUT]
aligned = aligned.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)

lag = (pd.to_datetime(CUT) - pd.to_datetime(aligned["period_end"])).dt.days
print(len(aligned), lag.median(), lag.quantile(.9))
```

```
Aligned on period_end at 2026-06-30: 504 companies
  staleness in days: median 181, 90th percentile 245
```

Every company in the index contributes a row, and the median one is reporting on a year that closed six months before the cut date. That staleness is unavoidable with annual data and it is worth printing, because it tells you how much of the cut date's world the sample can possibly reflect.

**Window alignment** is stricter and suits event studies or macro work. Keep only companies whose `period_end` falls inside a narrow band, for example the three months around a target date, and accept the smaller sample. Comparing 379 December filers to each other is often better than comparing 504 companies across eleven months of different trading conditions.

For quarterly work the same rule applies with a shorter cadence. See [Annual vs Quarterly Financial Data](/blog/annual-vs-quarterly-financial-data) for which period type suits which question.

**When is calendar alignment still not enough?**

Aligning the dates does not make two businesses comparable on its own.

Seasonality survives the alignment. A retailer's year ending in January contains one holiday season and a technology company's year ending in June contains none, so a margin comparison between them is comparing different mixes of trading conditions no matter how the dates line up. Compare within a fiscal-calendar cohort where the question is seasonal.

52- and 53-week years are a second wrinkle. Many retailers and food companies report years of 52 weeks, then occasionally 53, which adds roughly 2 percent to revenue in the longer year for no operating reason. Check the gap between consecutive `period_end` dates before reading a growth rate.

Backfilled index membership is a third. If the universe comes from today's index roster, the companies that left the index are missing from every historical period, which is a separate and larger problem than fiscal alignment. [What Is Survivorship Bias in Backtesting?](/blog/what-is-survivorship-bias-in-backtesting) covers the fix.

**Frequently asked questions**

**Which field should the join key be?**
`period_end`. It is a real date on the filing and it means the same thing at every company. The `fiscal_year` field is useful for pulling one company's own history in order, and both are available as filters on the [fundamentals endpoint](/docs).

**Does a January year end belong to the year that just closed or the year it starts?**
For any analysis of economic conditions, the year that just closed. Walmart's year ending 31 January 2026 is mostly 2025 trading. One workable rule: treat any close before June as belonging to the previous calendar year. That puts the January and February retail closes where they belong and leaves December filers untouched.

**How many S&P 500 companies does this actually affect?**
125 of 504, or one in four, as of August 2026. Fiscal calendars rarely change, so the count moves slowly.

**Is there a way to avoid the problem entirely?**
Only by restricting the universe to December filers, which throws away a quarter of the index and skews the sample away from retail and parts of technology. Aligning on `period_end` costs one line of code and keeps everything.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
