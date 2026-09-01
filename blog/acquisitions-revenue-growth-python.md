**How Much Revenue Does a Dollar of Acquisitions Buy? Growth Decomposition in Python**

September 1, 2026 · FUNDAMENTAL-ANALYSIS

**What's the question?**

Revenue growth is the headline number in almost every equity story, and the income statement does not say where it came from. A company that grew revenue 8 percent a year for a decade might have built that growth by selling more of what it already made, or it might have bought it, one acquisition at a time, using cash that could have gone to shareholders instead. The two look identical on the top line and are worth very different amounts.

The cash flow statement separates them. Cash spent on acquisitions is disclosed every year as its own line, so the total a company spent buying other companies over a decade is a matter of record. Setting that total against the revenue the company gained over the same decade gives a rate: how much annual revenue arrived per dollar spent.

The rate matters because it is the exchange the acquirer is making. A company paying a dollar for 50 cents of annual revenue is buying at two times sales, and whether that is sensible depends on the margin and durability of the revenue it bought. A company that grew without spending anything is running a different business model, and the comparison between the two is what this measures.

**The approach**

The sample is every company that sat in the S&P 500 at any year end between 2014 and 2024, addressed by entity identifier so that a rename does not split a company's history into two shorter ones. Fiscal years 2014 through 2024 give eleven annual observations.

1. Derive each row's year from its period end rather than its fiscal-year label, which keeps 52-week and January-ending filers in the right place.
2. Keep companies with at least nine annual periods and revenue above 500 million dollars in the first of them. The revenue floor matters because every ratio here divides by starting revenue, and a small denominator produces a large number that means nothing.
3. Sum cash acquisition spending across the decade, counting only the years where it was positive, so that a divestiture in one year does not cancel a purchase in another.
4. Express both spending and revenue growth as multiples of starting revenue, then fit a line through the origin. The slope is the answer: cents of additional annual revenue per dollar spent.

Nothing here establishes causation. A company that spends heavily on acquisitions may also be operating in a growing market, and the measurement cannot separate the two. What it can do is put a number on how the two travel together, and give a baseline from the companies that spent nothing at all.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ids = set()
for y in range(2014, 2025):
    ids.update(int(i) for i in xfl.index("sp500", as_of="%d-12-31" % y)["entity_id"].dropna())

fun = xfl.fundamentals(entity_id=sorted(ids), start="2013-06-30", end="2025-12-31",
                       period_type="annual", fields=["revenue", "acquisitions_net"],
                       max_rows=60000)
fun["period_end"] = pd.to_datetime(fun["period_end"])
fun["year"] = fun["period_end"].dt.year - (fun["period_end"].dt.month <= 6).astype(int)
fun = fun[(fun["year"] >= 2014) & (fun["year"] <= 2024)]

rows = []
for eid, g in fun.groupby("entity_id"):
    g = g.sort_values("year")
    first, last = g.iloc[0], g.iloc[-1]
    r0, r1 = first["revenue"], last["revenue"]
    if len(g) < 9 or pd.isna(r0) or pd.isna(r1) or r0 < 500 or r1 <= 0:
        continue
    spend = float(g["acquisitions_net"].clip(lower=0).sum())
    rows.append({"ticker": last["ticker"], "sector": last["gics_sector"],
                 "rev0": r0, "rev1": r1, "spend": spend,
                 "intensity": spend / r0, "rev_growth": (r1 - r0) / r0,
                 "cagr": (r1 / r0) ** (1 / (last["year"] - first["year"])) - 1})

res = pd.DataFrame(rows).dropna(subset=["sector"])
x, y = res["intensity"].values, res["rev_growth"].values
print((x * y).sum() / (x * x).sum(), np.corrcoef(x, y)[0, 1])
```

Full script with formatting and visualisation: [acquisitions-revenue-growth-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/acquisitions-revenue-growth-python.py)

**Output**

```
Point-in-time S&P 500 rosters at each year end 2014-2024
companies with >=9 annual periods and revenue above $500m at the start: 562
total acquisition spending in the sample: $3.74 trillion
companies spending nothing on acquisitions in the decade: 68

added revenue against acquisition spending, both as a multiple of starting revenue
  slope through the origin  0.793
  correlation               0.306
  pooled: $6.59tn of extra annual revenue against $3.74tn spent = 176 cents of annual revenue per dollar

quintiles of acquisition spending as a share of starting revenue
                        n  spend  cagr  growth
q
Q1 least acquisitive  113    0.0  1.86    20.2
Q2                    112    9.7  2.82    32.1
Q3                    112   28.8  3.82    45.5
Q4                    112   75.2  5.29    67.4
Q5 most acquisitive   113  227.6  9.46   144.0

by sector
                         n  intensity  cagr
sector
Health Care             63       83.4  7.37
Information Technology  70       82.0  7.36
Communication Services  23       41.0  3.95
Real Estate             34       40.0  5.89
Materials               29       31.3  3.25
Industrials             81       29.3  4.56
Consumer Staples        39       26.6  3.10
Financials              84       25.2  5.69
Utilities               30       18.6  1.68
Consumer Discretionary  74       10.3  4.14
Energy                  35        8.5 -0.30

the ten heaviest acquirers, spending against starting revenue
ticker                            name   rev0    rev1   spend  intensity  cagr
  AVGO                    BROADCOM INC 4269.0 51574.0 74440.0       17.4  28.3
   AMT             AMERICAN TOWER CORP 4100.0 10127.2 35183.8        8.6   9.5
   ROP          ROPER TECHNOLOGIES INC 3549.5  7039.2 25787.6        7.3   7.1
   ICE Intercontinental Exchange, Inc. 4221.0 11761.0 26581.0        6.3  10.8
  CPAY                      CORPAY INC 1199.4  3974.6  7272.5        6.1  12.7
  PANW          PALO ALTO NETWORKS INC  598.2  8027.5  3552.2        5.9  29.6
   TDG             TRANSDIGM GROUP INC 2372.9  7940.0 12624.1        5.3  12.8
  WDAY                     WORKDAY INC  787.9  8446.0  4182.5        5.3  26.8
   CRM                  SALESFORCE INC 5373.6 37895.0 28210.5        5.2  21.6
   ADI              ANALOG DEVICES INC 2864.8  9427.2 14197.4        5.0  12.6

companies that bought nothing: median revenue CAGR 2.38% (n=68)
companies in the top quintile: median revenue CAGR 9.46%
```

**What this tells us**

The relationship is monotone across all five quintiles, which is unusual for a cross-sectional sort of this kind. Median revenue growth runs 1.86, 2.82, 3.82, 5.29 and 9.46 percent a year as acquisition spending rises, and the top quintile grows more than five times as fast as the bottom. Companies that grew fast bought a lot.

Three different numbers answer the per-dollar question, and the gap between them is the more useful finding. The slope through the origin puts 79 cents of additional annual revenue against every dollar spent. The pooled figure, which divides all 6.59 trillion dollars of added revenue by all 3.74 trillion spent, gives 176 cents. The median company in the top quintile spent 2.28 times its starting revenue and grew revenue by 1.44 times, which is nearer 63 cents.

The pooled figure is the one to distrust. It credits acquisitions with every dollar of revenue growth in the sample, including the growth of companies that bought almost nothing, so it functions as a ceiling rather than an estimate. The 68 companies that spent nothing at all over the decade set the floor: they grew revenue at a median 2.38 percent a year, which is close to the general price level and is what organic growth looks like for a large American company without acquisitions. Subtracting a baseline of that size from the top quintile's growth leaves roughly half a dollar of extra annual revenue per dollar spent.

The correlation is 0.306, so acquisition spending accounts for under a tenth of the variation in growth across these companies. Spending is a real part of the story and nowhere near the whole of it.

Sector patterns follow the same line. Health care and information technology are the two most acquisitive sectors, at a median 83 and 82 percent of starting revenue spent, and they are also the two fastest growing, at 7.4 percent a year each. Energy sits at the other end with a median 8.5 percent spent and revenue that shrank slightly over the decade, which reflects the commodity price in 2024 against 2014 more than any decision about acquisitions. Broadcom is the extreme case in the sample: it spent 17.4 times its 2014 revenue and ended the decade with twelve times the revenue it started with.

**So what?**

When a company reports 8 percent revenue growth, look at cash spent on acquisitions in the same years before treating that growth as evidence of anything about the underlying business. The 68 non-acquirers here grew at 2.38 percent, so a large company growing at 8 percent without buying anything is doing something genuinely different from one growing at 8 percent while spending most of its free cash flow on purchases.

The exchange rate is the number to compute for any specific company: cumulative acquisition spending divided by the revenue added over the same window. Roughly half a dollar of annual revenue per dollar spent is the rough benchmark from this sample after allowing for organic growth, which is another way of saying the index paid something like two times sales for what it bought. A company well above that rate is either buying cheaply or buying revenue that arrives with better margins, and both are worth confirming rather than assuming.

The same measurement doubles as a forecasting check. A revenue projection that extends a historical growth rate forward assumes the acquisitions continue, since the history contains them. Separating the two before extrapolating is what stops a decade of purchases from turning into a permanent organic growth rate on a spreadsheet.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
