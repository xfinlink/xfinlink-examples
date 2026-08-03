# How Much of the Nasdaq 100 Is Already in the S&P 500? Index Overlap Analysis in Python

August 3, 2026 · INDEX-UNIVERSE

## What's the question?

A common allocation pairs an S&P 500 fund as the core with a Nasdaq 100 fund on top, and the second position is usually described as diversification, on the reasoning that two indices built under different rules produce two different baskets.

The rules are different. The S&P 500 draws from US-domiciled companies across every sector, applies an earnings test, and is picked by a committee. The Nasdaq 100 ranks the largest non-financial Nasdaq-listed companies, sets no profitability requirement, and admits foreign issuers. Different rules do not guarantee different holdings.

So the question is how much of the second fund an investor already owns through the first. Counting the shared members gives one answer; the share of Nasdaq 100 market value those members carry gives another, and the second decides portfolio outcomes, since exposure is weight rather than presence on a list.

## The approach

Membership is read at 31 December, as the roster stood on that date. Backdating today's list would answer a different question: only 32 of the 100 companies in the Nasdaq 100 at the end of 2005 were still members nineteen years later, and fewer than half of that year's S&P 500 remained.

1. Pull Nasdaq 100 and S&P 500 membership at each year end, keyed on entity identifiers rather than tickers. A ticker is a lease rather than a name: Q labelled Qwest Communications in the 2005 roster and now labels Qnity Electronics, spun out of DuPont in late 2025.
2. Intersect the two sets of identifiers for each year. Share classes collapse to one company under this keying, so a company with two listed classes counts once.
3. At the final year end, convert each Nasdaq 100 member's closing market capitalisation into an index weight, then divide that weight between shared members and members the Nasdaq 100 holds on its own.

The panel covers the sixteen year ends between 2005 and 2024 at which the point-in-time roster resolves to exactly one hundred distinct companies, holding the denominator fixed across years.

## Code

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

def members(index_name, as_of):
    df = xfl.index(index_name, as_of=as_of)
    return set(int(x) for x in df["entity_id"].dropna())

rows = []
for year in range(2005, 2025):
    as_of = f"{year}-12-31"
    ndx, spx = members("ndx100", as_of), members("sp500", as_of)
    if len(ndx) == 100:
        rows.append({"year": year, "ndx": len(ndx), "both": len(ndx & spx)})
panel = pd.DataFrame(rows)

FINAL = "2024-12-31"
ndx_final = sorted(members("ndx100", FINAL))
spx_final = members("sp500", FINAL)

cap = xfl.metrics(entity_id=ndx_final, period_type="daily", fields=["market_cap"],
                  start=FINAL, end=FINAL, max_rows=100000)
cap["weight"] = 100 * cap["market_cap"] / cap["market_cap"].sum()
cap["in_sp500"] = cap["entity_id"].isin(spx_final)

print(panel.to_string(index=False))
print(cap.groupby("in_sp500")["weight"].sum())
```

Full script with formatting and visualisation: [nasdaq-100-sp500-overlap-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/nasdaq-100-sp500-overlap-python.py)

## Output

![Two panel chart: the count of Nasdaq 100 members also in the S&P 500 rises from 51 in 2005 to 84 in 2024, and the 16 members held outside the S&P 500 at the end of 2024 carry between 0.06 and 1.01 percent of Nasdaq 100 market value](/blog-images/nasdaq-100-sp500-overlap-python.png)

```
Nasdaq 100 members that are also S&P 500 members, point-in-time rosters
year    NDX members  in S&P 500  outside
2005            100          51       49
2006            100          53       47
2007            100          54       46
2008            100          59       41
2009            100          64       36
2010            100          69       31
2011            100          67       33
2012            100          71       29
2014            100          73       27
2015            100          74       26
2018            100          83       17
2020            100          76       24
2021            100          75       25
2022            100          79       21
2023            100          83       17
2024            100          84       16

Market-cap weight inside the Nasdaq 100 at 2024-12-31
  84 shared members   94.64% of index weight
  16 members held only by the Nasdaq 100   5.36% of index weight
  aggregate market value  $27.02tn

Nasdaq 100 members outside the S&P 500 at 2024-12-31
ticker  company                            market cap $bn  NDX weight
ASML    A S M L HOLDING N V                         272.6       1.01%
AZN     ASTRAZENECA PLC                             203.2       0.75%
PDD     PDD HOLDINGS INC                            135.0       0.50%
ARM     A R M HOLDINGS PLC                          130.0       0.48%
APP     APPLOVIN CORP                               110.1       0.41%
MRVL    MARVELL TECHNOLOGY INC                       95.6       0.35%
MELI    MERCADOLIBRE INC                             86.2       0.32%
MSTR    Strategy Inc                                 71.2       0.26%
DASH    DOORDASH INC                                 70.4       0.26%
TEAM    ATLASSIAN CORP PLC                           63.6       0.24%
TTD     TRADE DESK INC                               58.3       0.22%
DDOG    DATADOG INC                                  48.9       0.18%
CCEP    COCA COLA EUROPACIFIC PARTNERS               35.4       0.13%
ZS      ZSCALER INC                                  27.7       0.10%
GFS     GLOBALFOUNDRIES INC                          23.7       0.09%
MDB     MONGODB INC                                  17.3       0.06%

A blended portfolio: how much sits in securities the S&P 500 fund does not hold
  90% S&P 500 fund / 10% Nasdaq 100 fund    0.54%
  80% S&P 500 fund / 20% Nasdaq 100 fund    1.07%
  70% S&P 500 fund / 30% Nasdaq 100 fund    1.61%
  60% S&P 500 fund / 40% Nasdaq 100 fund    2.15%
```

## What this tells us

The two indices have converged. In 2005 roughly half the Nasdaq 100 sat inside the S&P 500; by the end of 2024, 84 of its 100 members did. The count climbed through the 2008 crisis and the decade after it as the large Nasdaq-listed technology companies grew into the S&P 500's size requirement.

By weight the convergence is close to total. Shared members carry 94.64 percent of Nasdaq 100 market value, so the 16 companies the Nasdaq 100 holds alone amount to 5.36 percent of it, or $1.45tn out of $27.02tn. The gap between 84 percent by count and 94.64 percent by weight exists because the non-shared names are small; the largest of them, ASML, is 1.01 percent of the index.

Eligibility rules explain most of that residue. Six of the 16 are foreign issuers that the S&P 500's US-domicile requirement excludes outright: ASML, AstraZeneca, PDD, Arm, Coca-Cola Europacific and GlobalFoundries. Those six are 2.96 of the 5.36 percentage points, and ASML with AstraZeneca alone account for a third of it. The rest are US companies the committee had not selected by the end of 2024, several recently listed or short of the positive earnings the S&P 500 asks for.

The setback after 2018 shows the same mechanism running in reverse. The shared count fell from 83 at the end of 2018 to 75 three years later, while the Nasdaq 100 absorbed a cohort of newly listed companies the S&P 500 could not take: Airbnb, CrowdStrike, Datadog, Lucid, Moderna, Peloton and Zoom were all in the one index and not the other. As that cohort matured, was admitted, or left, the count recovered to 84.

## So what?

Treat a Nasdaq 100 sleeve as a weighting decision rather than a holdings decision: adding it to an S&P 500 core barely changes what is owned, only how much of each thing is owned. At a 30 percent allocation, 1.61 percent of the portfolio sits in securities the S&P 500 fund does not hold, and the other 28.4 percent is a second, heavier bet on companies already in the core.

That reframes the analysis an allocator should run. The question is not whether the two funds hold different names but how far the blend concentrates the top of the book: the ten largest Nasdaq 100 members are 71.3 percent of that index and are already the largest positions in the S&P 500.

If the goal is exposure the core does not provide, the 16 names are a short list, and holding them directly gives control over their size. Rebuild the rosters point in time for any historical version of this test: today's list backdated will always make the past look more like the present.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
