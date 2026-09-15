**How Much of the S&P 500's Margin Expansion Is Index Turnover? Shift-Share Decomposition in Python**

September 15, 2026 · PROFITABILITY-ANALYSIS

**What's the question?**

The S&P 500 keeps a far larger share of every sales dollar than it did ten years ago. The explanations usually involve software replacing labour, pricing power that outlasted the inflation shock, and a decade of cheap debt. All of them describe companies becoming more profitable.

A second mechanism has nothing to do with company performance. The index is a maintained list, and around twenty members are replaced each year. The replacements are not random: a company leaves after its market value collapses or it is acquired, and joins after several profitable years have made it large enough to qualify. An index that removes struggling businesses and admits successful ones reports a rising aggregate margin even if nothing inside it improves.

Both mechanisms produce the same headline number, so the headline number cannot separate them. A shift-share decomposition can. It splits the change in an aggregate into a within component, produced by the units inside it changing, and a composition component, produced by the mix changing. Aggregate net margin here is the sum of member net income divided by the sum of member revenue, which is how index margins are quoted.

**The approach**

1. Take the point-in-time S&P 500 roster at each year end from 2014 to 2025, carrying every member by its permanent entity id, so a reassigned ticker cannot substitute one business for another.
2. Pull annual filings for the union of those twelve rosters, 720 distinct entities, keeping revenue and net income.
3. Assign each report to a calendar year: a fiscal year ending in June or later belongs to that year, one ending January to May to the year before.
4. Compute each year's aggregate margin across that year's roster members.
5. Split each year's change. The within component is the margin change of companies present in both this year's roster and last year's; the composition component is the remainder, contributed by additions and removals.
6. Chain the within components to build the margin path the index would have shown had its membership never changed.

Two cross-checks run on the same panel: cumulative growth of index revenue and net income against chains that only compare a company with itself, and the decomposition repeated without Financials and Real Estate.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2014, 2026))
rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"]) for y in YEARS}
ids = sorted(set().union(*rosters.values()))

raw = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                                  start="2013-06-01", end="2026-06-30",
                                  fields=["revenue", "net_income"])
                 for i in range(0, len(ids), 100)], ignore_index=True)

df = raw.copy()
df["year"] = np.where(df["period_end"].dt.month >= 6,
                      df["period_end"].dt.year, df["period_end"].dt.year - 1)
df = df[df["year"].between(2014, 2025)].sort_values(["entity_id", "year", "period_end"])
df = df.drop_duplicates(["entity_id", "year"], keep="last")
df = df[df["revenue"].notna() & (df["revenue"] > 0) & df["net_income"].notna()]

rev = df.pivot(index="entity_id", columns="year", values="revenue")
inc = df.pivot(index="entity_id", columns="year", values="net_income")
memb = {y: sorted(i for i in rosters[y] if i in rev.index and not np.isnan(rev.at[i, y]))
        for y in YEARS}
lvl = {y: 100 * inc.loc[memb[y], y].sum() / rev.loc[memb[y], y].sum() for y in YEARS}

for y in YEARS[1:]:
    stay = [i for i in memb[y] if i in memb[y - 1]]
    within = 100 * (inc.loc[stay, y].sum() / rev.loc[stay, y].sum()
                    - inc.loc[stay, y - 1].sum() / rev.loc[stay, y - 1].sum())
    print(y, round(lvl[y] - lvl[y - 1], 2), round(within, 2),
          round(lvl[y] - lvl[y - 1] - within, 2))
```

Full script with formatting and visualisation: [sp500-margin-expansion-index-turnover-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/sp500-margin-expansion-index-turnover-python.py)

**Output**

![S&P 500 aggregate net profit margin from 2014 to 2025 against the path produced by continuing members only, with the yearly contribution of roster change below](/blog-images/sp500-margin-expansion-index-turnover-python.png)

```
S&P 500 year-end rosters 2014-2025: 720 distinct entities, annual rows retrieved: 8,279
Company-years with a usable revenue and net income line: 7,526

Year   Members   Revenue $bn   Net income $bn   Net margin %   Same-member path %
2014       474         10,555            940          8.91                8.91
2015       475         10,083            864          8.57                8.49
2016       484         10,289            924          8.98                8.87
2017       487         10,937          1,020          9.33                9.16
2018       490         11,662          1,147          9.83                9.54
2019       494         12,127          1,247         10.28               10.03
2020       494         11,889            866          7.29                6.85
2021       495         13,848          1,800         13.00               12.43
2022       498         15,503          1,584         10.22                9.69
2023       500         16,019          1,761         11.00               10.44
2024       500         17,006          1,908         11.22               10.63
2025       491         17,912          2,157         12.04               11.19

Year   Continuing members   Change in margin (pp)   Continuing (pp)   Roster change (pp)
2015              446                       -0.34             -0.41                 0.08
2016              454                        0.41              0.37                 0.03
2017              460                        0.35              0.29                 0.06
2018              466                        0.50              0.38                 0.12
2019              472                        0.45              0.49                -0.04
2020              477                       -2.99             -3.19                 0.19
2021              476                        5.71              5.58                 0.13
2022              481                       -2.78             -2.74                -0.04
2023              484                        0.78              0.75                 0.02
2024              484                        0.22              0.19                 0.04
2025              475                        0.83              0.56                 0.27
Total                                        3.14              2.28                 0.86

Roster change accounts for 27.3% of the margin expansion; its yearly contribution is positive in 9 of 11 years

Entering the index (222 entries): net margin  9.84%, mean revenue $8,977m
Leaving the index  (193 exits):   net margin  4.03%, mean revenue $9,141m

Cumulative 2014 to 2025          Index aggregate   Same-member chain
  revenue                                69.7%            70.4%
  net income                            129.5%           111.2%

Excluding Financials and Real Estate (566 entities): margin 8.07% to 10.74%, total +2.67pp = continuing +1.77pp + roster change +0.90pp (33.6%)

Chart saved to sp500-margin-expansion-index-turnover-python.png
```

**What this tells us**

The aggregate margin rose from 8.91% in 2014 to 12.04% in 2025, a gain of 3.14 percentage points, of which continuing members produced 2.28 and roster change produced 0.86. The company-level mechanism is larger, but 27.3% of the expansion came from editing the list. That composition term is tiny in any single year and almost never negative: between plus 0.02 and plus 0.27 points across the nine positive years, and minus 0.04 in each of the two negative ones. Drift that small is invisible on an annual chart and hard to ignore once it compounds for a decade.

The entry and exit figures show the mechanism directly. The 222 company-years that entered the index carried an aggregate net margin of 9.84%; the 193 that left carried 4.03%. Mean revenue was almost identical on both sides, $8,977m entering against $9,141m leaving, so this is not large companies displacing small ones. It is profitable businesses displacing unprofitable ones at similar scale, which is what an index built on market value and an earnings requirement does by construction.

The growth chains say the same thing from the other side. Index revenue grew 69.7% over the eleven years against 70.4% for companies compared only with themselves, while index net income grew 129.5% against 111.2%. Roster change added almost nothing to sales and 18.3 points to profit.

Removing Financials and Real Estate does not weaken the result: on the remaining 566 entities the margin rose 2.67 points, of which roster change supplied 0.90, a larger share at 33.6%.

**So what?**

Index-level margin, and any multiple built on index-level earnings, measures two things at once. Treat a long-run change in either as evidence about corporate profitability only after removing the composition term, which takes one extra line: recompute the aggregate on the companies present in both periods and difference the two.

The consequence falls hardest on anyone forecasting index earnings. A model fitted to the reported aggregate has absorbed roughly 0.08 points of annual margin drift that no company generates and no management team can be asked about. Projecting it forward assumes the index keeps upgrading its roster at the same rate, which is an assumption about index maintenance rather than about business conditions.

The same correction applies to aggregate return on equity, aggregate leverage, or the share of index earnings coming from one sector. Each is a mix statistic. Before attributing a decade of movement to companies, hold the membership fixed and see how much survives.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
