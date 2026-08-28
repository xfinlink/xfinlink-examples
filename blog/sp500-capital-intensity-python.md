**Is the S&P 500 Getting More Capital Intensive? Capex Analysis in Python**

August 28, 2026 · MACRO-RESEARCH

**What's the question?**

Capital intensity is the share of revenue a company spends on physical assets: factories, warehouses, networks, data centres. Capital expenditure, or capex, is the cash figure for that spending in the cash flow statement. Dividing it by revenue gives a measure that compares a steel mill to a software company on the same scale.

For most of the past two decades the direction was clear. The US index grew more software-heavy, asset-light business models took a larger share of profits, and capital intensity fell. That trend became one of the standard arguments for why equity valuations could sustain higher multiples: companies that convert revenue into cash without building anything need less reinvestment to grow.

Data centre construction has complicated the story. The question is whether the index has actually reversed course, and if so, whether the reversal is broad or confined to a small number of very large spenders.

**The approach**

The measure is capital expenditure divided by revenue, computed two ways for each fiscal year. The aggregate ratio sums capex across all companies and divides by summed revenue, which describes the index as a single entity. The median ratio takes each company's own ratio and reports the middle one, which describes the typical member. When those two diverge, spending is concentrating.

1. Rebuild the S&P 500 roster as it stood at each year end from 2012 to 2025, and keep each company only in the years it was actually a member.
2. Pull annual revenue and capital expenditure for that universe from company filings.
3. Exclude Financials and Real Estate, where capex against revenue does not describe the business in a comparable way.
4. Compute the aggregate ratio, the median ratio, and the share of total capex taken by the ten largest spenders, for each fiscal year.
5. Repeat the comparison on a constant sample of companies present in both endpoint years, so the result does not depend on which companies the panel happens to hold.

The panel holds 5,304 company-years across 581 companies.

**Code**

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
           for y in range(2012, 2026)}
universe = sorted(set().union(*rosters.values()))

fund = xfl.fundamentals(entity_id=universe, start="2011-01-01", end="2026-08-28",
                        period_type="annual", max_rows=60000,
                        fields=["revenue", "capital_expenditures", "total_assets"])

fund = fund.drop_duplicates(subset=["entity_id", "fiscal_year"], keep="last")
fund = fund[fund["fiscal_year"].between(2012, 2025)]
fund = fund.dropna(subset=["revenue", "capital_expenditures", "gics_sector"])
fund = fund[fund["revenue"] > 0]

# keep a company only in the years it was in the index
fund = fund[[e in rosters.get(int(y), set())
             for e, y in zip(fund["entity_id"], fund["fiscal_year"])]]
panel = fund[~fund["gics_sector"].isin({"Financials", "Real Estate"})].copy()
panel["intensity"] = panel["capital_expenditures"] / panel["revenue"]

def summarise(g):
    total = g["capital_expenditures"].sum()
    return pd.Series({
        "aggregate_pct": total / g["revenue"].sum() * 100,
        "median_pct": g["intensity"].median() * 100,
        "top10_share_pct": g.nlargest(10, "capital_expenditures")
                            ["capital_expenditures"].sum() / total * 100})

print(panel.groupby("fiscal_year").apply(summarise, include_groups=False))
```

Full script with formatting and visualisation: [sp500-capital-intensity-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/macro-research/sp500-capital-intensity-python.py)

**Output**

![Two-panel chart of S&P 500 capital intensity from fiscal 2012 to 2025, showing the index aggregate ratio rising to 7.44% while the median company falls to 3.46%, and the top ten spenders' share of capex rising to 45%](/blog-images/sp500-capital-intensity-python.png)

```
capital expenditure as a share of revenue, S&P 500 ex Financials and Real Estate
             companies  aggregate_pct  median_pct  top10_share_pct
fiscal_year
2012             385.0           6.59        4.09            30.90
2013             385.0           6.64        4.21            31.30
2014             341.0           7.71        4.45            31.11
2015             363.0           6.32        4.24            28.04
2016             381.0           6.03        4.06            26.11
2017             379.0           5.65        3.74            26.12
2018             376.0           6.04        3.99            27.58
2019             381.0           6.05        4.12            27.75
2020             377.0           5.88        3.86            31.29
2021             392.0           5.45        3.52            33.57
2022             390.0           5.92        3.69            33.44
2023             396.0           6.18        3.82            30.19
2024             391.0           6.58        3.81            35.75
2025             367.0           7.44        3.46            45.04

aggregate capital intensity by sector, change fiscal 2015 to 2025 (pp)
                        2015_pct  2025_pct  change_pp
Energy                     14.59      9.33      -5.27
Industrials                 5.47      3.83      -1.64
Health Care                 1.86      1.82      -0.04
Materials                   9.17      9.16      -0.01
Consumer Staples            2.67      3.11       0.44
Information Technology      5.28      8.06       2.78
Consumer Discretionary      4.93      8.96       4.04
Communication Services     10.64     17.56       6.92
Utilities                  24.66     37.99      13.33

ten largest capital spenders, fiscal 2025
ticker            gics_sector  capital_expenditures  revenue  pct_of_revenue
  AMZN Consumer Discretionary              131819.0 716924.0            18.4
 GOOGL Communication Services               91447.0 402836.0            22.7
  META Communication Services               69691.0 200966.0            34.7
  MSFT Information Technology               64551.0 281724.0            22.9
   XOM                 Energy               28358.0 332238.0             8.5
   WMT       Consumer Staples               23783.0 674538.0             3.5
  ORCL Information Technology               21215.0  57399.0            37.0
     T Communication Services               20842.0 125648.0            16.6
   CVX                 Energy               17347.0 184432.0             9.4
    VZ Communication Services               17011.0 138191.0            12.3

constant-sample check (companies present in both years)
  2015 vs 2025: FY2015  n=239  aggregate=6.14%  median=4.29%  top10=34.8%
  2015 vs 2025: FY2025  n=239  aggregate=7.88%  median=3.89%  top10=52.9%
  2024 vs 2025: FY2024  n=355  aggregate=6.56%  median=3.88%  top10=38.7%
  2024 vs 2025: FY2025  n=355  aggregate=7.47%  median=3.53%  top10=45.3%
```

**What this tells us**

Both statements are true at once, and they point in opposite directions. Measured as a single entity, the index is more capital intensive than in any year since 2014: 7.44% of revenue in fiscal 2025 against a 5.45% trough in 2021, with only fiscal 2014 higher at 7.71%. Measured by its typical member, the index is less capital intensive than at any point in the sample: the median company spent 3.46% of revenue, below the 4.45% peak in 2014 and below every year since.

The reconciling number is concentration. The ten largest spenders took 45.04% of all capital expenditure in fiscal 2025, up from 28.04% in 2015 and from 35.75% only a year earlier. Nearly half the index's capital budget now sits with ten companies out of roughly 370.

The identity of those ten explains the shift. In fiscal 2015 the largest spenders were oil producers and telecom carriers, with Chevron and Exxon at the top. In fiscal 2025 the four largest are Amazon, Alphabet, Meta and Microsoft, spending 131.8, 91.4, 69.7 and 64.6 billion dollars respectively, with Oracle in seventh place committing 37.0% of its revenue. These were the companies that made the asset-light argument in the first place. Meta at 34.7% of revenue and Oracle at 37.0% are now more capital intensive than Exxon at 8.5%.

The sector table shows the reversal is narrow. Utilities rose 13.33 points, which reflects grid and generation investment against rising load. Communication Services rose 6.92 points and Consumer Discretionary 4.04, both driven by the cloud businesses inside them rather than by the sector at large. Energy fell 5.27 points and Industrials 1.64. Health Care and Materials are unchanged to two decimal places.

The constant-sample check confirms the movement is real rather than a composition effect. Holding the panel to the 239 companies present in both 2015 and 2025, the aggregate ratio still rises from 6.14% to 7.88%, the median still falls from 4.29% to 3.89%, and the top-ten share still climbs from 34.8% to 52.9%. The same holds over the single year from 2024 to 2025 on 355 constant companies.

**So what?**

Any argument about the index that rests on its aggregate capital intensity is now an argument about five or six companies. Free cash flow margins, reinvestment rates, and return on invested capital computed at index level will move with hyperscaler construction schedules, and read as economy-wide shifts when they are not. Equal-weighted or median versions of those measures answer a different and often more useful question about the typical listed company.

For anyone modelling the index, the practical step is to compute both series and watch the gap. A widening spread between the aggregate and the median is a concentration signal that shows up in capital spending well before it shows up in earnings, because today's capex becomes tomorrow's depreciation and pressures margins on a two- to four-year lag.

For sector work, the useful cut is not the GICS label. Cloud infrastructure spending currently sits inside Communication Services, Consumer Discretionary and Information Technology, so a sector-level view splits one economic story across three buckets. Building the aggregate over a hand-picked list of companies, rather than over a classification, gets closer to what is actually being measured.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
