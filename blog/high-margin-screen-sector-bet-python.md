**Is a High-Margin Screen Just a Sector Bet? Sector-Neutral Profitability Ranking in Python**

September 11, 2026 · PROFITABILITY-ANALYSIS

**What's the question?**

A profitability screen ranks companies on a margin and keeps the top slice. The standing objection is that the ranking is not about companies at all: software houses run far wider gross margins than grocers and distributors do, so a list of the highest-margin names in an index may be a sector allocation wearing the clothes of a stock screen.

If most of the profitability difference between two companies came from the sectors they sit in, ranking on margin would be an indirect way of buying technology and health care. If most of it came from the businesses themselves, the sector labels matter far less than critics assume.

Two quantities answer it: the share of the cross-sectional spread that sits between sector averages rather than inside them, which is the one-way analysis of variance statistic eta squared, and the fraction of a screen's names that survive when the identical ranking runs inside each sector. Four measures are tested: gross margin, operating margin, net margin, and return on assets, the last being operating income over total assets.

**The approach**

The sample is every company that sat in the S&P 500 at any year end between 2013 and 2023, carried by entity id rather than by ticker, so a recycled symbol never swaps one company for another. Fiscal years 2014 through 2023 are used. Financials and Real Estate are set aside, because gross profit and profit measured against assets do not mean the same thing for a lender or a landlord. What remains is 543 companies, 4,607 company-years and 9 sectors.

1. Compute the four measures for each company-year from the annual income statement and balance sheet.
2. Winsorise each measure at the 1st and 99th percentile inside each fiscal year, so one collapsed year does not set the width of the cross-section.
3. Split the variance of each measure, in each year, into a between-sector and a within-sector part. The headline version runs on within-year percentile ranks, since a screen is a ranking exercise; the same split on raw values serves as a check.
4. Pool the top decile of every year and compare its sector mix against the sample.
5. Re-rank inside each sector, take each sector's top decile, and count how many of the original names remain.

**Code**

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

Y0, Y1 = 2014, 2023
EXCLUDE = {"Financials", "Real Estate"}
FIELDS = ["revenue", "gross_profit", "operating_income", "net_income", "total_assets"]
METRICS = ["Gross margin", "Operating margin", "Net margin", "Return on assets"]

# Every company that sat in the index at any year end in the window, so the
# cross-section is not restricted to the names that survived in it.
ids = set()
for y in range(Y0 - 1, Y1 + 1):
    ids |= set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
ids = sorted(ids)
batches = [ids[i:i + 100] for i in range(0, len(ids), 100)]


def pull(b):
    return xfl.fundamentals(entity_id=b, period_type="annual", fields=FIELDS,
                            start=f"{Y0}-01-01", end=f"{Y1 + 1}-06-30", max_rows=40000)


with ThreadPoolExecutor(max_workers=6) as ex:
    fund = pd.concat(ex.map(pull, batches), ignore_index=True)

d = fund[fund["fiscal_year"].between(Y0, Y1)].drop_duplicates(
    ["entity_id", "fiscal_year"], keep="last")
d = d[~d["gics_sector"].isin(EXCLUDE)].dropna(subset=FIELDS + ["gics_sector"])
d = d[(d["revenue"] > 0) & (d["total_assets"] > 0)].copy()

d["Gross margin"] = d["gross_profit"] / d["revenue"]
d["Operating margin"] = d["operating_income"] / d["revenue"]
d["Net margin"] = d["net_income"] / d["revenue"]
d["Return on assets"] = d["operating_income"] / d["total_assets"]
for m in METRICS:                       # trim the 1% tails inside each year
    d[m] = d.groupby("fiscal_year")[m].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))

print(f"{d['entity_id'].nunique()} companies, {len(d):,} company-years, "
      f"fiscal {Y0}-{Y1}, {d['gics_sector'].nunique()} sectors")


def eta_sq(sector, v):
    """Share of the cross-sectional variance of v that sits between sectors."""
    grand = v.mean()
    between = sum(len(x) * (x.mean() - grand) ** 2 for _, x in v.groupby(sector))
    return between / ((v - grand) ** 2).sum()


rows = []
for y, g in d.groupby("fiscal_year"):
    r = {"fiscal_year": y}
    for m in METRICS:
        r[m + " (value)"] = eta_sq(g["gics_sector"], g[m])
        r[m] = eta_sq(g["gics_sector"], g[m].rank(pct=True))
    rows.append(r)
expl = pd.DataFrame(rows).set_index("fiscal_year")

print("\nShare of cross-sectional variance explained by sector, fiscal "
      f"{Y0}-{Y1}")
print(f"{'':<18}{'ranks: mean':>12}{'min':>8}{'max':>8}{'values: mean':>14}")
for m in METRICS:
    print(f"{m:<18}{expl[m].mean():>11.1%}{expl[m].min():>8.1%}"
          f"{expl[m].max():>8.1%}{expl[m + ' (value)'].mean():>14.1%}")

base = d["gics_sector"].value_counts(normalize=True)
print(f"\nSector mix of the top decile, pooled {Y0}-{Y1} (sample share in brackets)")
for m in METRICS:
    top = d[d.groupby("fiscal_year")[m].rank(pct=True, ascending=False) <= 0.10]
    lead = top["gics_sector"].value_counts(normalize=True).head(2)
    print(f"{m:<18}" + "  ".join(
        f"{s} {v:.0%} [{base[s]:.0%}]" for s, v in lead.items())
        + f"   top two {lead.sum():.0%} vs {base[lead.index].sum():.0%}")

print("\nNames kept when the same screen is ranked inside each sector")
for m in METRICS:
    keep = []
    for y, g in d.groupby("fiscal_year"):
        raw = set(g.index[g[m].rank(pct=True, ascending=False) <= 0.10])
        neutral = set(g.index[g.groupby("gics_sector")[m].rank(
            pct=True, ascending=False) <= 0.10])
        keep.append(len(raw & neutral) / len(raw))
    print(f"{m:<18}{np.mean(keep):>7.1%}  (year range {min(keep):.0%}-{max(keep):.0%})")
```

Full script with formatting and visualisation: [high-margin-screen-sector-bet-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/high-margin-screen-sector-bet-python.py)

**Output**

![Share of the S&P 500 profitability spread explained by sector membership for four measures across fiscal years 2014 to 2023, and the sector mix of the top decile by gross margin against the sector mix of the sample](/blog-images/high-margin-screen-sector-bet-python.png)

```
543 companies, 4,607 company-years, fiscal 2014-2023, 9 sectors

Share of cross-sectional variance explained by sector, fiscal 2014-2023
                   ranks: mean     min     max  values: mean
Gross margin            15.5%   12.3%   18.9%         15.6%
Operating margin         9.7%    5.3%   21.8%          9.7%
Net margin               9.8%    2.6%   22.9%          9.5%
Return on assets        16.1%    9.9%   24.3%         13.6%

Sector mix of the top decile, pooled 2014-2023 (sample share in brackets)
Gross margin      Health Care 33% [15%]  Information Technology 29% [16%]   top two 62% vs 31%
Operating margin  Information Technology 26% [16%]  Health Care 21% [15%]   top two 47% vs 31%
Net margin        Information Technology 33% [16%]  Health Care 25% [15%]   top two 57% vs 31%
Return on assets  Consumer Discretionary 28% [16%]  Information Technology 21% [16%]   top two 49% vs 32%

Names kept when the same screen is ranked inside each sector
Gross margin        57.4%  (year range 53%-62%)
Operating margin    71.6%  (year range 64%-78%)
Net margin          64.7%  (year range 60%-74%)
Return on assets    70.3%  (year range 64%-84%)
```

**What this tells us**

Sector membership explains between 9.7% and 16.1% of the cross-sectional spread. Operating margin and net margin are the least sector-driven measures, both near 9.7%, while gross margin and return on assets sit at 15.5% and 16.1%. Splitting raw values instead of ranks agrees to within 2.5 percentage points on every measure. Between 84% and 90% of the profitability difference between two companies sits inside their sectors, not between them.

The yearly range carries a second point. Gross margin is stable, moving between 12.3% and 18.9%, because the cost structure separating a drug company from a steel company does not change from one year to the next. Operating margin swings from 5.3% to 21.8% and peaks in fiscal 2020, when the shock arrived with a sector shape: cruise lines and oil producers took losses together while software did not.

The tail contradicts the average. Health care supplies 33% of the pooled top decile by gross margin against 15% of the sample, and technology 29% against 16%, so two sectors out of nine hold 62% of the list. The arithmetic is consistent: a modest gap between sector averages, sitting on top of wide distributions, still stacks the extreme tail, because that is where shifted distributions differ most.

Neutralising measures the same effect directly. Ranking gross margin inside each sector keeps 57.4% of the original decile, against 71.6% for operating margin and 70.3% for return on assets.

**So what?**

Pick the measure to fit the intent. A screen meant to find unusually profitable businesses rather than unusually profitable industries should rank on operating margin, which carries the least sector content on both tests. Gross margin is the weakest choice for that purpose, and it is the measure most quality factors are built on.

If gross margin is the measure, rank it inside sectors. The change is one groupby, and the result is a different portfolio: replacing 43% of the names means a backtest of the raw screen says little about the neutral version, so the two belong in separate tests.

The opposite case is legitimate. An investor who wants exposure to high-margin business models is buying health care and technology deliberately, and the raw screen delivers that cheaply. The failure worth avoiding is calling the raw output stock selection, then finding during a technology drawdown that it was a sector position.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
