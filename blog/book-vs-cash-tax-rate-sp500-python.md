**Do Companies Pay the Tax Rate They Report? Book vs Cash Tax Rates in Python**

August 15, 2026 · FUNDAMENTAL-ANALYSIS

**What's the question?**

The tax line on an income statement is not a payment. Income tax expense is an accrual that mixes the tax owed on this year's profit with deferred amounts settled in some later year. The money that actually left the business is disclosed separately, as income taxes paid in the notes to the cash flow statement.

Valuation work almost always uses the first number. A discounted cash flow model that applies a reported effective tax rate to forecast operating profit is projecting an accounting charge and calling it a cash outflow.

Deferred tax accounting says any difference is temporary. A deduction claimed earlier for tax than for reporting purposes reverses later, and across enough years the two rates should converge. That claim is testable.

**The approach**

The sample is the current S&P 500, using annual filings for fiscal years 2022 through 2025, built from SEC EDGAR public filings and market data.

1. For every company-year, compute the book effective tax rate as income tax expense divided by pretax income, and the cash tax rate as income taxes paid divided by the same denominator. The gap is the first minus the second, expressed in percentage points.
2. Keep only years with positive pretax income. A tax rate computed against a loss carries no interpretation.
3. Measure the fiscal 2025 cross-section: median rates, the spread of the gap, and how many companies sit far from zero.
4. For the four-year test, sum tax expense, cash taxes and pretax income in dollars for each company and divide once at the end, which removes the sensitivity of a ratio to a small denominator.
5. Rank companies by research spending as a share of revenue and compare that ranking against the cumulative gap.

**Code**

```python
import pandas as pd
import xfinlink as xfl
from scipy.stats import spearmanr

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique().tolist())
frames = [xfl.fundamentals(tickers[i:i + 50], period_type="annual",
                           start="2021-06-01", end="2025-12-31",
                           fields=["fiscal_year", "revenue", "pretax_income",
                                   "income_tax_expense", "cash_taxes_paid",
                                   "research_and_development", "gics_sector"])
          for i in range(0, len(tickers), 50)]
df = pd.concat(frames, ignore_index=True)

d = df[df["fiscal_year"].between(2022, 2025)].dropna(
    subset=["pretax_income", "income_tax_expense", "cash_taxes_paid"])
d = d[d["pretax_income"] > 0]
d["gap"] = (d["income_tax_expense"] - d["cash_taxes_paid"]) / d["pretax_income"]

f25 = d[d["fiscal_year"] == 2025]
print(f25["gap"].median(), (f25["gap"].abs() > 0.05).mean())

# Four years of dollars per company, divided once
years = d.groupby("ticker")["fiscal_year"].nunique()
full = d[d["ticker"].isin(years[years == 4].index)].copy()
full["rd"] = full["research_and_development"].fillna(0).abs()
c = full.groupby(["ticker", "gics_sector"]).agg(
    pretax=("pretax_income", "sum"), tax_expense=("income_tax_expense", "sum"),
    cash_tax=("cash_taxes_paid", "sum"), rev=("revenue", "sum"),
    rd=("rd", "sum")).reset_index()
c["cum_gap"] = (c["tax_expense"] - c["cash_tax"]) / c["pretax"]
c["rd_intensity"] = c["rd"] / c["rev"]

print((c["cum_gap"].abs() > 0.05).mean())
print(spearmanr(c["rd_intensity"], c["cum_gap"]))
```

Full script with formatting and visualisation: [book-vs-cash-tax-rate-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/book-vs-cash-tax-rate-sp500-python.py)

**Output**

<img src="/blog-images/book-vs-cash-tax-rate-sp500-python.png" alt="Distribution of the gap between reported and cash tax rates for S&P 500 companies in fiscal 2025, and the median four-year cumulative gap by sector" style="width:100%;border-radius:8px;margin:16px 0;" />

```
Fetched 2,446 annual rows for 502 companies

==================================================================
FISCAL 2025 CROSS-SECTION
==================================================================
Companies                                 439
Median book effective tax rate          20.4%
Median cash tax rate                    20.8%
Median company-level gap                +0.2pp
25th percentile of gap                  -7.2pp
75th percentile of gap                  +6.5pp
Share with gap wider than  5pp          59.7%
Share with gap wider than 10pp          36.0%

==================================================================
FISCAL 2022-2025 CUMULATIVE (dollars summed, then divided)
==================================================================
Companies with all four years             387
Aggregate book rate                     18.5%
Aggregate cash rate                     21.4%
Median cumulative gap                   -1.7pp
Share with cumulative gap > 5pp         47.8%
Same-signed gap in all four years         134 of 387
  book above cash all four years           48
  cash above book all four years           86

Median cumulative gap by sector (percentage points)
Sector                      n     Book     Cash       Gap
Health Care                49    18.1%    24.8%     -7.6pp
Information Technology     52    16.5%    23.4%     -6.3pp
Industrials                65    21.9%    23.6%     -2.4pp
Materials                  18    22.7%    25.4%     -1.7pp
Consumer Discretionary     35    22.0%    23.4%     -1.2pp
Consumer Staples           27    22.1%    23.3%     -1.2pp
Financials                 63    20.6%    19.5%     -0.2pp
Real Estate                21     4.8%     3.5%     -0.1pp
Communication Services     14    23.3%    22.3%     +0.1pp
Energy                     18    21.5%    19.5%     +3.3pp
Utilities                  25    13.9%     8.0%     +4.7pp

==================================================================
RESEARCH INTENSITY VS THE CUMULATIVE GAP
==================================================================
Spearman rho, full sample         n=387   -0.480   p=1.14e-23
Spearman rho, R&D reporters only  n=184   -0.408   p=9.24e-09

Group                                   n   Median gap
No reported research spend            203       +0.1pp
R&D/revenue quintile 1 (0.0-1.4%)      37       -2.6pp
R&D/revenue quintile 2 (1.5-3.2%)      37       -3.1pp
R&D/revenue quintile 3 (3.3-6.8%)      36       -7.0pp
R&D/revenue quintile 4 (6.8-14.5%)     37       -6.8pp
R&D/revenue quintile 5 (14.8-47.8%)    37      -12.2pp
```

**What this tells us**

At the index level the two rates agree. The median reported rate in fiscal 2025 is 20.4% and the median cash rate is 20.8%, and the median company sits 0.2 percentage points apart.

For individual companies they do not. Half the sample falls between 7.2 points below zero and 6.5 points above it, 59.7% are more than 5 points from zero, and 36.0% are more than 10 points away. The median hides offsetting errors rather than small ones.

Four years do not settle the difference. Of the 387 companies with a complete 2022 to 2025 record, 47.8% still show a cumulative gap wider than 5 points once the dollars are added up, and 134 keep the gap pointing the same way in all four years. A timing difference that never changes sign behaves like a permanent one over any horizon a valuation model cares about.

The sector pattern has a clear cause on each side. Utilities at +4.7 points and Energy at +3.3 points own long-lived physical assets and take depreciation faster for tax than for reporting, so cash tax runs below book tax year after year. Health Care at −7.6 points and Information Technology at −6.3 points pay more cash than they charge. Research intensity ranks against the cumulative gap with a Spearman correlation of −0.480, and the quintile medians fall almost monotonically from +0.1 points for companies reporting no research spend to −12.2 points for the heaviest spenders. That traces to the Section 174 change effective for tax years beginning after 2021, which required research costs to be capitalised and amortised for tax purposes instead of deducted when incurred, lifting cash tax while book expense stayed on its own basis.

Twenty-five companies sit beyond the left edge of the histogram at -40 points, where a small pretax income makes the ratio large. The four-year dollar aggregation carries the weight of the conclusion.

**So what?**

Pull cash taxes paid when a model produces cash. Substituting the reported effective tax rate into a free cash flow forecast for a research-heavy business understates the tax outflow by roughly 6 to 12 points of pretax income, and for a regulated utility it overstates it.

Before writing off a gap as timing, check the sign across several years. A gap that alternates is genuinely reversing, while one holding the same sign for four straight years is structural, and the reported rate will keep missing in the same direction. Roughly a third of this sample sits in that group.

Read the direction as a balance sheet fact. A lasting positive gap is deferred tax the company holds and reinvests without interest, value the income statement never shows. A lasting negative gap means reported profit converts to less cash than the tax rate implies.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
