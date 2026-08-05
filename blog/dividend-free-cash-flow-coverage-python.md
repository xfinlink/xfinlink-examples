**Which Dividends Are Not Covered by Cash? Free-Cash-Flow Coverage Screening in Python**

August 5, 2026 · DIVIDENDS

**What's the question?**

The conventional test of dividend safety is the payout ratio: dividends divided by net income. Anything under 60% reads as comfortable, and a screen built on that threshold will pass most of the large-cap dividend universe.

Net income is an accrual figure. It charges depreciation on assets bought years ago, and it says nothing about what a company spends on new assets this year. Dividends are settled in cash. Free-cash-flow coverage measures the same obligation against the cash actually available: operating cash flow minus capital expenditure, divided by dividends paid. A coverage ratio above 1.0 means the company funded its distribution out of what the business produced after reinvestment. Below 1.0 means the difference came from borrowing, share issuance, or the cash balance.

The practical question is which dividend payers pass the earnings test and fail the cash test, and how far the two rankings diverge across a large universe.

**The approach**

The universe is the current S&P 500. Each company contributes its most recent annual filing, with fiscal periods ending between April 2025 and June 2026. Built from SEC EDGAR public filings and market data.

1. Pull annual net income, operating cash flow, capital expenditure, and the common stock dividend
2. Keep companies that paid a common dividend and reported positive net income
3. Exclude Financials and Real Estate, where the capital-expenditure line is not comparable: banks and insurers have no meaningful capex against operating cash flow, and REIT capital spending mixes maintenance with portfolio growth
4. Compute the payout ratio as dividends over net income, and coverage as free cash flow over dividends
5. Measure the Spearman rank correlation between the two, then isolate the companies where they disagree

That leaves 253 dividend payers.

**Code**

```python
import xfinlink as xfl
from scipy.stats import spearmanr

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

fields = ["period_end", "net_income", "operating_cash_flow",
          "capital_expenditures", "dividends_paid_common"]

tickers = xfl.index("sp500")["ticker"].dropna().unique().tolist()
raw = xfl.fundamentals(tickers, period_type="annual", start="2024-06-01",
                       fields=fields, max_rows=10000)

latest = raw.sort_values("period_end").groupby("ticker").tail(1)
df = latest[
    (latest["dividends_paid_common"] > 0)
    & (latest["net_income"] > 0)
    & latest["operating_cash_flow"].notna()
    & latest["capital_expenditures"].notna()
    & ~latest["gics_sector"].isin(["Financials", "Real Estate"])
].copy()

df["fcf"] = df["operating_cash_flow"] - df["capital_expenditures"]
df["payout"] = df["dividends_paid_common"] / df["net_income"]
df["coverage"] = df["fcf"] / df["dividends_paid_common"]

rho, pval = spearmanr(df["payout"], df["coverage"])
blind_spot = df[(df["payout"] < 0.60) & (df["coverage"] < 1.0)]

print(f"Spearman: {rho:.3f}   uncovered: {(df['coverage'] < 1).sum()} of {len(df)}")
print(blind_spot.sort_values("coverage")[["ticker", "gics_sector", "payout", "coverage"]])
```

Full script with formatting and visualisation: [dividend-free-cash-flow-coverage-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/dividend-free-cash-flow-coverage-python.py)

**Output**

![Scatter plot of dividend payout ratio against free-cash-flow coverage for 253 S&P 500 dividend payers, with utilities marked in blue](/blog-images/dividend-free-cash-flow-coverage-python.png)

```
=== Dividend Coverage: Cash vs Earnings ===
Universe: 253 S&P 500 dividend payers (latest annual filing, ex-Financials, ex-Real Estate)
Fiscal periods end 2025-04-25 to 2026-06-30
Spearman rank correlation, payout vs coverage: -0.756 (p=4.0e-48)
Dividend not covered by free cash flow: 43 of 253 (17.0%)
Median coverage: 2.57x   Median payout: 39.8%

--- Payout ratio under 60%, free cash flow below the dividend (18 names) ---
Ticker Sector                     NetInc      OCF    Capex       FCF      Div   Payout    Cover
PCG    Utilities                   2,703    8,716   11,787    -3,071      277   10.2%   -11.09x
CNP    Utilities                   1,052    2,486    4,870    -2,384      574   54.6%    -4.15x
ORCL   Information Technology     17,087   31,977   55,663   -23,686    5,725   33.5%    -4.14x
AES    Utilities                     910    4,306    5,929    -1,623      501   55.1%    -3.24x
ATO    Utilities                   1,199    2,049    3,561    -1,512      554   46.2%    -2.73x
MOS    Materials                     541      825    1,359      -535      282   52.2%    -1.89x
AWK    Utilities                   1,111    2,059    3,126    -1,067      648   58.3%    -1.65x
EXC    Utilities                   2,768    6,254    8,529    -2,275    1,617   58.4%    -1.41x
AEE    Utilities                   1,461    3,353    4,128      -775      768   52.6%    -1.01x
NI     Utilities                     930    2,362    2,782      -420      533   57.3%    -0.79x
AEP    Utilities                   3,696    6,944    8,453    -1,509    2,008   54.3%    -0.75x
EIX    Utilities                   4,459    5,800    6,515      -715    1,274   28.6%    -0.56x
NUE    Materials                   1,744    3,234    3,422      -188      511   29.3%    -0.37x
PEG    Utilities                   2,111    3,298    3,272        26    1,258   59.6%     0.02x
LEN    Consumer Discretionary      2,078      217      189        28      521   25.1%     0.05x
TRGP   Energy                      1,923    3,917    3,333       584      815   42.4%     0.72x
HUM    Health Care                 1,188      921      546       375      426   35.9%     0.88x
MMM    Industrials                 3,250    2,306      910     1,396    1,562   48.1%     0.89x

Sector mix of that group:
  Utilities                11
  Materials                 2
  Information Technology    1
  Consumer Discretionary    1
  Energy                    1
  Health Care               1
  Industrials               1

--- Payout ratio above 100%, cash covers the dividend more than 1.5x (5 names) ---
Ticker Sector                     NetInc       FCF      Div   Payout    Cover
TPR    Consumer Discretionary        183     1,094      299  163.4%     3.65x
CVS    Health Care                 1,768     7,807    3,409  192.8%     2.29x
TSN    Consumer Staples              474     1,177      697  147.0%     1.69x
HSY    Consumer Staples              883     1,823    1,085  122.9%     1.68x
ABBV   Health Care                 4,226    17,816   11,819  279.7%     1.51x
```

**What this tells us**

The rank correlation of −0.756 means the two measures mostly agree: companies with a high payout ratio tend to have low cash coverage. The agreement is not complete. Of 253 payers, 43 did not cover the dividend from free cash flow, and 18 of those reported a payout ratio below 60%. A screen with a 60% cutoff would have passed every one of them.

Eleven of the 18 are utilities, which points at structure rather than distress. Regulated utilities spend far more on plant than they charge to depreciation, because the rate base is deliberately growing. American Electric Power earned 3,696 and distributed 2,008, a payout of 54.3%; operating cash flow of 6,944 against capital expenditure of 8,453 left free cash flow of −1,509. The dividend was funded from debt and equity issuance, and the regulator permits a return on the enlarged asset base. That is a working model, but its safety depends on capital-market access and rate-case outcomes, neither of which the payout ratio observes.

Oracle is the largest shortfall in dollar terms. A payout ratio of 33.5% looks untroubled until capital expenditure of 55,663 is set against operating cash flow of 31,977, which produces free cash flow of −23,686 and coverage near −4.1x.

3M shows a different mechanism. Capital expenditure of 910 is modest, and the gap comes from operating cash flow of 2,306 sitting well below net income of 3,250. Payout reads 48.1%; coverage is 0.89x.

The reverse error also appears. Five companies posted a payout ratio above 100% while covering the dividend more than 1.5 times in cash. CVS Health earned 1,768, distributed 3,409, and still generated 7,807 of free cash flow, for coverage of 2.29x. Non-cash charges such as impairments and intangible amortisation depress reported earnings without touching the cash that pays the dividend.

**So what?**

Run both ratios and treat the gap between them as the signal. Where cash coverage sits far below the earnings payout, the dividend depends on external funding, and the relevant risk is a credit and capital-markets risk rather than an operating one. Where cash coverage sits far above the earnings payout, a payout ratio above 100% is an accounting artefact and not a warning.

For capital-intensive sectors, the earnings payout ratio has almost no discriminating power: every utility in this sample passes it. Coverage separates them. Setting a minimum coverage ratio alongside a maximum payout ratio, and reviewing the disagreements by hand, catches both failure modes that a single-ratio screen produces.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
