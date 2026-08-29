# What Is EBITDA and Why Do Sources Disagree?

EBITDA stands for earnings before interest, taxes, depreciation and amortisation, and no accounting standard defines it. US GAAP does not specify it, IFRS does not specify it, and the SEC treats it as a non-GAAP measure that a company may publish only alongside a reconciliation to the nearest GAAP line. Every source that reports an EBITDA figure has therefore made a private decision about which items to add back, and two sources that decided differently will print different numbers for the same company and the same year. Uber's fiscal 2025 shows the range. Built from operating income, the figure is $6,284 million. Built from net income, it is $6,866 million. The Adjusted EBITDA Uber itself reported for that year is $8,730 million. All three trace to the same audited statements, so the useful question about any EBITDA number is not whether it is correct but which build produced it.

## What is EBITDA supposed to measure?

The intent is a rough proxy for the cash an operating business throws off before the effects of how it is financed, where it is taxed, and how much of its asset base was paid for in earlier years. Two competitors in one industry can report very different net income because one carries debt and the other does not, or because one bought a rival and now amortises the intangibles it recognised in the deal. Stripping out interest, tax and the non-cash charges attached to past capital spending is an attempt to put the underlying operations on comparable footing.

Two standard routes lead to it, and they begin at opposite ends of the income statement. The top-down route starts at operating income and adds back depreciation and amortisation. The bottom-up route starts at net income and adds back interest expense, income tax expense, and depreciation and amortisation. Those two routes agree only when nothing sits between operating income and pretax income except interest expense. For any company that holds investments, runs a pension, or accounts for a joint venture under the equity method, something almost always does.

## Why do two sources report different EBITDA for the same company?

Four decisions sit behind every published figure, and none of them is usually disclosed.

<div style="overflow-x:auto;margin:18px 0;">
<table style="width:100%;border-collapse:collapse;font-size:14px;line-height:1.55;">
<thead>
<tr style="border-bottom:1px solid #2e2e2e;">
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">Decision</th>
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">Common choices</th>
<th style="text-align:left;padding:8px 10px;color:#fafafa;font-weight:500;">What it changes</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Starting line</td><td style="padding:8px 10px;">operating income, or net income</td><td style="padding:8px 10px;">Starting at net income pulls every non-operating item into the figure: interest income, investment revaluations, equity-method results, gains on disposals.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Which depreciation number</td><td style="padding:8px 10px;">the income statement line, or the cash flow statement add-back</td><td style="padding:8px 10px;">The two are not always the same number. The cash flow add-back can sweep in amortisation of acquired intangibles, capitalised software and depletion that the income statement presents inside cost of revenue.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Add-backs past the acronym</td><td style="padding:8px 10px;">stock-based compensation, restructuring, impairments, legal and regulatory reserves, one-off gains</td><td style="padding:8px 10px;">This is where the spread comes from. Each add-back raises the figure, and for a software or platform company stock-based compensation alone can move it by a third.</td></tr>
<tr style="border-bottom:1px solid #1f1f1f;"><td style="padding:8px 10px;">Period</td><td style="padding:8px 10px;">fiscal year, trailing twelve months, calendar year</td><td style="padding:8px 10px;">A trailing figure and a fiscal-year figure for a growing company differ by roughly a quarter of growth, on the same definition.</td></tr>
</tbody>
</table>
</div>

The third row does most of the damage. Interest, taxes, depreciation and amortisation are named in the acronym and nobody argues about them; everything after that is editorial, and the editorial part is what separates a computed EBITDA from the number in a company's earnings release. Our note on [annual versus quarterly data](https://xfinlink.com/blog/annual-vs-quarterly-financial-data) covers the period question in more detail.

## How much does the choice change the number?

Uber's fiscal year ended 31 December 2025. Both textbook builds, from the filed statement lines:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

fields = ["operating_income", "depreciation_amortization", "net_income",
          "interest_expense", "income_tax_expense", "stock_based_compensation", "ebitda"]

df = xfl.fundamentals("UBER", start="2025-01-01", end="2025-12-31",
                      period_type="annual", fields=fields)
r = df.iloc[-1]

from_operating = r.operating_income + r.depreciation_amortization
from_net_income = (r.net_income + r.interest_expense
                   + r.income_tax_expense + r.depreciation_amortization)
```

```
period_end                     2025-12-31
operating income                   5,565
depreciation and amortisation        719
net income                        10,053
interest expense                     440
income tax expense                -4,346
stock-based compensation           1,826

EBITDA from operating income       6,284
EBITDA from net income             6,866
served ebitda field                6,284
difference between the builds        582
```

Figures in millions of US dollars, and each of them matches the company's filed XBRL facts for the year.

Two features of that output deserve attention. The income tax line is negative, because Uber recorded a tax benefit rather than a tax charge in 2025, so the bottom-up build subtracts $4,346 million on its way up the statement instead of adding it. And the $582 million gap between the two builds is not an error in either one: it is the net of interest income, other non-operating items, equity-method losses and the share of profit belonging to non-controlling interests. Those items fall below operating income and above net income, so the bottom-up route carries them and the top-down route does not. A company with a large investment portfolio produces a much wider gap than this one.

## What does "adjusted EBITDA" add?

Uber reported Adjusted EBITDA of $8,730 million for fiscal 2025, and its fourth-quarter earnings release sets out the bridge. Starting from income from operations of $5,565 million, it adds depreciation and amortisation of $719 million, stock-based compensation of $1,826 million, legal, non-income tax and regulatory reserves of $564 million, acquisition and financing expenses of $43 million, restructuring charges of $9 million, goodwill and asset impairments of $2 million, and a $2 million loss on a lease arrangement.

So the adjusted figure sits $2,446 million above the top-down build, and stock-based compensation is three quarters of that difference. Nothing here is hidden. The reconciliation is published, Regulation G requires it, and credit agreements commonly define their own EBITDA in similar terms because a lender wants a covenant test that ignores charges the borrower cannot pay cash on. The figure becomes unreadable only when a third party republishes the total on its own, stripped of the bridge that explains it.

## Where does a stored EBITDA field come from?

Some APIs serve EBITDA as a stored field. A call to Alpha Vantage's `OVERVIEW` endpoint for IBM on 29 August 2026 returned `"EBITDA": "16473000000"` and `"EVToEBITDA": "15.58"`, two of 55 fields in the response, with no statement of the starting line, the add-backs, or the period the figure covers (alphavantage.co/documentation, as of August 2026). For sorting a watchlist into rough buckets, that is enough, and it saves a download.

It stops being enough as soon as the number is compared across companies. A stored field applies one vendor's editorial choices to every filer, and those choices interact with how each company tags its statements, so a cross-sectional screen inherits a definition nobody stated and cannot audit. The same argument applies to any single published statistic whose construction is not disclosed, which is the pattern behind [why beta differs between sources](https://xfinlink.com/blog/why-beta-differs-between-sources).

## Which build should you use?

Pick one and hold it fixed across every company in the sample, because consistency matters more than which variant you chose. For most equity work the top-down build, operating income plus depreciation and amortisation, is the sensible default: it stays inside the operating business and does not import investment gains that have nothing to do with trading performance. Add stock-based compensation back only if the analysis is a credit or cash-coverage question, since shares issued to employees are a real cost to existing shareholders even though no cash leaves the company. For anything sensitive to cash timing, use [free cash flow](https://xfinlink.com/blog/where-to-get-free-cash-flow-data-python) rather than an earnings proxy.

A definition you control needs the components served separately. xfinlink returns `operating_income`, `depreciation_amortization`, `net_income`, `interest_expense`, `income_tax_expense`, `stock_based_compensation`, `restructuring_charges` and `impairment_charges` as distinct columns on the same annual row, alongside an `ebitda` column that for Uber's fiscal 2025 equals operating income plus depreciation and amortisation. Every field name and the parameters `fundamentals()` accepts are in the [docs](https://xfinlink.com/docs), and [plans](https://xfinlink.com/pricing) start at a free tier.

## FAQ

**Is EBITDA the same as operating cash flow?**
No. Operating cash flow reflects movements in working capital, cash taxes actually paid, and cash interest, none of which EBITDA sees. A company can grow EBITDA while operating cash flow falls, which usually means receivables or inventory are absorbing the difference.

**Should stock-based compensation be added back?**
For equity valuation, no. Shares issued to employees dilute existing holders, so the cost is real even though it never appears as a cash outflow. For a credit analysis that asks whether a borrower can service debt out of cash, adding it back is defensible.

**Why does EBITDA computed from net income exceed EBITDA computed from operating income?**
Because non-operating income sits between the two lines. Interest income, gains on investments and equity-method results are inside the bottom-up figure and outside the top-down one, and for Uber in fiscal 2025 that is $582 million.

**Does EBITDA appear in a 10-K?**
Not as a GAAP line. Companies present it as a non-GAAP measure with a reconciliation to net income or income from operations, which is the table to read before using any figure a company publishes about itself.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
