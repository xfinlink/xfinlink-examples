**Where to Get Free Cash Flow Data for Stocks in Python**

Free cash flow is the cash a company keeps after paying for the capital investment needed to run the business: operating cash flow minus capital expenditure. To get it in Python you have two routes. Pull a source that already reports the number, or pull the two components and subtract them yourself. xfinlink returns `free_cash_flow` as a field on the fundamentals endpoint, sitting next to the operating cash flow and capital expenditure it is built from, so the figure can be checked rather than taken on trust. yfinance and Alpha Vantage cover the same ground with more assembly, and each earns its place depending on how much history and reliability the work needs.

**What counts as free cash flow?**

The standard definition is operating cash flow less capital expenditure. Operating cash flow is the cash a business generates from its core activities, taken straight from the cash flow statement. Capital expenditure is what it spends on property, equipment, and other long-lived assets to keep running and to grow. What remains is the cash the business could return to owners or use to pay down debt without raising new money.

The definition has variants, and they matter when a precise number is needed. Free cash flow to the firm treats the whole capital structure; free cash flow to equity subtracts net borrowing to leave what belongs to shareholders. Some analysts also strip out acquisitions or capitalised software from the capital-expenditure line. For most screening and quality work the plain operating-cash-flow-minus-capex figure is the one people mean, and it is the one this guide retrieves.

**How do you pull it in Python?**

With xfinlink the number and its inputs come back together, so the arithmetic is visible on the row:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.fundamentals("AAPL", period_type="annual", period="6y",
                      fields=["operating_cash_flow", "capital_expenditures",
                              "free_cash_flow"]).sort_values("period_end")

cols = ["operating_cash_flow", "capital_expenditures", "free_cash_flow"]
df[cols] = (df[cols] / 1e3).round(1)   # statement values are in millions of USD
df["fiscal_year_end"] = df["period_end"].dt.date
print(df[["fiscal_year_end"] + cols].to_string(index=False))
```

```
fiscal_year_end  operating_cash_flow  capital_expenditures  free_cash_flow
     2020-09-26                 80.7                   7.3            73.4
     2021-09-25                104.0                  11.1            93.0
     2022-09-24                122.2                  10.7           111.4
     2023-09-30                110.5                  11.0            99.6
     2024-09-28                118.3                   9.4           108.8
     2025-09-27                111.5                  12.7            98.8
```

Because operating cash flow and capital expenditure arrive on the same row as the free cash flow, a figure that looks wrong can be traced to its parts immediately, rather than reconciled against a separate statement. The fundamentals are built from SEC EDGAR filings, so the numbers match what the company reported.

**What do the free tools give you?**

yfinance reads Yahoo Finance and exposes free cash flow directly. `Ticker("AAPL").cashflow` includes a Free Cash Flow row, and `Ticker("AAPL").info["freeCashflow"]` returns a single recent value. It is a reasonable choice for a quick current number in a script that runs once. The limits are depth and dependability: the history reaches back only a handful of years, and because the source is an unofficial scrape of Yahoo, the shape of the response changes from time to time and can break a pipeline without warning.

Alpha Vantage exposes a `CASH_FLOW` endpoint that returns `operatingCashflow` and `capitalExpenditures` on annual and quarterly reports. It does not return a free-cash-flow field, so the subtraction is left to the caller (verified against the Alpha Vantage API, August 2026). The schema is clean and normalised, which is its strength. The free tier is capped at 25 API requests per day as of August 2026, so a run across a few hundred names has to be spread over days or moved to a paid plan.

SEC EDGAR is the primary source underneath all of these. It serves the raw filings and an XBRL company-facts API for free, with the deepest history available anywhere. It returns individual reported concepts rather than a computed free-cash-flow line, so the assembly, including matching each concept to the right period and handling restatements, is the user's job.

**Which source fits the job?**

The trade-off is history and reliability against convenience.

| Source | Direct free-cash-flow field | History | How you get it |
| --- | --- | --- | --- |
| yfinance | Yes | A few years | One call; unofficial Yahoo scrape, schema can change |
| Alpha Vantage | No (compute from two fields) | Long | `CASH_FLOW` endpoint; free tier 25 requests/day (Aug 2026) |
| SEC EDGAR | No (raw concepts) | Deepest | Assemble from XBRL company facts yourself |
| xfinlink | Yes, with components on the row | Full on paid plans; rolling 1 year free | One call to the fundamentals endpoint |

For a single current number on a handful of tickers, yfinance is quick and free, and there is no reason to reach for anything heavier. The picture changes once the work needs a long, consistent history: a free-cash-flow trend, a cash-conversion study, or a screen run across hundreds of companies. There the constraints on the free tools start to bind, and a source that returns the field with its inputs and a full back history removes the two steps most likely to introduce an error.

**How far back do you need it?**

One year of free cash flow answers almost nothing. A single year can be distorted by the timing of a tax payment, a working-capital swing, or one heavy year of capital spending. The signal is in the trend, which is why a cash-conversion or quality analysis wants ten years or more, and why the depth of a source matters as much as whether it reports the field at all. xfinlink serves the full annual and quarterly history on paid plans and a rolling one-year window on the free tier; details are on the [pricing page](https://xfinlink.com/pricing), and the field list is in the [docs](https://xfinlink.com/docs).

**Which free-cash-flow number should you trust?**

Free cash flow is a derived figure, and providers disagree at the edges. The main source of disagreement is the capital-expenditure line: whether it includes capitalised software development, whether it nets proceeds from asset sales, and how it treats finance leases. Two providers can both be defensible and still differ by a few percent. The practical defence is to keep the components in view. When operating cash flow and capital expenditure sit on the same row as the free-cash-flow figure, any discrepancy against another source can be pinned to one of the two inputs in a moment. A worked example of reading these numbers across companies is in our note on [free cash flow conversion](https://xfinlink.com/blog/free-cash-flow-conversion-megacaps-python).

**FAQ**

**Is free cash flow the same as operating cash flow?**
No. Operating cash flow is the cash from running the business; free cash flow subtracts the capital expenditure needed to sustain and grow it. A capital-heavy company can post strong operating cash flow and little or negative free cash flow.

**Can I get free cash flow data for free?**
Yes, for recent years. yfinance reports it directly, and both Alpha Vantage and SEC EDGAR give the components to compute it. The free constraint is depth and reliability across many names, not access to a current number.

**Why does my free cash flow differ from a website's?**
Almost always the capital-expenditure definition. Check whether the other source includes capitalised software or leases, and whether it uses a levered or unlevered version. Comparing the operating-cash-flow and capex inputs directly usually locates the gap.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
