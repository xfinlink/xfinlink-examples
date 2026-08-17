# Where to Get R&D Spending Data for Public Companies

August 17, 2026 · GUIDES

Research and development expense sits on the income statement of every US company that spends materially on it, which makes the raw data free and public. SEC EDGAR serves it through an XBRL API that needs no key, under the tag `ResearchAndDevelopmentExpense`. That answers the question for one company. The harder problem, and the reason most people end up looking for a data provider, is getting the number for several hundred companies across ten years, lined up so that a January fiscal year-end can be compared against a December one. This guide covers where the data comes from, what each route costs in effort, and how to make the numbers comparable once you have them.

## Where does R&D spending actually get reported?

It is an operating expense on the income statement, disclosed under US GAAP because ASC 730 requires research and development costs to be expensed as incurred rather than capitalised. In the XBRL data that accompanies every 10-K and 10-Q, it carries the tag `ResearchAndDevelopmentExpense`.

Two practical wrinkles come up immediately. Companies label the line to suit their business, so Amazon reports "technology and infrastructure" and Tesla reports plain "research and development", yet both map to the same underlying tag. And plenty of large companies report nothing at all, because they genuinely do no formal research: most banks, utilities, retailers and energy producers have no such line. An empty value for a bank is a fact about the bank, not a hole in the data.

The number is also raw. R&D spending in dollars tells you about company size more than anything else, so the working measure is R&D intensity, research expense divided by revenue for the same period.

## Can you get it free from SEC EDGAR?

Yes, and for a small number of companies this is the right answer. The SEC's XBRL APIs state plainly that "these APIs do not require any authentication or API keys to access" (sec.gov, August 2026). The `companyconcept` endpoint returns every disclosure a single company has made for a single tag:

```
https://data.sec.gov/api/xbrl/companyconcept/CIK0000789019/us-gaap/ResearchAndDevelopmentExpense.json
```

The SEC asks for restraint in return. Its automated-access guidelines "limit each user to a total of no more than 10 requests per second, regardless of the number of machines used to submit requests" (sec.gov, August 2026), and it blocks IP addresses that exceed that.

The cost of the free route is not the requests, it is everything after them. You will need to map ticker symbols to CIK numbers and keep that map current as companies change symbols. You will need to decide what to do when a company restates a prior year, since the endpoint returns every vintage of every fact. You will need to handle companies that switch between `ResearchAndDevelopmentExpense` and a related tag. And you will need to align fiscal calendars yourself. None of it is hard; all of it is work you have to redo whenever something changes upstream.

## What about yfinance?

yfinance does expose the field. Its `fundamentals_keys` definition includes a `ResearchAndDevelopment` entry in the financials section (yfinance `const.py` on GitHub main, August 2026), so `Ticker.get_income_stmt()` returns the line for companies that report it.

For a weekend project on a handful of tickers, that is genuinely the fastest path, and the library is free under the Apache-2.0 licence. The constraints are worth reading before building anything durable on it. The package page states that yfinance "is not affiliated, endorsed, or vetted by Yahoo, Inc.", that it "is intended for research and educational purposes", and that "the Yahoo! finance API is intended for personal use only" (pypi.org, August 2026). Commercial use is a question for Yahoo's terms rather than the library's licence.

## How do you compare companies with different fiscal year ends?

This is the step that quietly ruins most cross-company R&D comparisons. Microsoft closes its fiscal year on 30 June, Nvidia in late January, and Apple in late September. Comparing "fiscal 2024" across those three compares periods that barely overlap.

The fix is to map each fiscal period onto the calendar year it mostly covers: a year ending in January through June belongs to the previous calendar year, everything else to its own. Two lines of pandas, applied once, and every later comparison is honest.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["MSFT", "NVDA", "PFE", "AMZN", "JNJ", "TSLA"]
f = xfl.fundamentals(tickers, period_type="annual", start="2021-06-01", end="2025-12-31",
                     fields=["revenue", "research_and_development"])
f["period_end"] = pd.to_datetime(f["period_end"])
# a fiscal year ending in January-June mostly covers the previous calendar year
f["year"] = f["period_end"].dt.year - (f["period_end"].dt.month <= 6).astype(int)
f["rd_pct"] = f["research_and_development"] / f["revenue"] * 100

print(f[f["year"].between(2022, 2024)]
      .pivot_table(index="ticker", columns="year", values="rd_pct").round(1))
```

```
R&D expense as a percent of revenue
year    2022  2023  2024
ticker
AMZN    14.2  14.9  13.9
JNJ     16.2  17.7  19.4
MSFT    12.8  12.0  11.5
NVDA    27.2  14.2   9.9
PFE     11.3  21.0  17.0
TSLA     3.8   4.1   4.6
```

Nvidia is the instructive row. Its intensity fell from 27.2 percent to 9.9 percent over two years, which looks like a company retreating from research. Its R&D spending actually rose sharply over that period; revenue simply rose far faster. Intensity is a ratio, and a falling ratio can mean either a smaller numerator or a larger denominator. Always check which.

## Which source should you use?

| Source | Cost | Coverage | What you assemble yourself |
| --- | --- | --- | --- |
| SEC EDGAR XBRL API | Free, no key, 10 requests per second | Every US filer, back to the start of XBRL | Ticker-to-CIK mapping, restatement vintages, tag variants, fiscal alignment |
| yfinance | Free, Apache-2.0 | Yahoo's coverage; personal and research use per its own notice | Fiscal alignment, sector labels, index membership |
| xfinlink | Free tier, paid plans for full history | US-listed companies, fundamentals back to 1950 on paid plans | Fiscal alignment |

xfinlink returns `research_and_development` as a field on the fundamentals endpoint and `rd_to_revenue` as a precomputed metric, both documented in the [API reference](https://xfinlink.com/docs). Companies can be addressed by permanent entity id rather than ticker, which matters for any study spanning years in which symbols change, and index membership comes from the same client, so a whole S&P 500 cross-section is a loop over one call. The free tier covers a rolling one-year window; full fundamentals history comes with the paid plans listed on the [pricing page](https://xfinlink.com/pricing).

## Does the number tell you anything?

It does, in the sectors where research is a real strategic lever. A [study of 1,417 company-years across the S&P 500](https://xfinlink.com/blog/rd-intensity-forward-revenue-growth-python) found that companies in the heaviest-spending quintile, around 20 percent of revenue, grew revenue at 15.1 percent a year over the following three years, against 4.8 percent for the lightest spenders. Inside Information Technology the rank correlation between intensity and later growth was 0.460. Inside Consumer Staples it was -0.004, which is nothing.

That split is the practical lesson. R&D intensity is a useful growth predictor in technology and healthcare, and worthless in sectors that spend around 1 percent of revenue on research. Whichever source you pull it from, apply it where it means something.

## FAQ

**Why is R&D missing for some large companies?**
Most likely because they do not report it. Banks, utilities, retailers and energy producers frequently have no research line at all, and US GAAP only requires disclosure where the spending exists.

**Should R&D be capitalised rather than expensed?**
Under US GAAP it must be expensed as incurred, which is why reported earnings understate the economic profit of research-heavy companies. Some analysts capitalise and amortise it over an assumed useful life to compare a heavy spender against a light one. That is an adjustment you make yourself; the reported figure is the input to it.

**Is R&D intensity comparable across sectors?**
No. Median intensity runs above 13 percent in Information Technology and near 1 percent in Consumer Staples, so a cross-sector ranking mostly sorts companies by industry. Compare within a sector, or use [the DuPont style of decomposition](https://xfinlink.com/blog/dupont-roe-decomposition-sp500-python) to separate what a business does from how it is financed.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
