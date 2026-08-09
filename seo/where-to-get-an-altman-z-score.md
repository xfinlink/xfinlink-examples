# Altman Z-Score: Where To Get It in Python

Two routes, and the choice depends mostly on how many companies are involved. The Altman Z-Score is a weighted sum of five ratios, four of which come straight off a balance sheet and an income statement. The fifth needs the market value of equity, which no filing reports on a current basis. So either assemble the inputs yourself from SEC filings plus a price source, or read the finished number from an API that has already joined the two. In Python the second route is one call:

```python
import xfinlink as xfl
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

xfl.metrics("KHC", period_type="ttm", fields=["altman_z_score"])
```

Edward Altman published the model in 1968 after fitting it on public manufacturers. Above 2.99 a company resembles the survivors in that sample, below 1.81 it resembles the firms that failed within two years, and the span between is the grey zone where the model declines to commit.

## What goes into an Altman Z-Score?

Five terms, each carrying a weight fixed by the original discriminant analysis.

| Term | Weight | What it measures | Where the input comes from |
|---|---|---|---|
| Working capital / total assets | 1.2 | Short-term liquidity cushion | Balance sheet |
| Retained earnings / total assets | 1.4 | Profit accumulated over the company's life | Balance sheet |
| EBIT / total assets | 3.3 | Operating earning power before financing and tax | Income statement |
| Market value of equity / total liabilities | 0.6 | How far equity can fall before liabilities swallow the firm | Share price and share count |
| Revenue / total assets | 1.0 | Asset turnover | Income statement |

Two features of that table drive most of what you will see. Operating earnings carry the heaviest coefficient, so a loss-making year moves the score further than any balance-sheet item does. And the fourth term is the only one that changes while the market is open, which makes a Z-score partly a market opinion rather than a pure accounting measurement. A company can drift from the safe zone into the grey zone without filing anything.

## What score counts as distressed?

Below 1.81 is the distress zone, above 2.99 the safe zone, and the middle is deliberately undecided. Those cut-offs came from a small sample of manufacturers in the 1960s, which is worth remembering before treating 1.79 and 1.83 as different answers.

Read the score as a screen rather than a verdict. It ranks a large universe quickly and tells you which names deserve a reading of the actual filings; it does not tell you that a company will default, and it says nothing at all about whether the equity is cheap. The market-based [distance-to-default measure](https://xfinlink.com/blog/merton-distance-to-default-sp500-python) answers a related question from the opposite direction, using volatility rather than accruals, and the two disagree often enough to be worth running together.

## Can you build it from SEC filings alone?

Four of the five terms, yes. The SEC states that its EDGAR APIs "do not require any authentication or API keys to access", and the companyfacts endpoint "returns all the company concepts data for a company into a single API call", which covers working capital, retained earnings, revenue and the asset and liability totals.

The market value term is the one that breaks. A 10-K does carry a market figure on its cover page, the aggregate public float, but it counts only the shares held by non-affiliates and it is measured once a year at the end of the second fiscal quarter. Apple's fiscal 2025 filing reports a public float of $3,253,431,000,000 as of 28 March 2025, against a market capitalisation of $4.58 trillion when the numbers below were pulled. Substituting the cover-page figure would put the fourth term roughly thirty per cent low and leave it stale for up to a year, so a price source is not optional.

There is also the assembly work. EBIT is usually not a tagged concept in a filing; it is derived. Retained earnings appear under different tags for filers that carry an accumulated deficit. For two or three companies this is an afternoon's work and EDGAR alone is the right answer, since it is free, authoritative and updated within a minute of a filing. For a universe it becomes a mapping project that you then own forever.

## How do you pull the score in Python?

One request per company, which keeps the example inside the free plan's one-ticker-per-request limit.

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

rows = [xfl.metrics(t, period_type="ttm", fields=["altman_z_score", "piotroski_f_score"])
        for t in ["AAPL", "NVDA", "LUV", "F", "KHC", "JPM"]]
table = pd.concat(rows)
print(table[["ticker", "period_end", "altman_z_score", "piotroski_f_score"]].to_string(index=False))
```

```
ticker period_end altman_z_score  piotroski_f_score
  AAPL 2026-06-27          12.55                  8
  NVDA 2026-04-26          55.35                  7
   LUV 2026-06-30           2.22                  7
     F 2026-03-31           0.89                  3
   KHC 2026-06-27           0.52                  4
   JPM 2026-06-30           None                  5
```

Nvidia at 55.35 shows the fourth term dominating: a five trillion dollar market value against a small liability base pushes the score somewhere no manufacturer in the original sample ever sat. Southwest at 2.22 sits in the grey zone for a structural reason rather than a worrying one, because airlines run negative working capital by design, having taken cash for flights not yet flown. Ford at 0.89 reflects a captive finance arm that puts a lending book on the balance sheet of a manufacturer. Kraft Heinz at 0.52 is the case the model was built to catch: an accumulated deficit after years of write-downs, and an operating loss over the trailing four quarters.

None of those four is a prediction of bankruptcy. Each is a reason to read further, which is what a screen is for.

## Why is the score empty for a bank?

JPMorgan returns nothing, and that is the correct answer rather than a gap. Banks and most REITs present an unclassified balance sheet with no current-asset subtotal, so the first term has no inputs and the ratio has no meaning. Insurers sit in the same position.

Serving a number anyway would be worse than serving nothing. The score would compute, look plausible next to its peers, and rank an entire sector on a formula that was never fitted for it. Any provider that returns a Z-score for a large bank is telling you something about its own conventions.

## How do you check a score you did not compute?

Pull the inputs and rebuild it. This is the test that separates a served metric from an opaque one, and it takes about ten lines.

```python
f = xfl.fundamentals("KHC", period_type="quarterly", period="2y",
                     fields=["current_assets_total", "current_liabilities_total",
                             "retained_earnings", "ebit", "revenue",
                             "total_assets", "total_liabilities"]).sort_values("period_end")
q, ttm = f.iloc[-1], f.tail(4)
mc = xfl.metrics("KHC", period_type="ttm", fields=["market_cap"])["market_cap"].iloc[0]

z = (1.2 * (q.current_assets_total - q.current_liabilities_total) / q.total_assets
     + 1.4 * q.retained_earnings / q.total_assets
     + 3.3 * ttm.ebit.sum() / q.total_assets
     + 0.6 * mc / q.total_liabilities
     + 1.0 * ttm.revenue.sum() / q.total_assets)

print(f"balance sheet {str(q.period_end)[:10]}  working capital "
      f"{q.current_assets_total - q.current_liabilities_total:,.0f}M")
print(f"retained earnings {q.retained_earnings:,.0f}M  EBIT (4q) "
      f"{ttm.ebit.sum():,.0f}M  market cap {mc:,.0f}M")
print(f"recomputed Z {z:.2f}")
```

```
balance sheet 2026-06-27  working capital 535M
retained earnings -9,291M  EBIT (4q) -3,177M  market cap 30,030M
recomputed Z 0.52
```

The recomputed figure matches the served one, and the components can be checked one level further down: the retained earnings term of -9,291 million is the accumulated deficit Kraft Heinz reports in its own filing for the quarter ended 27 June 2026. A score that reproduces from published inputs is a score you can defend in a memo. Every field used above comes from the same endpoints, documented in the [metrics reference](https://xfinlink.com/docs), with history depth by plan set out on the [pricing page](https://xfinlink.com/pricing).

## Frequently asked questions

**Does the Z-score change every day?** Yes, through the market-value term. The four accounting terms update when the company files, so between filings the score moves only with the share price. A screen run on Monday and repeated on Friday will not return the same ranking.

**Can the score be used to pick short candidates?** Not on its own. A low score identifies a balance sheet that looks like historical failures, which is already widely known and often already in the price. It works better as a filter on a universe before a valuation screen than as a signal in its own right, and the same applies when [building a screener](https://xfinlink.com/blog/what-api-to-use-for-a-stock-screener) around any single score.

**Should the inputs be annual or trailing twelve months?** Trailing twelve months for the flow items, so the score reflects the most recent four quarters rather than a fiscal year that may have ended eleven months ago. Balance-sheet terms take the latest reported values. The reasoning behind that split is covered in [annual versus quarterly data](https://xfinlink.com/blog/annual-vs-quarterly-financial-data).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
