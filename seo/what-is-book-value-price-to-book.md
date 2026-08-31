# What Is Book Value? Why Price-to-Book Stopped Working

Book value is what a company's own balance sheet says the shareholders own: total assets minus total liabilities. It is the accounting record of money put into the business plus profits kept inside it, less whatever has been paid out or bought back. Price-to-book (P/B) divides the company's market capitalisation by that figure. The ratio was a serviceable value screen for decades, and it has since become close to meaningless across sectors, because accounting rules build the balance sheets of a software company and a refinery by different methods.

## How is book value calculated from a balance sheet?

Take total assets. Subtract total liabilities. What remains is total equity, which is book value. Divide by shares outstanding and you have book value per share; divide market capitalisation by total equity and you have P/B.

Every number in that calculation comes from a filing rather than from a market. Assets sit at historical cost less depreciation, not at what they would fetch today. Retained earnings accumulate over the life of the company, so a firm that has repurchased shares aggressively carries a smaller book than an otherwise identical firm that has not. None of this was a serious problem when a company's productive capacity was mostly machinery and inventory.

## Why does price-to-book break down for software and drug companies?

Because the assets those companies actually run on never enter the balance sheet.

US accounting rules require research and development spending to be charged against profit in the year it happens (ASC 730), with a carve-out for certain internal-use software development costs under a separate rule (ASC 350-40). IFRS expenses research and allows development costs to become an intangible asset only when specific criteria are met (IAS 38). Advertising and brand building are treated the same way in practice. So a pharmaceutical company that spends eighteen billion dollars developing a drug pipeline records eighteen billion of expense and zero of asset, and its equity is smaller by exactly that amount.

Now consider the same pipeline bought rather than built. The acquirer pays a premium over the target's net assets, and that premium lands on the balance sheet as goodwill, which is an asset and counts toward equity. Identical economics, opposite accounting.

Read the ratio without knowing which path a company took and you are ranking on corporate history rather than on value.

## What happens if research spending is treated as an asset?

The standard correction is to capitalise research spending yourself: treat each year's R&D as an asset written off over five years, then add the unamortised balance back to equity. Current-year spending sits on the books in full, the previous year's at eighty percent, and so on.

Ten large non-financial companies, fiscal 2024 annual filings, market capitalisation taken on each company's own fiscal year-end trading day:

| Company | Sector | Book equity ($m) | Capitalised R&D ($m) | Goodwill ($m) | Reported P/B | Adjusted P/B | Change |
|---|---|---|---|---|---|---|---|
| PFE | Health Care | 88,203 | 32,108 | 68,527 | 1.70 | 1.25 | −27% |
| XOM | Energy | 263,705 | 2,755 | 0 | 1.78 | 1.76 | −1% |
| MRK | Health Care | 46,313 | 58,069 | 21,668 | 5.43 | 2.41 | −56% |
| GOOGL | Communication Services | 325,084 | 127,507 | 31,885 | 7.11 | 5.11 | −28% |
| PG | Consumer Staples | 50,226 | 5,920 | 40,303 | 7.73 | 6.91 | −11% |
| META | Communication Services | 182,637 | 109,414 | 20,654 | 8.12 | 5.08 | −37% |
| CAT | Industrials | 19,491 | 5,839 | 5,241 | 8.89 | 6.84 | −23% |
| KO | Consumer Staples | 24,856 | 0 | 18,139 | 10.77 | 10.77 | 0% |
| MSFT | Information Technology | 268,477 | 78,113 | 119,220 | 12.38 | 9.59 | −22% |
| AMGN | Health Care | 5,877 | 15,221 | 18,637 | 23.81 | 6.63 | −72% |

![Reported price-to-book against price-to-book with research capitalised, ten large caps, fiscal 2024](/blog-images/what-is-book-value-price-to-book.png)

Exxon moves by one percent. Coca-Cola does not move at all, since it discloses no separate research line. Merck's ratio more than halves, and Amgen's falls by roughly three quarters.

The ordering changes, which is the part that matters for a screen. On reported figures Amgen is the most expensive of the ten and Coca-Cola sits eighth. After the adjustment Amgen ranks sixth and Coca-Cola is the most expensive name in the set. Nothing about either business changed between the two columns.

The code is short:

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["MSFT", "GOOGL", "META", "MRK", "PFE", "AMGN", "XOM", "CAT", "KO", "PG"]

f = xfl.fundamentals(TICKERS, period_type="annual", start="2019-01-01", end="2025-01-31",
                     fields=["total_equity", "goodwill", "research_and_development"])
m = xfl.metrics(TICKERS, period_type="annual", fields=["market_cap", "pb_ratio"],
                start="2024-01-01", end="2025-01-31")
```

Both endpoints report money in millions of US dollars, which removes the scaling mistake that otherwise ruins this calculation. Full working script: [what-is-book-value-price-to-book.py](https://github.com/xfinlink/xfinlink-examples/blob/main/seo/what-is-book-value-price-to-book.py).

## Why does goodwill push the ratio the other way?

Goodwill is the premium paid above the fair value of what was acquired. It is an asset by rule, it counts toward equity, and it represents nothing you could sell separately.

Amgen carries 18.6 billion dollars of goodwill against 5.9 billion of book equity. Strip the goodwill out and equity is negative by 12.8 billion. Procter & Gamble's goodwill covers 80 percent of its book, Coca-Cola's 73 percent, Microsoft's 44 percent. Exxon reports none at all.

The asymmetry compounds: acquisitive companies get an inflated denominator and organic developers get a deflated one, so P/B rewards the buyer and punishes the builder twice over. A [goodwill-adjusted price-to-book screen](/blog/goodwill-adjusted-price-to-book-python) run across the S&P 500 finds that 67 of 310 non-financial names carry more goodwill than total book equity.

## What should you use instead of price-to-book?

Four practical moves, in rough order of effort.

Compare within an industry, never across. Book value means something fairly consistent among refiners, or among railways. Between a refiner and a software firm it means almost nothing.

Adjust the denominator yourself when the sector is research-heavy. The five-year capitalisation above takes about twenty lines and is defensible; the exact write-off period matters less than applying one consistently.

Look at equity net of goodwill alongside the reported figure. Where the two disagree sharply, the reported ratio is describing acquisition history.

Rank on more than one measure. Free cash flow yield, return on invested capital and P/B disagree with each other in useful ways, and a [multi-factor screen](/blog/three-factor-stock-screen-python) exposes the disagreement rather than hiding it.

Price-to-book still earns its place in banking, insurance and property trusts, where assets are financial, marked regularly, and the balance sheet genuinely describes the business. Outside those, treat the ratio as a starting question.

## Frequently asked questions

**Is a low price-to-book ratio still a value signal?**
Within a sector where balance sheets are built the same way, often yes. Across sectors it mostly identifies companies whose assets happen to be tangible, which is a different thing entirely.

**Can book value be negative?**
Yes, and it is not always a distress signal. Sustained buybacks and accumulated research expense can both drive equity below zero at a profitable company, at which point the ratio stops being defined rather than becoming attractive.

**Where do the underlying numbers come from?**
Book equity, goodwill and the research line come from company filings; the fields used above are `total_equity`, `goodwill` and `research_and_development`. The [API reference](/docs) lists the full statement schema and the [plans page](/pricing) sets out per-request limits. What to check in any fundamentals source before trusting a screen is covered in [what to look for in fundamentals data](/blog/what-to-look-for-in-fundamentals-data).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
