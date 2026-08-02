# How Are Shares Outstanding Reported (and Why They Disagree)

A public company reports several different share counts in the same filing, and none of them is wrong. The balance sheet gives the number issued and outstanding on the last day of the period. The income statement gives a weighted average across the period, in a basic version and a diluted version, and those two drive earnings per share. The cover page gives a third figure, dated a few weeks after the period ended. Any of these can end up behind a field labelled "shares outstanding", which is why two providers quoting the same company for the same quarter can differ by two percent, or by a factor of ten if a stock split fell in between.

## What share counts does a filing actually contain?

Five fields cover almost every question anyone asks about share counts, and they answer different questions.

`shares_outstanding` is the count issued and held by investors on the period end date. `common_shares_issued` includes shares the company has bought back and holds in treasury; `treasury_shares` is that repurchased block on its own. Issued minus treasury equals outstanding, for companies that hold treasury stock rather than retiring it.

The other two are averages, not snapshots. `weighted_avg_shares_basic` averages the outstanding count across the reporting period, weighted by the number of days each level was in force. `weighted_avg_shares_diluted` adds the shares that would exist if options, restricted stock units, and convertible securities were settled. Accounting rules require the weighted averages in the EPS denominator, because profit accrues over a period while a share count exists at an instant.

Apple in fiscal 2024 shows the spread clearly:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

f = xfl.fundamentals("AAPL", start="2024-01-01", end="2024-12-31",
                     period_type="annual",
                     fields=["shares_outstanding", "common_shares_issued",
                             "weighted_avg_shares_basic",
                             "weighted_avg_shares_diluted"])
print(f.iloc[0])
```

```
AAPL FY2024 (period_end 2024-09-28)
  shares_outstanding            15,116.8 M
  common_shares_issued          15,116.8 M
  weighted_avg_shares_basic     15,343.8 M
  weighted_avg_shares_diluted   15,408.1 M
  diluted vs outstanding            1.93%
```

Three numbers, one fiscal year, 291.3 million shares between the smallest and the largest. Apple retires repurchased stock instead of parking it in treasury, so issued and outstanding are identical. The basic average sits above the year-end count because buybacks ran through the year and the average still carries the shares that existed in October 2023. The diluted figure adds 64.3 million for unvested equity awards.

## Which count belongs in which calculation?

Earnings per share needs a weighted average, and it needs the matching one: basic EPS over basic shares, diluted EPS over diluted shares. Dividing net income by the period-end count produces a number that is close to reported EPS and not equal to it, which is the usual reason a hand-computed EPS fails to reconcile.

Market capitalisation needs the opposite. Price is an instantaneous quantity, so it pairs with the share count at that instant, not with an average over the preceding year. In [xfinlink](https://xfinlink.com/docs) the `metrics` endpoint follows that rule: `market_cap` is close price times `shares_outstanding`, and `market_cap_diluted` is close price times `weighted_avg_shares_diluted`, kept as a separate field rather than blended into one. Book value per share, cash per share, and the rest of the per-share family all divide by the outstanding count.

## Why does the count jump tenfold after a stock split?

Because the filing said so. Share counts in company filings are as-filed: they carry the value the company printed on the filing date, with no retroactive restatement for splits that happened afterwards. NVIDIA split ten for one on 7 June 2024, and the quarterly balance sheets step across it:

```
NVDA 2024-01-28  shares_outstanding      2,464 M
NVDA 2024-04-28  shares_outstanding      2,460 M
NVDA 2024-07-28  shares_outstanding     24,530 M
NVDA 2024-10-27  shares_outstanding     24,490 M
```

This is a feature of the record, not a defect in it. Restating the April count to a post-split basis would make that row disagree with NVIDIA's own 10-Q, and it would break every calculation that pairs the count with an as-traded price from April. The two sides have to sit on the same basis, and the filing basis is the one that reconciles to the source document.

The daily price series carries its own contemporaneous share count, which steps on the split date alongside the price. Pairing them gives the right answer at every point in history:

```
NVDA 2024-06-28 close $123.54
  price-file share count  24,568,840,000
  market cap             $3.035 T
  using the April balance-sheet count $0.304 T
```

Off by a factor of ten, which is what a split does to anyone who mixes bases. Reconstructing an old market cap correctly means taking the price and the share count from the same date. If a split-adjusted share history is what you want instead, multiply the as-filed count by the cumulative product of `split_ratio` for every split after the period end date; the price series carries `split_ratio` for that purpose. The same trap appears in historical P/E ratios, covered in [Split Adjustment Explained](https://xfinlink.com/blog/split-adjustment-explained).

## Why do two websites show different counts for the same company?

Four reasons, roughly in order of how often they bite.

The first is field choice, which the section above covers. The second is dating: SEC Form 10-Q cover pages report the share count as of the latest practicable date, typically several weeks after the balance sheet date, so a cover-page count and a balance-sheet count for the same quarter legitimately differ.

Third is free float. Outstanding counts every share in existence; float excludes insider holdings, strategic stakes, and restricted stock. Index providers weight by float, so a float-based count runs below the outstanding count, sometimes far below for a founder-controlled company. Two sources are not contradicting each other when one reports each.

Fourth is share classes. A company with Class A and Class C stock files one balance sheet, and the issuer-level count adds the classes together. Alphabet reports 12,211 million shares outstanding for fiscal 2024 across all classes, which will not match a per-class figure quoted for GOOGL alone.

## Where can you pull a share-count history?

For a single symbol, yfinance exposes `Ticker.get_shares_full(start, end)`, which returns a share-count time series (as of August 2026, per its API reference). That is enough for a quick look at one company.

Panel work has a different requirement: the count, the filing date it arrived on, and a price on a matching basis, for many companies at once. `xfl.fundamentals()` returns all five share fields with `period_end` and `filing_date` on every row, `xfl.prices()` returns the daily contemporaneous count next to `close` and `split_ratio`, and `xfl.metrics()` returns market cap already computed on the correct pairing. All three are filing-derived, so the numbers reconcile back to the 10-K or 10-Q they came from. The [free tier](https://xfinlink.com/pricing) covers one ticker per request with twelve months of history, which is enough to check the reconciliation yourself before committing to anything.

## FAQ

**Is shares outstanding the same as free float?**
No. Outstanding includes every issued share held by investors; float excludes insider and strategic holdings. Float is the smaller number and the one index providers use for weighting.

**Which share count should I use for market cap?**
The outstanding count on the same date as the price. Averages belong in EPS, not in market cap.

**Why does my computed EPS not match the reported figure?**
Most likely the denominator. Reported EPS divides by a weighted average across the period, not by the period-end count.

**Are share counts in filings adjusted for later splits?**
Not in the source documents, and not in as-filed data. To move a count onto today's basis, compound the split ratios that occurred after the period end date.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
