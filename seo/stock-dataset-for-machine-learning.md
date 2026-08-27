# How to Build a Stock Dataset for Machine Learning

A stock dataset for machine learning is a panel keyed on company and date, where every feature on a row was public knowledge on that date and the label lies in the future. Most of the work is in the first half of that sentence. Fundamentals get joined on the fiscal period end, which is the date the accounts closed rather than the date anyone could read them, and in a sample of 100 S&P 500 members the median quarter reached the SEC 32 days after its period end. A row dated on the period end that carries those numbers is training the model on information that did not exist yet.

## What shape does the dataset need to be?

One row per company per rebalance date. Features on the left, drawn only from what had been published by that date, and a label on the right measured over the window after it.

Two decisions inside that shape do more damage than any model choice. The first is the join key: an equity panel keyed on ticker strings will splice two unrelated companies together whenever a symbol is reassigned, and nothing in the data raises an error when it happens. Key on a permanent company identifier instead. Every xfinlink frame carries `entity_id` for that purpose, and the guide on [data requirements for backtesting](/blog/data-requirements-for-backtesting) works through the price-side version of the same problem.

The second is the date on each row, which is the subject of the rest of this guide.

## When did each number actually become public?

Every fundamentals row from xfinlink carries both `period_end` and `filing_date`. The first is when the quarter closed; the second is the filing the served figure came from. Subtracting one from the other gives the window during which the figure existed but was not yet readable.

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

members = xfl.index("sp500")
tickers = members["ticker"].dropna().tolist()[:100]

f = xfl.fundamentals(tickers, period_type="quarterly",
                     start="2023-01-01", end="2026-06-30",
                     fields=["revenue", "net_income"])
f = f[f["fiscal_period"].isin(["Q1", "Q2", "Q3"])].copy()
f["lag_days"] = (f["filing_date"] - f["period_end"]).dt.days

print(f"quarters: {len(f)}   companies: {f['entity_id'].nunique()}")
print("days from period end to filing")
for q in (0.25, 0.50, 0.75):
    print(f"  {int(q * 100)}th percentile: {f['lag_days'].quantile(q):.0f}")
for d in (30, 35, 40):
    print(f"  filed within {d} days: {(f['lag_days'] <= d).mean() * 100:.1f}%")
```

```
quarters: 1077   companies: 100
days from period end to filing
  25th percentile: 29
  50th percentile: 32
  75th percentile: 36
  filed within 30 days: 32.4%
  filed within 35 days: 70.6%
  filed within 40 days: 96.7%
```

The sample is the first 100 members of the current S&P 500 alphabetically, quarterly statements from 2023 to mid-2026. Fiscal Q4 is left out because those figures arrive with the annual report rather than a quarterly one, which puts them on a different schedule.

Half of these quarters took more than 32 days to reach a filing, and only a third were filed inside 30. Nothing about that is unusual: it is the reporting calendar working normally, and it is the size of the hole a period-end join digs.

## How much does joining on period end actually cost?

It depends on where the rebalance date falls in the reporting calendar, which is why the error is easy to miss in testing and expensive in production. Same 100 companies, five rebalance dates, counting the companies whose most recent `period_end` had not yet been filed.

| Rebalance date | Companies whose newest quarter was not yet filed |
| --- | --- |
| 2025-11-14 | 5 of 100 |
| 2026-01-15 | 10 of 100 |
| 2026-02-13 | 7 of 100 |
| 2026-03-31 | 85 of 100 |
| 2026-05-15 | 5 of 100 |

Mid-quarter the leak is small and looks like noise. Rebalance on a quarter end and it swallows the sample, because the quarter that closed that day will not be filed for another month. A model trained on the March 2026 row learned 85 companies' results before the market did, and the resulting backtest cannot be repaired by any amount of cross-validation.

Apple sits in the contaminated group at that date:

```
AAPL rows around a 2026-03-31 rebalance
period_end filing_date  revenue
2025-12-27  2026-01-30 143756.0
2026-03-28  2026-05-01 111184.0
2026-06-27  2026-07-31 109417.0
```

On 31 March 2026 the newest Apple figure available to anyone was the December quarter, $143.8bn of revenue, filed on 30 January. The quarter ending 28 March reported $111.2bn, and Apple announced it on 30 April with the 10-Q reaching EDGAR the next day. A panel built on `period_end` puts the second number on a March row, five weeks early. Filtering on `filing_date <= rebalance_date` puts the first one there, which is what a model standing on that date could have seen.

Note that the earnings press release beats the filing by a day or two, so `filing_date` is the conservative edge of the availability window rather than the exact moment the market learned the number. Erring on the late side is the right direction for research: it costs a little signal and it cannot manufacture any. The measured version of what happens after that moment is in the note on [post-earnings announcement drift](/blog/post-earnings-announcement-drift-filing-date-python).

## Which companies belong in the panel on each date?

The ones in the universe then, including the ones that later failed or were bought. Taking today's index roster and running it backwards deletes every casualty from history, and the model learns from a sample selected on surviving. `xfl.index("sp500", as_of="2018-06-29")` returns the membership as it stood on that date, which is the fix; the damage the shortcut does is measured in [what is survivorship bias in backtesting](/blog/what-is-survivorship-bias-in-backtesting).

## What should the label be?

Forward total return, not forward price change. Dividends leave the share price on the ex-date and arrive in the holder's account, so a label built from price alone understates the return on exactly the income and value names a fundamentals model tends to select. The `return_daily` column carries the total daily return including dividends, and a forward 21-day label is the product of the next 21 values of it. Keep that window strictly after the feature date.

## How many API requests does building the panel take?

Fewer than the per-request limits suggest, if the source accepts a list of companies in one call. On xfinlink's Pro plan a single `fundamentals` call takes up to 100 tickers, so a full S&P 500 cross-section for a set of fields is five requests rather than 500. The sample above is one index call plus one fundamentals call.

The comparison worth making is against the sources people usually try first. Everything below was read from each provider's own pages on 27 August 2026.

| Source | Request budget |
| --- | --- |
| SEC EDGAR XBRL APIs | Free; the SEC states they "do not require any authentication or API keys to access" |
| Alpha Vantage free tier | "25 API requests per day" |
| Alpha Vantage premium | From $49.99/month for "75 requests/min" |
| xfinlink Free | 100 requests/day, 1 ticker per call, 1-year rolling history |
| xfinlink Pro | $29/month, 10,000 requests/day, 100 tickers per call, full history |

EDGAR deserves the credit here. Each fact in its company-concept API carries `filed`, `accn` and `form`, so the availability date is in the primary source at no cost, and a small study can be built on it directly. Pulling Apple's revenue concept today returns, among others, a fact for the year to 30 September 2017 stamped `"filed": "2019-10-31"`, which is the same figure re-presented as a comparative in a later 10-K. That is the shape of the work EDGAR leaves you: one company, one concept, many facts, several vintages, and a panel to assemble before any of it becomes a training set.

A commercial API sells that assembly. The dataset above took two calls and arrived as a DataFrame with `entity_id`, `period_end` and `filing_date` already on every row. A free xfinlink key covers writing and debugging the pipeline against a rolling twelve-month window; the [Pro plan](/pricing) at $29 a month opens full history and the 100-ticker request cap, and the [docs](/docs) list every field the panel can carry.

## FAQ

**Can a free tier build a machine-learning dataset?**
It can build and test the pipeline. A free xfinlink key serves a rolling one-year window and one ticker per call, which is enough to confirm the joins are right, and any model that needs several market regimes needs the history a paid plan carries.

**Is the filing date the same as when the market learned the number?**
Not exactly. The earnings press release usually lands a day or two before the filing, so `filing_date` is slightly conservative. For research that direction is the safe one, because a late feature can only weaken a result, never invent one.

**How much history does a stock dataset need?**
Enough to contain more than one regime. A panel starting in 2015 has never seen a sustained bond bear market or a credit crisis, so any risk claim it produces is untested. Daily prices reach 1996 and financial statements reach 1950 on paid plans.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
