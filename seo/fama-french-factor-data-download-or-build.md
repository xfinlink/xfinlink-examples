# Fama-French Factor Data: Download or Build Your Own?

Download the standard factors from the Kenneth French Data Library at Dartmouth. They cost nothing, they reach back to 1926, and they are the series a referee, a risk committee or a co-author already has open. Build the factors yourself when the sort has to run on a universe the library does not publish, on a schedule it does not publish, or when what you need is the exposure of an individual company rather than the return on a portfolio. Most factor work uses both, and the split is simple: if the factor sits on the right-hand side of a regression, download it; if the factor is the thing being constructed, build it.

## What does the Ken French Data Library actually contain?

Files, and a careful account of how they were made. Read on 30 August 2026, the library publishes the Fama/French three-factor series daily from 1 July 1926, weekly from 2 July 1926 and monthly from July 1926; the five-factor series monthly and daily; a momentum factor monthly from January 1927; short-term and long-term reversal factors; portfolio sorts on size, book-to-market, operating profitability, investment and other characteristics; and industry portfolios at 5, 10, 12, 17, 30, 38, 48 and 49 groupings. Downloads come as TXT or CSV, under the notice "Copyright Eugene F. Fama and Kenneth R. French".

The documentation is the part people underrate. The momentum page states that Mom is built from "six value-weight portfolios formed on size and prior (2-12) returns", that "the monthly size breakpoint is the median NYSE market equity", and that "the monthly prior (2-12) return breakpoints are the 30th and 70th NYSE percentiles". The main library page adds that "although the portfolios include all NYSE, AMEX, and NASDAQ firms with the necessary data, the breakpoints use only NYSE firms", and that "the momentum and short term reversal portfolios are reconstituted monthly and the other research portfolios are reconstituted annually". Separate downloads publish the breakpoints themselves.

The monthly three-factor CSV, downloaded on 30 August 2026, is one row per month and four columns: Mkt-RF, SMB, HML and RF. It ends at June 2026. No company appears in it anywhere.

## When should you simply download them?

Whenever the factor series is an input rather than the output. Alpha estimates, factor loadings, risk attribution and any replication somebody else will check all want the canonical series, and a rebuilt version that differs by a few basis points a month buys nothing except an argument about why it differs. Reading the file takes two lines of pandas against the zip on the site, and pandas-datareader wraps the same source as `web.DataReader("F-F_Research_Data_Factors", "famafrench")`. The [academic research guide](/blog/financial-data-for-academic-research) covers the rest of a free research stack.

## When do the published files stop being enough?

**The universe is not the whole US market.** A mandate that holds only S&P 500 names, a single sector, a liquidity screen, a client exclusion list: none of these is the set of stocks the library sorts. A published SMB measures small firms against large ones across NYSE, AMEX and NASDAQ. The same sort inside a large-cap index measures something quite different.

**The sort is not one of the published sorts.** Momentum with a different skip month, momentum ranked within sectors, a signal scaled by trailing volatility, a weekly rebalance. Each is a small change to a well-understood recipe, and none exists as a file to download.

**The series has to be current.** The monthly momentum file read on 30 August 2026 runs through June 2026. The library also states that it reconstructs the full history of returns each month when it updates the portfolios, so a copy saved last quarter is not guaranteed to match the copy downloaded today.

**You need the names, not the portfolio.** The files carry returns. They do not carry membership, so there is no way to ask which decile a given company sat in on a given date, and no way to turn the series into positions.

| | Ken French Data Library | Factors you build |
| --- | --- | --- |
| Universe | All NYSE, AMEX and NASDAQ firms with the required data | Whatever roster you specify |
| Breakpoints | NYSE percentiles, published separately | Your choice, applied to your universe |
| Sorts available | The published set of characteristics | Any signal you can compute |
| Output | Portfolio return series | Returns plus per-company rank and membership |
| Reconstitution | Monthly for momentum and short-term reversal, annual for the other research portfolios | Any schedule |
| Latest monthly observation, read 30 August 2026 | June 2026 | The last trading day in your price data |
| Cost | Free | Data access plus the code below |

## How do you build size and momentum factors in Python?

Size and momentum need nothing from an accounting statement. Both come out of prices, a share count and a list of which companies were in the universe on the formation date. Value and profitability sorts need book equity aligned to the date it became public, which is a harder problem.

The universe is what decides whether the build is honest. Membership has to be read as of each formation date rather than as of today; the [survivorship bias guide](/blog/what-is-survivorship-bias-in-backtesting) shows the size of the distortion when it is not. Across the 48 monthly S&P 500 rosters from December 2020 to November 2024, 565 different companies appear. The December 2020 roster held 501 of them, and 61 had left the index four years later. A sort run on today's roster never sees them.

The construction below follows the shape of the published factors without pretending to reproduce them:

1. Take the point-in-time roster on each of 48 month-end formation dates, carrying every company by its permanent entity identifier so that a ticker change cannot splice one company's history onto another's.
2. Compound daily total returns to monthly, and read market capitalisation on the formation date.
3. Rank the members. Size sorts on market capitalisation, long the bottom 30% and short the top 30%. Momentum sorts on cumulative return from month t-12 to t-2, long the top 30% and short the bottom 30%.
4. Hold equal-weighted for one month, then reform.

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM = pd.date_range("2020-12-31", "2024-11-30", freq="ME")

rosters = {d: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in FORM}
ids = sorted({int(i) for r in rosters.values() for i in r["entity_id"]})

frames = []
for i in range(0, len(ids), 50):
    frames.append(xfl.prices(entity_id=ids[i:i + 50], start="2019-12-01", end="2024-12-31",
                             fields=["return_daily", "market_cap"], max_rows=200000))
px = pd.concat(frames, ignore_index=True)
px["month"] = px["date"].dt.to_period("M")

ret = (px.dropna(subset=["return_daily"]).groupby(["entity_id", "month"])["return_daily"]
         .apply(lambda s: (1 + s).prod() - 1).unstack(0))
cap = (px.dropna(subset=["market_cap"]).sort_values("date")
         .groupby(["entity_id", "month"])["market_cap"].last().unstack(0)
         .reindex(columns=ret.columns))

signal = ((1 + ret).rolling(11).apply(np.prod, raw=True) - 1).shift(1)

def spread(sig, fwd):
    names = sig.dropna().index.intersection(fwd.dropna().index)
    pct = sig[names].rank(pct=True)
    return fwd[pct[pct >= 0.7].index].mean() - fwd[pct[pct <= 0.3].index].mean()

rows = []
for d, roster in rosters.items():
    formation = pd.Period(d, freq="M")
    members = [int(i) for i in roster["entity_id"] if int(i) in ret.columns]
    fwd = ret.loc[formation + 1, members]
    rows.append({"month": formation + 1,
                 "size": -spread(cap.loc[formation, members], fwd),
                 "momentum": spread(signal.loc[formation, members], fwd)})

built = pd.DataFrame(rows).set_index("month")
```

Full script, including the download of the published series it is compared against: [fama-french-factor-data-download-or-build.py](https://github.com/xfinlink/xfinlink-examples/blob/main/seo/fama-french-factor-data-download-or-build.py)

![Cumulative return of size and momentum factors built on S&P 500 point-in-time rosters against the Ken French Data Library series, January 2021 to December 2024](/blog-images/fama-french-factor-data-download-or-build.png)

```
companies ever in the index: 565   months: 48
S&P 500 size         annualised   0.13%   volatility  9.19%   cumulative  -1.10%
library SMB          annualised  -5.13%   volatility 10.68%   cumulative -20.38%
correlation          0.59

S&P 500 momentum     annualised  -0.03%   volatility 12.05%   cumulative  -2.98%
library Mom          annualised   3.37%   volatility 13.64%   cumulative  10.15%
correlation          0.81
```

## Does a factor built on one index behave like the published one?

Not closely enough to stand in for it. Over the 48 months from January 2021 to December 2024 the momentum sort inside the S&P 500 correlates 0.81 with the library's Mom series, and the size sort correlates 0.59 with SMB. Levels separate further than the correlations suggest: SMB lost 5.13% a year over those four years while the same-signed sort inside the large-cap index returned 0.13% a year.

The reason is the universe, not the arithmetic. Small in the library means small across NYSE, AMEX and NASDAQ. Small inside the S&P 500 means a company worth thirty billion dollars rather than three trillion, which is a different bet, and a portfolio built on one cannot be judged against the return series of the other.

So the two answers divide cleanly. Download the published factors when they are the yardstick. Build them when the portfolio is real and the question is what your own names did. Building needs a membership list correct as of each formation date, an identifier that survives ticker changes, and daily total returns with a market capitalisation attached. The [docs](/docs) list the fields each endpoint returns and the [pricing page](/pricing) sets out the plan limits; a free key covers a rolling twelve months of history.

## FAQ

**Can the published Fama/French factors be reproduced exactly?**
Not without matching the universe and the breakpoints. The library states that its portfolios include all NYSE, AMEX and NASDAQ firms with the necessary data while the breakpoints use only NYSE firms, and small departures compound over a long sample. Cite the published series when the argument turns on a benchmark, and run your own build when it turns on the names you hold.

**Do you need book equity to build a factor?**
Not for size or momentum. Both need prices, a share count and a membership list. Value, profitability and investment sorts need accounting data aligned to the date it became public, which is where the work moves from ranking to matching.

**What should a factor panel be keyed on?**
A permanent company identifier, never the ticker string. Tickers are reassigned, and a panel joined on the symbol will splice one company's returns onto another's without raising an error. The [backtest data requirements guide](/blog/data-requirements-for-backtesting) covers the rest of the checklist.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
