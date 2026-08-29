**Do Company Insiders Predict Their Own Stock's Returns? Form 4 Cross-Section in Python**

August 29, 2026 · SIGNAL-EVALUATION

**What's the question?**

Corporate officers and directors must report open-market trades in their own company on SEC Form 4 within two business days. The appeal of reading those filings is obvious: an executive spending personal money is backing an opinion with cash.

Academic work supports the idea with a qualifier. Lakonishok and Lee found in 2001 that insider purchases predicted returns mainly in small companies, and Jeng, Metrick and Zeckhauser measured a similar size effect in 2003.

The S&P 500 is therefore the hardest place for the edge to survive, since any private view held by an officer competes with dozens of analysts. The question is narrow: does the direction of insider trading in one quarter say anything about the following year? Market-adjusted return below means a stock's total return minus the equal-weighted average across index members over the same window.

**The approach**

Insider signals usually use dollar amounts. This one uses transaction counts, treating each insider's purchase as one signal.

1. Rebuild the S&P 500 roster at each quarter end from March 2014 to June 2025, keyed on company identifier rather than ticker, so a company removed later still counts for the quarters it was a member. That gives 46 formation quarters across 721 companies.
2. Keep Form 4 transaction codes P and S, read as filed and normalised for case. P is an open-market purchase, S an open-market sale. Grants, option exercises and tax withholding are compensation mechanics rather than a decision to trade.
3. Sort each company-quarter into buying only, buying and selling, selling only, or no open-market trades.
4. Open the holding period one full month after the quarter closes, so every filing is public before a day of return is counted.
5. Compound total returns over 1, 3, 6 and 12 months, winsorise at the 1st and 99th percentiles, then subtract the cross-sectional mean.
6. Measure the 12 months running up to each quarter end as well, which shows what kind of company arrives in each group.

Significance rests on the 46 quarterly cross-sectional averages rather than the 22,396 company-quarters, because overlapping annual windows share months and would inflate every t-statistic.

**Code**

```python
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

def chunked(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]

quarter_ends = pd.date_range("2014-03-31", "2025-06-30", freq="QE")
rosters = {q: set(xfl.index("sp500", as_of=q.strftime("%Y-%m-%d"))["entity_id"])
           for q in quarter_ends}
universe = sorted(set().union(*rosters.values()))

ins = pd.concat([xfl.insiders(entity_id=chunk, form_type="4",
                              start=q.to_period("Q").start_time.strftime("%Y-%m-%d"),
                              end=q.strftime("%Y-%m-%d"))
                 for chunk in chunked(universe, 50) for q in quarter_ends])

# codes read as filed, then normalised: P is an open-market purchase, S a sale
ins["code"] = ins["transaction_code"].astype(str).str.strip().str.upper()
trades = ins[ins["code"].isin(["P", "S"])]
trades["q"] = trades["transaction_date"].dt.tz_localize(None).dt.to_period("Q")

px = pd.concat([xfl.prices(entity_id=chunk, start="2014-01-01", end="2026-08-31",
                           interval="1mo", fields=["return_daily"], max_rows=200_000)
                for chunk in chunked(universe, 50)])

px["m"] = px["date"].dt.tz_localize(None).dt.to_period("M")
gross = 1.0 + px.drop_duplicates(["entity_id", "m"]).pivot(
    index="m", columns="entity_id", values="return_daily").sort_index()

rows = []
for qend, members in sorted(rosters.items()):
    q = pd.Period(qend, freq="Q")
    sub = trades[(trades["q"] == q) & (trades["entity_id"].isin(members))]
    d = pd.DataFrame(index=pd.Index(sorted(members), name="entity_id"))
    d["n_buy"] = sub[sub["code"] == "P"].groupby("entity_id").size().reindex(d.index).fillna(0)
    d["n_sell"] = sub[sub["code"] == "S"].groupby("entity_id").size().reindex(d.index).fillna(0)
    d["group"] = np.select(
        [(d.n_buy > 0) & (d.n_sell == 0), (d.n_buy > 0) & (d.n_sell > 0),
         (d.n_buy == 0) & (d.n_sell > 0)],
        ["Insider buying only", "Buying and selling", "Insider selling only"],
        default="No open-market trades")

    # holding period opens one month after the quarter closes
    months = pd.period_range(q.end_time.to_period("M") + 2, periods=12, freq="M")
    fwd = gross.loc[months, [c for c in d.index if c in gross.columns]].prod() - 1.0
    fwd = fwd.dropna().clip(*fwd.quantile([0.01, 0.99]))
    t = d.loc[fwd.index].assign(abn=fwd.values - fwd.mean(), q=str(q))
    rows.append(t.reset_index())

panel = pd.concat(rows, ignore_index=True)
per_q = panel.pivot_table(index="q", columns="group", values="abn", aggfunc="mean")
spread = per_q["Insider buying only"] - per_q["Insider selling only"]
print(stats.ttest_1samp(spread, 0.0), spread.mean())
```

Full script with formatting and visualisation: [do-insiders-predict-their-own-stock-returns-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/insider-transactions/do-insiders-predict-their-own-stock-returns-python.py)

**Output**

![Two bar charts comparing four groups of S&P 500 companies sorted by insider trade direction. In the twelve months before the signal the buying-only group trails the index by 15.5 percent and the selling-only group beats it by 4.8 percent. In the twelve months after the signal every group sits within 1.5 percent of the index.](/blog-images/do-insiders-predict-their-own-stock-returns-python.png)

```
companies in the point-in-time universe: 721
open-market purchase transactions 9,799  sale transactions 109,290
formation quarters 46  2014Q1 to 2025Q2

company-quarters by insider activity, 12-month window (22,396 in total)
  Insider buying only     1,144 ( 5.1%)   prior 12m vs index -15.51%
  Buying and selling      1,949 ( 8.7%)   prior 12m vs index  -1.97%
  Insider selling only   13,674 (61.1%)   prior 12m vs index  +4.76%
  No open-market trades   5,629 (25.1%)   prior 12m vs index  -7.47%

forward return after the signal, mean across company-quarters (%)
                             1m      3m      6m     12m         1m      3m      6m     12m
                                                    raw                        vs index
  Insider buying only     1.49    2.25    4.53   11.63     -0.15   -0.55   -1.23   -1.32
  Buying and selling      1.43    2.74    5.25   12.59      0.02   -0.02   -0.09    0.28
  Insider selling only    1.58    2.86    5.67   11.61      0.09    0.11    0.32    0.66
  No open-market trades   1.52    2.52    5.34   10.61     -0.19   -0.16   -0.49   -1.44

spread between groups, tested on the 46 quarterly cross-sectional means
   1m  Insider buying only minus Insider selling only    -0.23pp  t=-0.56  p=0.577
   1m  Insider buying only minus No open-market trades   +0.08pp  t=+0.30  p=0.765
   1m  Insider selling only minus No open-market trades   +0.31pp  t=+1.24  p=0.221
   3m  Insider buying only minus Insider selling only    -0.51pp  t=-0.69  p=0.495
   3m  Insider buying only minus No open-market trades   -0.16pp  t=-0.35  p=0.731
   3m  Insider selling only minus No open-market trades   +0.34pp  t=+0.74  p=0.461
   6m  Insider buying only minus Insider selling only    -1.76pp  t=-1.38  p=0.176
   6m  Insider buying only minus No open-market trades   -0.85pp  t=-1.12  p=0.270
   6m  Insider selling only minus No open-market trades   +0.91pp  t=+1.18  p=0.243
  12m  Insider buying only minus Insider selling only    -2.54pp  t=-1.39  p=0.170
  12m  Insider buying only minus No open-market trades   -0.36pp  t=-0.30  p=0.768
  12m  Insider selling only minus No open-market trades   +2.18pp  t=+2.08  p=0.043

12-month window: buying-only beat selling-only in 19 of 46 quarters
  2014Q1-2019Q4:  -6.65pp  t=-3.12  p=0.005  n=24
  2020Q1-2025Q2:  +1.95pp  t=+0.71  p=0.487  n=22
```

**What this tells us**

The counts set the scale. Across 11 and a half years the filings hold 109,290 open-market sales against 9,799 purchases, and quarters with buying alone make up 5.1 percent of the sample against 61.1 percent for selling alone. Selling is the ordinary state of affairs, because equity compensation keeps converting into shares that executives diversify away from.

The prior-return column carries more information than the whole forward table. Companies whose insiders only bought had trailed the index by 15.51 percent over the preceding year, while companies whose insiders only sold had beaten it by 4.76 percent. Buying at this size of company is a contrarian act that follows a decline; sorting on trade direction is close to sorting on past return with extra steps.

Those gaps then disappear. Every group lands within 1.5 percentage points of the index over the following year, and the buying-minus-selling spread is −2.54 percentage points with a t-statistic of −1.39, which fails to reject and points the wrong way besides. Buying-only quarters beat selling-only quarters in 19 of 46 attempts, slightly worse than a coin. Splitting the sample settles what remains: −6.65 points with a t-statistic of −3.12 through 2019, then +1.95 points with a t-statistic of +0.71 from 2020 onward. A relationship that reverses sign between halves, with the significant half running opposite to theory, is what noise looks like.

One test clears 0.05, and it should not be read as foresight: selling-only companies beat quiet ones by 2.18 percentage points, one result near the threshold out of twelve. Vesting concentrates selling at companies whose shares have risen, so that group inherits recent momentum.

**So what?**

Insider trade direction does not work as a stock-selection signal in the S&P 500, and its honest use in a large-cap process is descriptive. A cluster of purchases marks a stock that has already fallen a long way against its peers, and the evidence above indicates the fall does not reverse on average over the year that follows. Insider selling carries no warning at all at this size, so a headline about executives cashing out is not a reason to trim a position.

None of this contradicts the published work, because none of it looks where that work found the effect. Swapping `sp500` for `russell2000` in the roster call reruns the same test on smaller companies, where the prior evidence suggests there is more to find.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
