**Does Unstable Volatility Warn of Deeper Drawdowns? Vol-of-Vol Sorts in Python**

August 21, 2026 · PRICE-ANALYSIS

**What's the question?**

Volatility of volatility measures how much a stock's own risk level moves around. A stock reaches 30% annualized volatility by sitting near 30% every month, or by alternating between 15% and 45%. The average is identical; the experience is not. The VVIX index has quoted the volatility of the VIX since 2012, and a rising VVIX reads as a market that has lost confidence in its own risk estimate.

The same logic carries to single stocks, and gives a testable claim: two stocks with the same average volatility should not behave alike if one has a volatility that wanders, and the wanderer should be the one that falls harder. The outcome measured here is maximum drawdown over the next quarter, the worst peak-to-trough fall inside the window, which is what triggers a margin call or a stop.

**The approach**

1. Take the S&P 500 roster as it stood on 2 January of each year from 2016 to 2026, so companies later removed stay in the sample for the years they were members
2. Pull daily total returns by entity id rather than by ticker, so a reassigned symbol cannot enter a series
3. At each month end, split the prior six months into six blocks of 21 trading days and compute annualized realized volatility inside each block
4. Set the level to the mean of those six figures and vol-of-vol to their standard deviation divided by that mean, making the measure scale-free
5. Record the worst peak-to-trough fall over the next 63 trading days
6. Sort on vol-of-vol alone, then sort on the level first and on vol-of-vol inside each level bucket

The panel holds 59,608 stock-months across 122 formation dates and 681 companies. Windows without a full six-month history and forward quarter are set aside, as is any window holding a single-day move beyond 75%. A cross-check measure runs alongside: the standard deviation of daily log changes in a 21-day rolling volatility, correlated 0.57 by rank to the first.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LOOK, FWD, BLOCK = 126, 63, 21

rosters = pd.concat([xfl.index("sp500", as_of=f"{y}-01-02").assign(roster_year=y)
                     for y in range(2016, 2027)])
member = {y: set(g["entity_id"].astype(int)) for y, g in rosters.groupby("roster_year")}
ids = sorted({int(i) for i in rosters["entity_id"].dropna()})

px = pd.concat([xfl.prices(entity_id=ids[i:i + 4], start="2015-01-01", end="2026-06-30",
                           fields=["close", "return_daily"], max_rows=200000)
                for i in range(0, len(ids), 4)], ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
ret = px.drop_duplicates(["entity_id", "date"]).pivot(
    index="date", columns="entity_id", values="return_daily").sort_index()

dates = ret.index
month_ends = pd.DatetimeIndex(pd.Series(dates).groupby([dates.year, dates.month]).max().values)
pos = {d: i for i, d in enumerate(dates)}

rows = []
for t in month_ends:
    i = pos[t]
    if i - LOOK + 1 < 0 or i + FWD >= len(dates) or t.year not in member:
        continue
    cols = [c for c in ret.columns if int(c) in member[t.year]]
    past, fwd = ret.iloc[i - LOOK + 1:i + 1][cols], ret.iloc[i + 1:i + 1 + FWD][cols]
    keep = past.notna().all() & fwd.notna().all() & (past.abs().max() <= 0.75) & (fwd.abs().max() <= 0.75)
    cols = [c for c in cols if keep.get(c, False)]

    p = past[cols].to_numpy()
    rv = np.stack([p[k * BLOCK:(k + 1) * BLOCK].std(axis=0, ddof=1)
                   for k in range(LOOK // BLOCK)]) * np.sqrt(252)
    level = rv.mean(axis=0)
    vov = rv.std(axis=0, ddof=1) / level

    cum = np.cumprod(1.0 + fwd[cols].to_numpy(), axis=0)
    mdd = (cum / np.maximum.accumulate(cum, axis=0) - 1.0).min(axis=0)
    rows.append(pd.DataFrame({"date": t, "level": level, "vov": vov, "mdd": mdd}))

panel = pd.concat(rows, ignore_index=True)
cut = lambda s, n: pd.qcut(s, n, labels=False, duplicates="drop")
panel["lt"] = panel.groupby("date")["level"].transform(lambda s: cut(s, 3))
panel["vt"] = panel.groupby("date").apply(
    lambda g: g.groupby("lt")["vov"].transform(lambda s: cut(s, 3))).reset_index(level=0, drop=True)

print(panel.pivot_table(index="lt", columns="vt", values="mdd", aggfunc="mean") * 100)
```

Full script with formatting and visualisation: [vol-of-vol-forward-drawdown-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/vol-of-vol-forward-drawdown-python.py)

**Output**

![Average worst three-month fall for S&P 500 stocks, grouped by volatility level and by how much that volatility moved around](/blog-images/vol-of-vol-forward-drawdown-python.png)

```
point-in-time rosters 2016-2026: 696 distinct companies
daily total returns: 1,776,356 rows, 691 companies, 2015-01-02 to 2026-06-30
panel: 59,608 stock-months over 122 formation dates, 2016-01-29 to 2026-02-27, 681 companies
  set aside: 1,325 without a full six-month history and forward quarter, 1 with a one-day move beyond 75%
rank correlation between volatility level and vol-of-vol: 0.12

Sorted on vol-of-vol alone (quintiles rebuilt every month)
           vol-of-vol   annual vol   mean 3m drawdown   median   next 3m return
  Q1 steadiest      0.19        28.0%           -13.57%   -11.32%            3.13%
  Q2                0.25        28.0%           -13.58%   -11.34%            2.99%
  Q3                0.31        28.0%           -13.44%   -11.15%            3.13%
  Q4                0.37        28.6%           -13.61%   -11.17%            3.22%
  Q5 wobbliest      0.51        31.1%           -13.73%   -11.42%            3.51%
  Q5 minus Q1, vol-of-vol: -0.17pp   t = -0.63   p = 0.529
  Q5 minus Q1, vol-of-vol, log-change measure: -0.60pp   t = -2.81   p = 0.005
  For comparison, wildest minus calmest volatility third: -7.26pp   t = -21.63   p = 0.0000

Mean worst 3-month fall, sorted first on volatility, then on vol-of-vol inside it
                      steadiest vol   middle   wobbliest vol
  calm (20% vol)          -10.27%    -10.41%    -10.31%     wobbliest minus steadiest -0.03pp
  middle (27% vol)        -13.02%    -13.05%    -12.44%     wobbliest minus steadiest +0.58pp
  wild (40% vol)          -18.11%    -17.74%    -16.92%     wobbliest minus steadiest +1.18pp

The six formation dates whose next quarter was worst
  2019-12-31   all stocks  -44.26%   wobbliest minus steadiest  +4.08pp
  2020-01-31   all stocks  -42.87%   wobbliest minus steadiest  +2.32pp
  2020-02-28   all stocks  -37.28%   wobbliest minus steadiest  +3.76pp
  2022-03-31   all stocks  -23.09%   wobbliest minus steadiest  -1.23pp
  2018-09-28   all stocks  -22.88%   wobbliest minus steadiest  -0.17pp
  2025-01-31   all stocks  -21.70%   wobbliest minus steadiest  -1.80pp
```

**What this tells us**

Sorted on vol-of-vol by itself, the signal is flat. The steadiest fifth falls 13.57% over the next quarter and the wobbliest falls 13.73%, a gap of 0.17 percentage points at a t-statistic of 0.63. Ten years of monthly sorts across roughly 490 companies a month give a precise estimate of almost nothing.

The volatility level does what it is supposed to do. The calmest third averages 20% annualized volatility and gives back 10.33% over the next quarter; the wildest third averages 40% and gives back 17.59%. That spread of 7.26 points carries a t-statistic of 21.6, and it is the entire result.

Rank correlation between level and wobble is only 0.12, so the double sort has real independent variation to work with, and inside every level bucket the wobble either does nothing or runs backwards. Among the wildest third, steady-volatility names fall 18.11% and wobbly ones fall 16.92%.

The mechanism is a selection effect in how a stock reaches a high average volatility. One route is persistent risk, such as heavy debt or an earnings stream tied to a commodity price. The other is a single violent month sitting inside five ordinary ones, and only that route scores high on vol-of-vol. A stock that has already had its accident is often one whose bad news is priced, so sorting on instability separates recent damage from pending damage. The worst quarters make the point where it would have mattered most: measured at the end of December 2019, the average S&P 500 stock was three months from a 44.26% fall, and the wobbliest fifth fell 4.08 points less.

**So what?**

Rank on the level and stop there. Six months of realized volatility already delivers the 7.26-point drawdown spread, and an instability term adds nothing.

Vol-of-vol still earns a place, just not as a drawdown forecast. It describes how much confidence a volatility estimate deserves. A name in the wobbliest fifth carries a coefficient of variation of 0.51, so a 30% reading is compatible with 15% and with 45%. Sizing that treats such an estimate as exact will be wrong more often on those names, and the fix is a wider buffer and more frequent re-estimation, not a larger expected fall.

The broader lesson applies to any candidate risk signal. The cross-check measure gave a 0.60-point spread at a t-statistic of 2.81, significant by any conventional test, and it reversed once the volatility level was held fixed. That standalone result was the level in disguise. Run the double sort before a new variable reaches a risk model.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
