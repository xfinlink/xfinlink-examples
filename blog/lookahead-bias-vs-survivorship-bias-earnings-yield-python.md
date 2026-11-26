# Look-Ahead Bias vs Survivorship Bias in an E/P Backtest

November 26, 2026 · INDEX-UNIVERSE

Description: Look-ahead bias vs survivorship bias, S&P 500 E/P backtest 2011-2024: today’s roster adds 4.0 points a year to returns; early earnings move the spread 0.29.

In an S&P 500 earnings-yield backtest over 2011-2024, survivorship bias adds 3.99 points a year to the average return of the five E/P quintiles (t 9.78, in all 14 years), while look-ahead bias, which only re-ranks the same companies, moves the cheap-minus-dear spread by +0.29 points a year (t 0.24). Neither bias produces a value premium: the spread is -0.92 points a year on the honest setup and -2.62 on the naive one, and survivorship moves it by -0.66 (t -0.55).

## Which bias distorts an earnings-yield backtest more?

Earnings yield (E/P) is annual net income over market value; a high E/P marks a cheap stock. Look-ahead bias uses a figure before it was public: a company with a 31 December year-end files its annual report weeks later, so ranking it on that year’s earnings at the December close reads numbers the market did not have. Survivorship bias picks the universe with hindsight: today’s index list holds only companies that grew enough to join or stay. [The guide to look-ahead bias in backtesting](/blog/what-is-look-ahead-bias-in-backtesting) explains how each enters a test, and [the dated history of S&P 500 additions and deletions](/blog/sp500-additions-and-deletions-history) lists the changes between today’s roster and earlier ones.

Rolf Banz and William Breen argued in the Journal of Finance in 1986 that studies relating accounting data to prices could carry both a look-ahead bias and an ex-post-selection bias.

## Two rosters, two earnings dates

1. Sort on the last business day of each December, 2010 to 2023, either the S&P 500 roster in force that day (`xfl.index("sp500", as_of=...)`; the [historical S&P 500 constituents](/historical-sp500-constituents) page lists it) or today’s roster of 502 lines traced back, every company on it that traded on the sort date and had both earnings figures.
2. Point-in-time earnings: net income of the latest fiscal year on record by the sort date, read with `as_of` (the [point-in-time reference](/docs/point-in-time)). Look-ahead earnings: the latest fiscal year ending by 31 December, filed or not. Both rules read the same figure for a given fiscal year, so the cells differ only in which year they pick.
3. Divide by market value of all share classes at the sort-date close. Loss-makers form their own group; positive E/P is cut into quintiles, Q5 the cheapest. Financials stay in; three companies are set aside for two to six years each.
4. Hold each group the next calendar year, equal weight, bought and held; a stock that stops trading takes its recorded delisting return, then sits in cash. Biases are paired over 14 years.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = pd.Timestamp("2026-11-26")  # the day today's roster was read
forms = pd.date_range("2010-12-01", "2023-12-31", freq="BYE")  # last business day of December
snaps = list(forms) + [pd.Timestamp("2024-12-31")]             # statement snapshots, one a year
ymd = lambda d: d.strftime("%Y-%m-%d")

def mcap(d, tickers=None, issuers=None):
    """Market value (all share classes, millions USD) on the close of d."""
    kw = dict(period_type="ttm", fields=["market_cap"], as_of=ymd(d))
    if tickers is not None:  # start/end only date the tickers' resolution; as_of fixes the snapshot
        m = xfl.metrics(tickers, start=ymd(d - pd.Timedelta(days=300)), end=ymd(d), **kw)
    else:
        m = pd.concat([xfl.metrics(entity_id=issuers[i:i + 500], **kw) for i in range(0, len(issuers), 500)])
    return m[["entity_id", "ticker", "entity_name", "priced_on", "market_cap"]]

# Today's roster: security ids for prices, issuer ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
today_iss = mcap(RUN_DATE, tickers=today["ticker"].tolist())
sec_of = dict(zip(today_iss["entity_id"], today_iss["ticker"].map(dict(zip(today["ticker"], today["entity_id"])))))

rows = []
for d in forms:
    roster = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = mcap(d, tickers=roster["ticker"].tolist()).assign(pit=True)
    a["sec"] = a["ticker"].map(dict(zip(roster["ticker"], roster["entity_id"])))
    b = mcap(d, issuers=sorted(set(sec_of) - set(a["entity_id"]))).assign(pit=False)  # today's names outside it
    b["sec"] = b["entity_id"].map(sec_of)
    rows.append(pd.concat([a, b]).assign(form=d, roster=len(roster)))
S = pd.concat(rows, ignore_index=True).dropna(subset=["sec", "market_cap"])
S = S[(S["market_cap"] > 0) & ((S["form"] - pd.to_datetime(S["priced_on"])).dt.days.between(0, 7))]
S = S.drop_duplicates(["form", "entity_id"])
S["sec"] = S["sec"].astype(int)
S["today"] = S["entity_id"].isin(set(sec_of))

# Annual net income on file at each year-end snapshot (two years of fiscal years)
F = []
for k, d in enumerate(snaps):
    ids = sorted(set(S.loc[S["form"].isin(snaps[max(k - 1, 0):k + 1]), "entity_id"]))
    for i in range(0, len(ids), 500):
        f = xfl.fundamentals(entity_id=ids[i:i + 500], period_type="annual", as_of=ymd(d),
                             start=ymd(d - pd.DateOffset(years=2)), end=ymd(d),
                             fields=["net_income", "revenue"], max_rows=50_000)
        F.append(f.assign(snap=d))
F = pd.concat(F, ignore_index=True)

# Keep true fiscal years: the company's usual year-end month, keyed on rows that carry revenue;
# a 12-month window ending in another month within 300 days of a neighbouring year-end is dropped
F["m"] = F["period_end"].dt.month
usual = F.groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax())
usual.update(F[F["revenue"].notna()].groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax()))
F["off"] = ((F["m"] - F["entity_id"].map(usual)) % 12).map(lambda x: min(x, 12 - x))
ends = F.drop_duplicates(["entity_id", "period_end"]).sort_values(["entity_id", "period_end"])
g = ends.groupby("entity_id")["period_end"]
ends["near"] = np.minimum((ends["period_end"] - g.shift()).dt.days.fillna(999),
                          (g.shift(-1) - ends["period_end"]).dt.days.fillna(999))
F = F.merge(ends[["entity_id", "period_end", "near"]], on=["entity_id", "period_end"])
F = F[F["net_income"].notna() & ~((F["off"] >= 2) & (F["near"] < 300))]
# One figure per fiscal year: the one in the first snapshot that holds it
FY = F.sort_values("snap").drop_duplicates(["entity_id", "period_end"])[
    ["entity_id", "period_end", "snap", "net_income", "filing_date"]]

def pick(rule):
    """Each company's fiscal year under a timing rule, ended within 18 months of the sort."""
    x = S[["form", "entity_id"]].merge(FY, on="entity_id")
    x = x[(x["form"] - x["period_end"]).dt.days <= 548]
    x = x[x["snap"] <= x["form"]] if rule == "pit" else \
        x[x["period_end"] <= pd.to_datetime(x["form"].dt.year.astype(str) + "-12-31")]
    return x.sort_values("period_end").drop_duplicates(["form", "entity_id"], keep="last").set_index(["form", "entity_id"])

for rule in ("pit", "la"):
    p = pick(rule)
    S = S.join(p[["period_end", "net_income", "filing_date"]].add_suffix("_" + rule), on=["form", "entity_id"])
same = (S["period_end_la"] - S["period_end_pit"]).dt.days.between(1, 299)  # two dates, one fiscal year
S.loc[same, ["period_end_la", "net_income_la", "filing_date_la"]] = \
    S.loc[same, ["period_end_pit", "net_income_pit", "filing_date_pit"]].values

# Monthly total returns, December 2010 to December 2024, by security
ids = sorted(S["sec"].unique())
px = pd.concat(xfl.prices(entity_id=ids[i:i + 200], start="2010-12-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily", "close"], max_rows=250_000) for i in range(0, len(ids), 200))
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2010-12", "2024-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(index=R.index, columns=ids).notna()

# Exits: recorded delisting return added to the last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    g = g.index.tolist()
    for i in range(0, len(g), 300):
        d = xfl.prices(entity_id=g[i:i + 300], start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

S["y"] = S["form"].dt.year
S = S[[traded.at[pd.Period(f"{y}-12", "M"), s] for y, s in zip(S["y"], S["sec"])]]
S = S.dropna(subset=["net_income_pit", "net_income_la"])  # one sample for both timings
# Companies sharing a company record with a merger partner before the merger, set aside for those
# years: MeadWestvaco (WestRock's record) to 2014; Tyco and Johnson Controls Inc to 2015
SET_ASIDE = {8097: 2014, 4074: 2015}
SET_ASIDE_YEARS = {670: (2012, 2013)}  # AGL Resources 2012-2013: an annual record dated ahead of its fiscal year
S = S[~(S["y"] <= S["entity_id"].map(SET_ASIDE))]
S = S[[not (e in SET_ASIDE_YEARS and SET_ASIDE_YEARS[e][0] <= y <= SET_ASIDE_YEARS[e][1])
       for e, y in zip(S["entity_id"], S["y"])]].copy()

def held(secs, y, m0=1, m1=12):
    """Equal weight at the start of month m0 of y+1, bought and held to m1; cash after an exit."""
    return (1 + R.loc[f"{y + 1}-{m0:02d}":f"{y + 1}-{m1:02d}", secs].fillna(0)).prod() - 1

for y, g in S.groupby("y"):
    secs = sorted(set(g["sec"]))
    for col, (m0, m1) in {"r": (1, 12), "r_jm": (1, 3), "r_ad": (4, 12)}.items():
        S.loc[g.index, col] = g["sec"].map(held(secs, y, m0, m1)).values

def quintiles(u, t):
    """0 = loss-makers (net income <= 0); 1 = lowest positive E/P ... 5 = highest (cheapest)."""
    ep = u["net_income_" + t] / u["market_cap"]
    q = pd.Series(0, index=u.index)
    pos = ep > 0
    q[pos] = ep[pos].groupby(u.loc[pos, "y"]).transform(lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)
    return q

ROSTERS = {"Roster in force": S["pit"], "Today's roster": S["today"]}
TIMING = {"point-in-time": "pit", "look-ahead": "la"}
cells, spread, Q = {}, {}, {}
for kr, mask in ROSTERS.items():
    for kt, t in TIMING.items():
        u = S[mask].assign(q=quintiles(S[mask], t))
        Q[kr, kt] = u
        tab = u.groupby(["y", "q"])["r"].mean().unstack()
        cells[kr, kt] = tab.assign(n=u.groupby("y").size())
        spread[kr, kt] = tab[5] - tab[1]
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
cagr = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
for k, s in spread.items():
    print(k, f"Q5-Q1 {s.mean() * 100:+.2f} points a year, t {tstat(s):.2f}")
```

Full script with formatting and visualisation: [lookahead-bias-vs-survivorship-bias-earnings-yield-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/lookahead-bias-vs-survivorship-bias-earnings-yield-python.py)

## Output

![Compound annual return of the lowest and highest earnings-yield quintiles and the mean Q5 minus Q1 spread, S&P 500 2011-2024: roster in force 11.9% and 11.0% (spread -0.9) point-in-time, 11.8% and 11.0% (-0.6) with look-ahead earnings; today’s roster 17.1% and 15.5% (-1.6) point-in-time, 17.8% and 15.2% (-2.6) with look-ahead](/blog-images/lookahead-bias-vs-survivorship-bias-earnings-yield-python.png)

```
14 December sorts 2010-12-31 to 2023-12-29, holding years 2011-2024; today's roster 502 lines on 2026-11-26; 782 securities, 163 exit records
Roster companies with both E/P figures: 474-500 a year; today's roster traced back: 416-490
Look-ahead reads a newer fiscal year for 74.9% of roster company-years (December year-ends 99.1%, other year-ends 2.9%); that report was filed a median 54 days after the sort (91.1% within 90 days)

roster, timing                   names   loss  Q1 dear  Q5 cheap   Q5-Q1     t  Q5>Q1  Q1 cagr  Q5 cagr
Roster in force, point-in-time     490  10.5%    12.9%     12.0%   -0.92 -0.27   6/14    11.9%    11.0%
Roster in force, look-ahead        490   8.8%    12.7%     12.1%   -0.63 -0.20   5/14    11.8%    11.0%
Today's roster, point-in-time      457  36.0%    18.2%     16.6%   -1.59 -0.43   5/14    17.1%    15.5%
Today's roster, look-ahead         457  32.6%    19.0%     16.4%   -2.62 -0.76   5/14    17.8%    15.2%

Bias in the Q5-Q1 spread, points a year, paired over 14 years
  Look-ahead, roster in force    +0.29  paired t  0.24  larger in  7/14 years
  Look-ahead, today's roster     -1.04  paired t -0.93  larger in  7/14 years
  Survivorship, point-in-time    -0.66  paired t -0.55  larger in  6/14 years
  Survivorship, look-ahead       -1.99  paired t -1.93  larger in  5/14 years
  Both (naive minus honest)      -1.70  paired t -1.09  larger in  6/14 years
Bias in the average quintile's return, points a year
  Look-ahead, roster in force    +0.05  paired t  0.19  larger in  8/14 years
  Survivorship, point-in-time    +3.99  paired t  9.78  larger in 14/14 years
  Both (naive minus honest)      +4.30  paired t  9.42  larger in 14/14 years

Roster in force: 44.2% of company-years change quintile under look-ahead
  cheaper under look-ahead    1570 company-years  median change in net income  +67.8%  Jan-Mar return  5.17%  Apr-Dec  7.09%
  dearer under look-ahead     1462 company-years  median change in net income  -30.3%  Jan-Mar return  3.26%  Apr-Dec  9.20%
  same quintile               3834 company-years  median change in net income   +0.0%  Jan-Mar return  3.35%  Apr-Dec 10.70%
  point-in-time  Q5-Q1, January-March  -1.76 points (t -0.66)
  point-in-time  Q5-Q1, April-December +2.78 points (t  0.97)
  look-ahead     Q5-Q1, January-March  -0.28 points (t -0.12)
  look-ahead     Q5-Q1, April-December +1.21 points (t  0.53)
```

## Why reading earnings early barely moves the result

Look-ahead keeps the same companies in both cells and only changes their ranks, so its effect on the average of the five quintiles is close to zero by design (+0.05). Survivorship changes the companies, so it acts on levels. The like-for-like comparison is the spread: +0.29 points for look-ahead against -0.66 for survivorship on the honest bases, and -1.04 against -1.99 on the other two. None is significant at the 5% level.

Look-ahead still reads a newer fiscal year for 74.9% of roster company-years (filed a median 54 days after the sort) and moves 44.2% into a different quintile. Companies it pushes into a cheaper quintile, whose net income rose a median 67.8%, returned 5.17% from January to March against 3.35% for companies that kept their quintile. Dearer movers returned 3.26%, close to the unchanged group, so the early gap appears on the cheaper side only, and the output does not test it. From April to December the cheaper movers lagged, 7.09% against 10.70%. On the Q5-Q1 spread the two windows offset: +1.48 points from January to March, -1.57 from April to December.

Today’s roster holds only companies that later reached or kept a place in the index; its loss-makers returned 36.0% a year against 10.5% on the roster in force. On the spread it makes cheap stocks look slightly worse against expensive ones, not better.

## Which bias to fix first in an earnings backtest

Fix the roster first: the wrong list added 3.99 points a year to the five-quintile average in all 14 years, which a long-only backtest reports as return. Then read earnings with `as_of` at the filing date rather than at period end. Early earnings moved the spread little here, but companies that looked cheaper on unfiled figures earned more in the January-to-March window (5.17% against 3.35%, untested). [A price-to-book backtest on the same rosters](/blog/value-investing-backtest-survivorship-bias-python) repeats the roster comparison for book value.

On the rosters and earnings investors had, the cheapest E/P quintile trailed the dearest by 0.92 points a year from 2011 to 2024 (t -0.27, ahead in 6 of 14 years).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
