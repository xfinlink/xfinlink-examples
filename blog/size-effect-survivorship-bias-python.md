# Size Effect Backtest: Survivorship Bias in the S&P 500

November 21, 2026 · INDEX-UNIVERSE

Description: Size effect survivorship bias in the S&P 500, 2000-2024: the smallest fifth beat the largest by 4.34 points a year on past rosters, 24.74 on today’s roster.

Survivorship bias makes the size effect inside the S&P 500 look nearly six times larger than it was. Sorted on the roster in force each December, the smallest fifth of members beat the largest fifth by an average 4.34 percentage points a year over 2000-2024 (t 1.34, ahead in 13 of 25 years); the same sort on today’s roster reports 24.74 points, with the small end ahead in all 25 years. Of the 20.40 points of bias, 14.38 come from companies that were not yet in the index when sorted and 6.01 from members the index later removed.

## Why today’s members flatter the small end of a size sort

The size effect is the finding that small companies have earned higher returns than large ones; [Rolf Banz reported it for NYSE stocks in 1981](https://econpapers.repec.org/RePEc:eee:jfinec:v:9:y:1981:i:1:p:3-18). Inside a large-cap index the question is narrower: do the smallest members of the S&P 500 beat its giants?

Sorting today’s member list by each company’s market value in past years answers with hindsight. A company that was small in 2000 and sits in the index today is there because it grew, while the small members that shrank, were taken over or failed are absent. The small end of such a sort fills with future winners. [A comparison of cap-weighted and equal-weight backtests](/blog/survivorship-bias-cap-weighted-vs-equal-weight-python) looks at the same two kinds of roster from the weighting side.

## Sorting 25 December rosters into size quintiles

1. Rosters: the S&P 500 on the last business day of each December, 1999-2023, from `xfl.index("sp500", as_of=...)` ([index membership reference](/docs/index-membership)), and today’s roster of 21 November 2026.
2. Size: company market value at the December month-end close, share classes summed where a company has two index lines (Berkshire Hathaway’s class A shares added).
3. Three universes each December: the roster in force; the members of that roster still in the index today; and today’s roster valued on the same date, where companies not yet trading drop out. Each is split into five equal groups by value, Q1 the smallest.
4. Each group is held the next calendar year, equal weight, bought and held. A holding that stops trading takes its recorded delisting return, then sits in cash. Returns end in December 2024.
5. Q1-Q5 is the smallest group’s calendar-year return minus the largest group’s, averaged over 25 years, with its t-statistic (the mean over its standard error).

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Second share classes count toward the company's first index line; BRK-A trades but is not an index line
CLASS_OF = {10003590: 10030636, 10032892: 10031087, 10006698: 10006697, 10003102: 10003101,
            10030753: 10030752, 10004763: 10031256, 10029970: 10029932, 10006133: 10025221}
co = lambda e: CLASS_OF.get(e, e)

# S&P 500 rosters on the last business day of each December 1999-2023, and today's roster
forms = pd.date_range("1999-12-01", "2023-12-31", freq="BYE")
member = {d.year: set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))["entity_id"].dropna().astype(int))
          for d in forms}
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))
ids = sorted(set().union(*member.values()) | today | {10006133})

# Monthly total returns and month-end market values, December 1999 to December 2024
px = pd.concat(xfl.prices(entity_id=ids[i:i + 50], start="1999-11-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily", "market_cap"], max_rows=250_000)
               for i in range(0, len(ids), 50)).reset_index(drop=True)
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
cap = px.pivot(index="m", columns="entity_id", values="market_cap")
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2000-01", "2024-12", freq="M"))

# Each exit's recorded delisting return, added to its last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    d = xfl.prices(entity_id=g.index.tolist(), start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                   fields=["return_daily"])
    delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def sort(names, y):
    """Company market value on the December sort date (share classes summed), split into quintiles."""
    v = cap.loc[pd.Period(f"{y}-12", "M")].reindex(sorted(names | {10006133} if 10025221 in names else names))
    v = v.groupby(co).sum(min_count=1).dropna()
    v = v[v.index.isin(names)]
    return v, np.ceil(v.rank(pct=True) * 5).astype(int)           # 1 = smallest fifth

def held(names, y):
    """Equal weight in January of y+1, bought and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12", sorted(names)]
    return ((1 + r.fillna(0)).prod() - 1)

books = {"Point-in-time rosters": lambda y: member[y],
         "Members then, still in today's index": lambda y: member[y] & today,
         "Today's roster": lambda y: today}
years = range(1999, 2024)
res, info = {}, {}
for name, f in books.items():
    rows = []
    for y in years:
        v, q = sort(f(y), y)
        r = held(set(q.index), y)
        rows.append({"year": y + 1, "n": len(q), **{k: r[q.index[q == k]].mean() for k in range(1, 6)}})
        info[name, y] = (v, q, r)
    res[name] = pd.DataFrame(rows).set_index("year")
cagr = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5

print(f"{len(member)} December rosters 1999-2023, today's roster {len(today)} lines, {len(ids):,} securities, "
      f"{len(delist)} exit records; holding years 2000-2024")
print(f"\n{'universe':38}{'names':>6}" + "".join(f"{'Q' + str(k):>8}" for k in range(1, 6))
      + f"{'Q1-Q5':>8}{'t':>6}{'Q1 ahead':>10}")
smb = {}
for name, df in res.items():
    smb[name] = df[1] - df[5]
    print(f"{name:38}{df['n'].mean():6.0f}" + "".join(f"{cagr(df[k]):8.2%}" for k in range(1, 6))
          + f"{smb[name].mean() * 100:+8.2f}{tstat(smb[name]):6.2f}{(smb[name] > 0).sum():>7}/25")
P, M, T = books
for a, b, label in [(T, P, "today's roster minus point-in-time (total bias)"),
                    (M, P, "  of which members since removed"),
                    (T, M, "  of which names not yet in the index")]:
    x = smb[a] - smb[b]
    print(f"{label:50}{x.mean() * 100:+6.2f} points a year in Q1-Q5, paired t {tstat(x):5.2f}, "
          f"ahead {(x > 0).sum()}/25")
for a, b in [(2000, 2012), (2013, 2024)]:
    print(f"  {a}-{b} Q1-Q5 a year: " + ", ".join(f"{n.split(',')[0]} {smb[n].loc[a:b].mean() * 100:+.2f}"
                                                  for n in books))

# Who sits in the smallest fifth of each universe
print("\nSmallest fifth (Q1), averaged over the 25 sort dates")
for name in books:
    med, outside, later, rq1 = [], [], [], []
    for y in years:
        v, q, r = info[name, y]
        q1 = q.index[q == 1]
        med.append(v[q1].median() / 1e9)
        outside.append(np.mean([e not in member[y] for e in q1]))
        later.append(np.mean([e not in today for e in q1]))
    print(f"  {name:38} median value ${np.mean(med):5.1f}bn, not an S&P 500 member that December "
          f"{np.mean(outside):6.1%}, not in today's index {np.mean(later):6.1%}")
fate = {"still in today's index": 0, "left the index, still trading": 0}
ret_by = {k: [] for k in fate}
for y in years:
    v, q, r = info[P, y]
    for e in q.index[q == 1]:
        k = "still in today's index" if e in today else (
            f"stopped trading ({delist.get(e, {}).get('event_type', 'no exit record')})" if e in ended.index
            else "left the index, still trading")
        fate[k] = fate.get(k, 0) + 1
        ret_by.setdefault(k, []).append(r[e])
n_q1 = sum(fate.values())
print(f"\nPoint-in-time Q1 holdings, 25 years: {n_q1} company-years, by status at end-2024")
for k, c in sorted(fate.items(), key=lambda x: -x[1]):
    print(f"  {k:34}{c:5} ({c / n_q1:5.1%}), mean return that year {np.mean(ret_by[k]):7.2%}")
v, q, r = info[T, 1999]
top = r[q.index[q == 1]].sort_values(ascending=False).head(5)
tick = px[px["m"] == pd.Period("1999-12", "M")].set_index("entity_id")["ticker"]
print("Today's roster, Q1 of December 1999: " + ", ".join(
    f"{tick[e]} ${v[e] / 1e9:.1f}bn {r[e]:+.0%}" for e in top.index) + " in 2000")
```

Full script with formatting and visualisation: [size-effect-survivorship-bias-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/size-effect-survivorship-bias-python.py)

## Output

![Annual return by size quintile, S&P 500 2000-2024: on point-in-time rosters the smallest fifth returned 11.0% a year and the largest 7.7%; on members still in today’s index 18.2% and 8.5%; on today’s roster 32.2% and 8.9%](/blog-images/size-effect-survivorship-bias-python.png)

```
25 December rosters 1999-2023, today's roster 502 lines, 1,098 securities, 430 exit records; holding years 2000-2024

universe                               names      Q1      Q2      Q3      Q4      Q5   Q1-Q5     t  Q1 ahead
Point-in-time rosters                    500  10.96%   9.75%  10.99%   8.24%   7.67%   +4.34  1.34     13/25
Members then, still in today's index     303  18.17%  14.48%  11.20%   8.70%   8.51%  +10.35  3.93     20/25
Today's roster                           421  32.24%  17.67%  13.86%  10.47%   8.94%  +24.74  6.05     25/25
today's roster minus point-in-time (total bias)   +20.40 points a year in Q1-Q5, paired t  7.46, ahead 24/25
  of which members since removed                   +6.01 points a year in Q1-Q5, paired t  5.19, ahead 22/25
  of which names not yet in the index             +14.38 points a year in Q1-Q5, paired t  5.13, ahead 24/25
  2000-2012 Q1-Q5 a year: Point-in-time rosters +11.14, Members then +15.67, Today's roster +32.31
  2013-2024 Q1-Q5 a year: Point-in-time rosters -3.03, Members then +4.59, Today's roster +16.53

Smallest fifth (Q1), averaged over the 25 sort dates
  Point-in-time rosters                  median value $  5.6bn, not an S&P 500 member that December   0.0%, not in today's index  68.2%
  Members then, still in today's index   median value $  7.6bn, not an S&P 500 member that December   0.0%, not in today's index   0.0%
  Today's roster                         median value $  4.4bn, not an S&P 500 member that December  79.3%, not in today's index   0.0%

Point-in-time Q1 holdings, 25 years: 2500 company-years, by status at end-2024
  stopped trading (merger)            808 (32.3%), mean return that year  13.52%
  still in today's index              794 (31.8%), mean return that year  23.24%
  left the index, still trading       764 (30.6%), mean return that year   8.50%
  stopped trading (dropped)           134 ( 5.4%), mean return that year -15.17%
Today's roster, Q1 of December 1999: LH $0.5bn +377%, VRTX $0.9bn +309%, AAS $0.7bn +233%, RE $1.0bn +223%, UHS $1.0bn +210% in 2000
```

## Where the 20-point gap comes from

On the rosters of the time the size effect is weak. The smallest fifth returned 10.96% a year against 7.67% for the largest, but the yearly gap of 4.34 points has a t-statistic of 1.34 and favoured the small end in only 13 of 25 years: 11.14 points in 2000-2012, then -3.03 in 2013-2024, when the largest members led.

Dropping the members that later left the index lifts the smallest fifth to 18.17% a year and the gap to 10.35 points (t 3.93). On the point-in-time rosters, 68.2% of the smallest fifth were companies no longer in the index. Of its 2,500 company-years, the 794 from companies still in the index returned 23.24% on average in the year held; the 764 from companies that had left the index but still traded at the end of 2024 returned 8.50%, the 808 later taken over in a merger 13.52%, and the 134 later dropped from their exchange -15.17%.

Today’s roster adds the larger part. On an average sort date, 79.3% of its smallest fifth were not S&P 500 members at all, and the group’s median value, $4.4 billion, was below the real smallest fifth’s $5.6 billion. In December 1999 it held LabCorp at $0.5 billion, which returned 377% in 2000, and Vertex (309%), AmeriSource (233%), Everest Re (223%) and Universal Health Services (210%); none was then a member. LabCorp’s figure matches [published rankings for 2000](https://statmuse.com/money/ask/best-performance-stocks-in-the-sandp-500-from-jan-2000-to-dec-2000), which put Universal Health at 209%. That smallest fifth compounds at 32.24% a year, and its gap exceeds the point-in-time one by 20.40 points (paired t 7.46, larger in 24 of 25 years).

## Testing a small-cap tilt on the rosters investors actually held

A size tilt inside the S&P 500 earned a modest, statistically unreliable premium in 2000-2024 and a negative one after 2012. A backtest on today’s members reports a premium 5.7 times as large that was positive in all 25 years; a result that clean is the mark of a sample chosen after the fact.

Keeping only companies that were members on the sort date is not enough: about 6 points a year remain unless members since removed, acquired or delisted stay in. Sort the roster as it stood on each date, which the [historical S&P 500 constituents](/historical-sp500-constituents) page lists back to 1925; [an analysis of how small members get before removal](/blog/sp500-removals-market-cap-at-removal-python) follows the companies that leave from the bottom of the index.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
