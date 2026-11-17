# S&P 500 Profitability Requirement: Do New Additions Pass?

November 17, 2026 · INDEX-UNIVERSE

Description: S&P 500 profitability requirement tested on 341 new additions since 2011: 90% of established listings passed on figures filed by the day before joining.

Nine in ten S&P 500 additions that had traded for at least a year passed the index's profitability rule on the figures filed by the day before joining: 242 of 269 since 2011. Of the 19 that failed, 17 had moved up from the S&P MidCap 400 or SmallCap 600, a route the methodology has expressly exempted since July 2017. Passing did not protect new members afterwards: 14.1% reported a trailing four-quarter loss within three years, against 16.8% of incumbents that passed the same test, and additions that lost money were removed within five years at about three times the rate of the rest (10 of 26 against 18 of 153).

## Do S&P 500 companies have to be profitable?

Outside companies do. S&P Dow Jones Indices' [2026 consultation results](https://press.spglobal.com/2026-06-04-S-P-Dow-Jones-Indices-Consultation-on-Treatment-of-MegaCap-Companies-Results) state the rule: GAAP net income from continuing operations must be positive for the most recent quarter and for the sum of the most recent four consecutive quarters. Migrations between the S&P 500, MidCap 400 and SmallCap 600 are exempt (written into the methodology in [July 2017](https://press.spglobal.com/2017-07-31-S-P-Dow-Jones-Indices-Announces-Changes-to-the-S-P-U-S-Indices-Methodology-and-to-the-Pricing-of-Bankrupt-U-S-Listed-Securities)), as are spin-offs from current constituents, and a company that acquires a constituent with stock may be added at the committee's discretion. Until 21 March 2014 the [methodology](https://press.spglobal.com/2014-02-24-S-P-U-S-Indices-Methodology-Update) "usually measured" viability as four consecutive quarters of positive as-reported earnings.

The question is how often additions cleared the rule on what was public at the time, and what followed.

## Rebuilding each addition's earnings as filed the day before

1. Take every addition from 2011 to September 2026 from `xfl.index_events`: 348, less 7 extra share classes of existing members. Starting in 2011 is a methodology choice that keeps every test on one kind of quarterly filing record.
2. Request quarterly net income with `as_of` set to the day before the effective date, so only figures made public by then count ([point-in-time reference](/docs/point-in-time)). The test uses net income attributable to the parent, not continuing operations alone.
3. Pass: the four most recent consecutive quarters sum to more than zero and the latest is positive. Without four recent quarters on file, a company is not testable.
4. Split additions by whether the stock had traded for 12 months before joining; shorter histories are spin-offs and merger-created companies. Each failure's route in comes from S&P Dow Jones Indices' announcements.
5. After entry, flag a trailing four-quarter loss in first-reported figures within one, two and three years. The control is every member on each 31 December roster with a year of tenure that passed the test that day, weighted to the additions' years.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
NI = ["net_income", "net_income_attributable_to_parent"]


def ttm_series(f):
    """Trailing four-quarter net income at each quarter end, consecutive quarters only."""
    q = f.assign(ni=f[NI[1]].fillna(f[NI[0]]), pri=(f["source"] != "filing").astype(int))
    keep = []
    for r in q.sort_values(["period_end", "pri"]).itertuples():   # one row per quarter, filings first
        if keep and (r.period_end - keep[-1].period_end).days < 45:
            continue
        keep.append(r)
    q = pd.DataFrame(keep).dropna(subset=["ni"])
    out = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3:i + 1]
        if 240 <= (w["period_end"].iloc[-1] - w["period_end"].iloc[0]).days <= 300:
            out.append((w["period_end"].iloc[-1], w["ni"].sum(), w["ni"].iloc[-1]))
    return pd.DataFrame(out, columns=["pe", "ttm", "lastq"])


def earnings_test(f, known):
    t = ttm_series(f) if len(f) else pd.DataFrame()
    if len(t) == 0 or (known - t["pe"].iloc[-1]).days > 230:
        return "untested", np.nan, np.nan
    x = t.iloc[-1]
    return ("pass" if x.ttm > 0 and x.lastq > 0 else "fail"), x.ttm, x.lastq


# Dated additions, one year per call (each checked against meta.total)
ev = pd.concat([xfl.index_events("sp500", start=f"{y}-01-01", end=f"{y}-12-31" if y < 2026 else None)
                for y in range(2011, 2027)])
add = ev[ev["event_type"] == "added"].assign(d=lambda x: pd.to_datetime(x["effective_date"]))
add = add[add["d"] <= "2026-09-30"]
add["known"] = add["d"] - pd.Timedelta(days=1)

# Quarterly net income as filed by the day before joining, one call per effective date
rows = []
for known, g in add.groupby("known"):
    try:
        f = xfl.fundamentals(sorted(set(g["ticker"])), period_type="quarterly", fields=NI,
                             as_of=known.strftime("%Y-%m-%d"),
                             start=(known - pd.DateOffset(months=30)).strftime("%Y-%m-%d"))
    except xfl.XfinlinkError:        # no company under the ticker that day (a when-issued spin-off)
        f = pd.DataFrame()
    for a in g.itertuples():
        status, ttm, lastq = earnings_test(f[f["ticker"] == a.ticker] if len(f) else f, known)
        rows.append(dict(ticker=a.ticker, d=a.d, sec=a.entity_id, status=status, ttm=ttm, lastq=lastq,
                         issuer=f.loc[f["ticker"] == a.ticker, "entity_id"].max() if len(f) else np.nan))
S = pd.DataFrame(rows)

# Later losses on first-reported quarters
ids = sorted(set(S["issuer"].dropna().astype(int)))
H = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="quarterly", version="original",
                                start="2010-01-01", fields=NI, max_rows=50_000) for i in range(0, len(ids), 100)])
TT = {e: ttm_series(g) for e, g in H.groupby("entity_id")}
ok = S[S["status"] == "pass"].copy()
for n in (1, 2, 3):
    ok[f"loss{n}"] = [None if d + pd.DateOffset(years=n) > pd.Timestamp("2026-06-30") else
                      bool(((TT[e]["pe"] > d) & (TT[e]["pe"] <= d + pd.DateOffset(years=n)) & (TT[e]["ttm"] < 0)).any())
                      for e, d in zip(ok["issuer"].astype(int), ok["d"])]
    print(n, ok[f"loss{n}"].dropna().mean())   # 6.1%, 8.2%, 13.0% here; the full script's share-class and issuer checks give the Output figures
# The full script adds the share-class filter, the 12-month listing split, issuer checks for
# spin-offs, the 31 December incumbent control, restated-figure re-runs and removals.
```

Full script with formatting and visualisation: [sp500-additions-profitability-rule-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-additions-profitability-rule-python.py)

## Output

![Left: S&P 500 additions 2011 to September 2026 by earnings test result; 242 of 269 established listings passed, 19 failed, 8 not testable; of 72 spin-offs and merger-created companies 10 passed, 4 failed, 58 not testable. Right: share reporting a trailing four-quarter loss within 1, 2 and 3 years, additions 6.6%, 8.8%, 14.1% against incumbents 6.9%, 12.0%, 16.8%](/blog-images/sp500-additions-profitability-rule-python.png)

```
Additions 2011-01-01 to 2026-09-30: 348; another share class of a member or no company record: 7; sample: 341
  check TSLA 2020-12-21: four-quarter net income 556m, latest quarter 331m -> pass
  check UBER 2023-12-18: four-quarter net income 1,053m, latest quarter 221m -> pass
  check ABNB 2023-09-18: four-quarter net income 2,300m, latest quarter 650m -> pass
Passed on figures filed by the day before joining: 252 of 341 (73.9%)
  Traded 12+ months before joining: 269  pass 242 (90.0%)  fail 19  not testable 8
  Traded under 12 months (spin-offs, merger-created): 72  pass 10 (13.9%)  fail 4  not testable 58
Passers that also had four positive quarters (the pre-2014 test): before 24 Mar 2014 36 of 39, since 168 of 203

Established listings that failed (net income, $m)
    joined ticker 4 quarters latest             route
2011-12-13    AGL        203     -3   from MidCap 400
2013-09-23   VRTX       -499    -57   from MidCap 400
2014-07-02    MLM        128    -22   from MidCap 400
2014-08-19    MNK         67    -24   from MidCap 400
2015-01-27   ENDP     -1,444   -252   from MidCap 400
2015-03-23   EQIX       -260   -355   from MidCap 400
2016-10-03   COTY        157    -31  outside addition
2017-03-20    AMD       -497    -51   from MidCap 400
2017-03-20    ARE        -66     15   from MidCap 400
2018-03-19   NKTR        -97    -34 from SmallCap 600
2018-08-28   ANET        227   -155  outside addition
2021-03-22    CZR     -1,757   -555   from MidCap 400
2021-03-22   PENN       -670     11   from MidCap 400
2021-09-20   CDAY        -63    -26   from MidCap 400
2022-10-03    EQT       -803    891   from MidCap 400
2022-12-19   FSLR         95    -49   from MidCap 400
2024-05-08    VST      1,493   -184   from MidCap 400
2025-03-24    EXE       -714   -399   from MidCap 400
2026-03-23   SATS    -14,499 -1,207   from MidCap 400
New listings that failed: ADNT 2016, CTVA 2019, VTRS 2020, SNDK 2025

Verdicts that change on today's figures for the same quarters: 1 of 275: AMTM 2024 pass->fail
... on the four quarters that had ended (filed or not) before joining: 7: MLM 2014 fail->pass, WLTW 2016 pass->fail, CXO 2016 pass->fail, ALB 2016 pass->fail, KEYS 2018 pass->fail, EQT 2022 fail->pass, AMTM 2024 pass->fail

Trailing four-quarter loss reported within N years of joining (additions that passed at entry)
  1 year(s): additions 15/228 = 6.6%   incumbents 6.9% (6,303 member-years)   binomial p 1.000
  2 year(s): additions 19/217 = 8.8%   incumbents 12.0% (5,868 member-years)   binomial p 0.145
  3 year(s): additions 29/205 = 14.1%   incumbents 16.8% (5,434 member-years)   binomial p 0.350
  3 years, additions still in the index today only: 14/136 = 10.3%

Removed within 5 years of joining (additions to June 2021)
  no loss               153  removed 11.8%  (still trading after removal 4.6%, stopped trading 7.2%)
  loss within 3 years    26  removed 38.5%  (still trading after removal 26.9%, stopped trading 11.5%)
  Fisher exact p = 0.0018
```

## Most failures were promotions from the MidCap 400

Of the 19 failures that had traded for a year, 16 came up from the MidCap 400 and Nektar from the SmallCap 600; eight of those moves predate the July 2017 exemption. EchoStar had lost $14.5bn over four quarters. AGL Resources, a MidCap 400 member, replaced Nicor, which it had just acquired. The two outside additions that failed, Coty in 2016 and Arista Networks in 2018, had positive four-quarter sums and missed only on the latest quarter.

The 2014 change shows in the passers: before it, 36 of 39 also had four positive quarters; since, 35 of 203 carried a losing quarter.

Of the 72 spin-offs and merger-created companies, 58 could not be tested on four quarters of their own. Four failed, among them Corteva, added [alongside its spin-off from DowDuPont](https://press.spglobal.com/2019-05-28-Corteva-Set-to-Join-S-P-500-Fluor-to-Join-S-P-MidCap-400-and-Realogy-and-Bloomin-Brands-to-Join-S-P-SmallCap-600).

Timing mattered more than restatement. Today's figures for the same quarters change one verdict, that of Amentum (2024), formed by a merger the week it joined; counting the quarters that had ended, filed or not, changes seven, among them Concho Resources (2016) and Keysight (2018), which would fail on a quarter reported after they joined.

## Passing the rule did not mean staying profitable

Additions that passed reported trailing losses at about the incumbents' rate: 6.6% against 6.9% within one year, 8.8% against 12.0% within two, 14.1% against 16.8% within three, none significant (binomial p 0.145 to 1.000). On additions still in the index today alone, the three-year rate is 10.3%, so a study on today's roster understates it.

Of additions up to June 2021 that lost money within three years, 38.5% were removed within five years of joining, against 11.8% of the rest (Fisher exact p 0.0018), mostly while their shares kept trading (26.9% against 4.6%).

## Reading the rule when screening for the next additions

A screen for likely additions should apply the rule to figures filed by the announcement date, not to quarters that have merely ended, and should leave MidCap 400 and SmallCap 600 members out of the earnings filter. After entry, a past profit says little, but new members that reported a trailing loss within three years left within five at about three times the rate of other new members, on 26 cases. Past rosters, including members later removed, are on the [historical S&P 500 constituents](/historical-sp500-constituents) page, and [a study of new members' market value at entry](/blog/sp500-additions-market-cap-at-entry-python) sizes each addition on its first day.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
