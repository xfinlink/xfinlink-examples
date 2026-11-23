# Altman Z-Score Backtest: Survivorship Bias in the S&P 500

November 23, 2026 · INDEX-UNIVERSE

Description: Altman Z-score survivorship bias, S&P 500 2000-2024: today’s roster adds 8.53 points a year to distressed stocks and 6.35 to safe ones (spread +2.18, t 1.52).

Survivorship bias adds 8.53 percentage points a year to the distressed stocks in an Altman Z-score backtest of the S&P 500 over 2000-2024 (t 5.13, larger in 21 of 25 years), and 6.35 points to the safe ones. The distress-minus-safe gap moves 2.18 points (t 1.52), and neither roster shows a significant distress premium (+0.52 on the rosters of the time, +2.69 on today’s). Set aside utilities, pipelines, power producers and waste services (SIC 4900-4999), and the bias in the distress zone rises to 15.62 points, producing a 9.49-point distress premium on today’s roster (t 2.58) against 0.20 on the real rosters.

## Why a distress sort on today’s members is a sort on recoveries

Edward Altman published the Z-score in 1968: five weighted ratios fitted on [66 publicly held manufacturers](https://en.wikipedia.org/wiki/Altman_Z-score), half of them bankrupt. Below 1.81 is the distress zone, above 2.99 the safe zone. [Ilia Dichev (1998)](https://www.econbiz.de/Record/is-the-risk-of-bankruptcy-a-systematic-risk-dichev-ilia/10005302353) found that US firms with high bankruptcy risk earned lower returns.

Today’s member list builds in hindsight: a distressed company of 2002 that is in the index today recovered, and the failures are absent. [A book-to-market backtest on the same rosters](/blog/value-investing-backtest-survivorship-bias-python) makes the comparison for value.

## Scoring 25 December rosters with the Altman Z-score

1. Rosters: the S&P 500 on the last business day of each December 1999-2023, from `xfl.index("sp500", as_of=...)`, and today’s roster of 23 November 2026.
2. Score: Altman’s original Z = 1.2 × working capital / total assets + 1.4 × retained earnings / total assets + 3.3 × EBIT / total assets + 0.6 × market value of equity / total liabilities + 1.0 × sales / total assets, read as `altman_z_score` from the latest annual report filed by the sort date (`as_of`; [point-in-time reference](/docs/point-in-time)), market value at the fiscal year-end close. The report is the latest with all five inputs, ended within 18 months.
3. Screens use each company’s SIC code on record today, except where that code does not fit its business on the sort date (REIT conversions such as American Tower’s, and successor records), classified by its business then. Financials (SIC 6000-6999) are left out, as the model is not recommended for them; SIC 4900-4999 companies stay in the main run and are set aside in a second.
4. Three universes: the roster in force, its members still in the index today, and today’s roster scored on the same date.
5. Each zone is held the next calendar year, equal weight; an exit takes its recorded delisting return, then cash. D-S is the distress zone’s yearly return minus the safe zone’s, averaged over 25 years.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

forms = pd.date_range("1999-12-01", "2023-12-31", freq="BYE")  # last business day of December
ymd = lambda d: d.strftime("%Y-%m-%d")
chunks = lambda xs: [xs[i:i + 50] for i in range(0, len(xs), 50)]

def z_rows(d, tickers=None, issuers=None):
    """Annual Altman Z rows from statements filed on or before d."""
    kw = dict(period_type="annual", fields=["altman_z_score"], as_of=ymd(d),
              start=ymd(d - pd.DateOffset(years=2)), end=ymd(d))
    out = [xfl.metrics(c, **kw) for c in chunks(tickers)] if tickers is not None else \
          [xfl.metrics(entity_id=c, **kw) for c in chunks(issuers)]
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# Today's roster: security ids for prices, issuer ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
tm = pd.concat([xfl.metrics(c, period_type="annual", fields=["altman_z_score"], start="2024-06-01")
                for c in chunks(today["ticker"].tolist())], ignore_index=True)
tsec = dict(zip(today["ticker"], today["entity_id"].astype(int)))
today_sec = tm.drop_duplicates("entity_id").set_index("entity_id")["ticker"].map(tsec).dropna().astype(int).to_dict()

# Each December: Z for the roster in force, and for today's companies outside it
rows = []
for d in forms:
    ro = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = z_rows(d, tickers=ro["ticker"].tolist()).assign(pit=True)
    a["sec"] = a["ticker"].map(dict(zip(ro["ticker"], ro["entity_id"].astype(int))))
    b = z_rows(d, issuers=sorted(set(today_sec) - set(a["entity_id"])))
    if len(b):
        b = b.assign(pit=False, sec=b["entity_id"].map(today_sec))
    rows.append(pd.concat([a, b]).assign(form=d))
Z = pd.concat(rows, ignore_index=True)
Z["altman_z_score"] = pd.to_numeric(Z["altman_z_score"])
Z = Z.dropna(subset=["altman_z_score", "sec"])
Z = Z[(Z["form"] - Z["period_end"]).dt.days <= 548]               # fiscal year ended within 18 months
Z = Z.sort_values("period_end").drop_duplicates(["form", "entity_id"], keep="last")
Z["sec"], Z["y"] = Z["sec"].astype(int), Z["form"].dt.year
Z["today"] = Z["entity_id"].isin(set(today_sec))
Z["zone"] = np.select([Z["altman_z_score"] < 1.81, Z["altman_z_score"] > 2.99], ["distress", "safe"], "grey")

# SIC code per company; financials (6000-6999) leave the sample
sic = {}
for d, g in Z.drop_duplicates("entity_id").groupby("form"):
    for c in chunks(sorted(g["entity_id"])):
        f = xfl.fundamentals(entity_id=c, period_type="annual", fields=["sic"], as_of=ymd(d),
                             start=ymd(d - pd.DateOffset(years=2)), end=ymd(d))
        sic.update(f.dropna(subset=["sic"]).drop_duplicates("entity_id").set_index("entity_id")["sic"].to_dict())
Z["sic"] = pd.to_numeric(Z["entity_id"].map(sic), errors="coerce")
# Classified by their business on the sort date where it differs from today's code (REIT conversions, successor records)
NOT_FINANCIAL_THROUGH = {83480: 2011, 32228: 2017, 28515: 2011, 28947: 2013, 26005: 2013, 30242: 2014, 29347: 2015, 5950: 2005}
Z = Z[~(Z["sic"].between(6000, 6999) & ~(Z["y"] <= Z["entity_id"].map(NOT_FINANCIAL_THROUGH)))]

# Monthly total returns 2000-2024, each exit's recorded delisting return added to its last month
ids = sorted(Z["sec"].unique())
px = pd.concat([xfl.prices(entity_id=c, start="1999-12-01", end="2024-12-31", interval="1mo",
                           fields=["return_daily", "close"], max_rows=100_000) for c in chunks(ids)])
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("1999-12", "2024-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(columns=ids).notna()
Z = Z[[traded.at[pd.Period(f"{y}-12", "M"), s] for y, s in zip(Z["y"], Z["sec"])]]  # priced at the sort
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    for c in chunks(g.index.tolist()):
        d = xfl.prices(entity_id=c, start=f"{y}-01-01", end="2024-12-31", interval="1mo", fields=["return_daily"])
        for e, v in (d.attrs["meta"].get("delisting") or {}).items():
            e, dr = int(e), v.get("delisting_return")
            if dr is not None and not np.isnan(dr):
                m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
                if m <= R.index[-1]:
                    R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def held(secs, y):
    """Equal weight in January of y+1, bought and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12", sorted(secs)]
    return (1 + r.fillna(0)).prod() - 1

U = {"Point-in-time rosters": Z["pit"], "Members then, still in today's index": Z["pit"] & Z["today"],
     "Today's roster": Z["today"]}
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
for label, keep in [("All non-financial", pd.Series(True, index=Z.index)), ("SIC 4900-4999 set aside", ~Z["sic"].between(4900, 4999))]:
    for k, mask in U.items():
        ds = []
        for y in range(1999, 2024):
            g = Z[mask & keep & (Z["y"] == y)]
            r = g["sec"].map(held(set(g["sec"]), y))
            ds.append(r[g["zone"] == "distress"].mean() - r[g["zone"] == "safe"].mean())
        print(label, k, f"D-S {np.mean(ds) * 100:+.2f} t {tstat(pd.Series(ds)):.2f}")
```

Full script with formatting and visualisation: [altman-z-score-survivorship-bias-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/altman-z-score-survivorship-bias-python.py)

## Output

![Annual return by Altman Z zone, S&P 500 2000-2024, three universes: in the main run the distress zone returned 10.0% a year on point-in-time rosters and 18.1% on today’s roster, the safe zone 9.7% and 16.0%, a distress-minus-safe gap of 0.5 and 2.7 points; with SIC 4900-4999 set aside the gap is 0.2 and 9.5 points](/blog-images/altman-z-score-survivorship-bias-python.png)

```
25 December rosters 1999-2023, today's roster 502 lines (2026-11-23), 783 securities, 283 exit records; holding years 2000-2024
10,607 company-years with a Z-score; financial (SIC 6000-6999) 406 set aside, no SIC on record 33 kept

All non-financial members
universe                               names  in D  distress    grey    safe    D-S     t  D ahead
Point-in-time rosters                    329    72     9.97%  10.92%   9.73%  +0.52  0.25    12/25
Members then, still in today's index     206    46    12.68%  13.69%  12.14%  +0.51  0.22    12/25
Today's roster                           284    62    18.08%  16.68%  15.97%  +2.69  0.99    12/25
today's roster minus point-in-time, distress zone    +8.53 points a year, paired t  5.13, ahead 21/25
  of which members since removed                     +2.31 points a year, paired t  1.86, ahead 19/25
  of which names not yet in the index                +6.22 points a year, paired t  3.65, ahead 20/25
today's roster minus point-in-time, safe zone        +6.35 points a year, paired t  8.25, ahead 24/25
today's roster minus point-in-time, D-S              +2.18 points a year, paired t  1.52, ahead 16/25
Lowest-Z fifth minus highest-Z fifth: Point-in-time rosters +0.61 (t 0.23), Members then +0.91 (t 0.27), Today's roster +0.65 (t 0.21)

Utilities, pipelines, power and waste (SIC 4900-4999) set aside
universe                               names  in D  distress    grey    safe    D-S     t  D ahead
Point-in-time rosters                    298    43     8.76%  10.63%   9.75%  +0.20  0.08    13/25
Members then, still in today's index     181    21    14.19%  13.35%  12.15%  +2.45  1.29    15/25
Today's roster                           253    34    23.41%  16.48%  15.98%  +9.49  2.58    15/25
today's roster minus point-in-time, distress zone   +15.62 points a year, paired t  4.55, ahead 21/25
  of which members since removed                     +4.55 points a year, paired t  2.38, ahead 21/25
  of which names not yet in the index               +11.07 points a year, paired t  3.16, ahead 19/25
today's roster minus point-in-time, safe zone        +6.34 points a year, paired t  8.18, ahead 24/25
today's roster minus point-in-time, D-S              +9.29 points a year, paired t  3.04, ahead 19/25
Lowest-Z fifth minus highest-Z fifth: Point-in-time rosters +1.54 (t 0.54), Members then +2.16 (t 0.81), Today's roster +4.07 (t 1.58)

Point-in-time zones, 8,235 company-years
  distress  1797 company-years, SIC 4900-4999 40.1%, not in today's index 36.6%, stopped trading in the year held 2.95% (dropped from exchange 4)
  grey      1882 company-years, SIC 4900-4999  2.3%, not in today's index 48.0%, stopped trading in the year held 2.76% (dropped from exchange 1)
  safe      4556 company-years, SIC 4900-4999  0.3%, not in today's index 33.3%, stopped trading in the year held 1.71% (dropped from exchange 2)
  distress zone, mean return in the year held: still in today's index 14.3% (1139), left the index, still trading 8.7% (263), stopped trading later 9.2% (395)
  safe zone, mean return in the year held: still in today's index 13.9% (3040), left the index, still trading 3.0% (528), stopped trading later 6.9% (988)
Dropped from their exchange during the year held: U Z 1.03 (distress, Dec 2001) -92%; KM Z 3.51 (safe, Dec 2001) -95%; CPN Z 0.67 (distress, Dec 2004) -94%; DAL Z 0.33 (distress, Dec 2004) -92%; DCN Z 1.72 (distress, Dec 2005) -89%; MERQ Z 2.65 (grey, Dec 2005) -1%; CC Z 4.86 (safe, Dec 2007) -97%
Enron, December 2000: Z 2.43 from the 1999 annual report (grey zone), -99% in 2001
Today's roster, distress zone without SIC 4900-4999, mean return in the year held: members at the sort 16.8% (532 company-years), not yet members 44.2% (314 company-years)
  largest: SBAC (Z 0.20, Dec 2002) +817% in 2003, APP (Z 0.95, Dec 2023) +713% in 2024, AKAM (Z -28.35, Dec 2002) +522% in 2003, EQIX (Z -1.06, Dec 2002) +395% in 2003
```

## Where the distress premium on today’s roster comes from

In the main run the distress zone compounds at 9.97% a year on the real rosters and 18.08% on today’s, the safe zone at 9.73% and 15.97%. Of the 8.53-point average yearly gap, 6.22 come from companies not yet in the index and 2.31 from members since removed.

The SIC 4900-4999 group, mostly electric and gas utilities, supplied 40.1% of the real distress zone and 0.3% of the safe zone, since heavy debt and low sales relative to assets pull these scores down. Without it, the distress zone returns 8.76% a year on the real rosters and 23.41% on today’s, against 9.75% and 15.98% for the safe zone.

There, companies not yet in the index supply 11.07 of the 15.62 points; their 314 distressed company-years averaged 44.2% in the year held, against 16.8% for members. AppLovin scored 0.95 in December 2023 and returned 713% in 2024.

Members since removed supply the other 4.55. In the main run’s real distress zone, 36.6% of company-years belong to companies no longer in the index; those that left but still trade returned 8.7% in the year held, against 14.3% for those still in it. Four distress-zone holdings, US Airways among them, were delisted in the year held. The zones missed others: Enron scored 2.43, grey, on its 1999 annual report and lost 99% in 2001, and Kmart scored 3.51, safe, in December 2001 and filed for bankruptcy in January 2002.

## Running a distress screen on the rosters of the time

On the rosters investors held, Altman’s distress zone earned no significant premium over the safe zone in either run. In the main run, today’s list adds 6 to 9 points a year to both zones, overstating any long-only distress backtest. Only without SIC 4900-4999 does it also show a significant distress premium, 9.49 points a year at the 5% level, built on hindsight about membership.

Sort the roster in force on each date; the [historical S&P 500 constituents](/historical-sp500-constituents) page lists it back to 1925, and [a study of how removed members traded before their exit](/blog/stocks-removed-from-sp500-before-removal-python) follows the companies that leave. [The guide to getting an Altman Z-score in Python](/blog/where-to-get-an-altman-z-score) covers scoring one company.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
