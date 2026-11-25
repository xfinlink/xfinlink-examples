# S&P 500 Dual-Class Shares History: Two Lines, One Company

November 25, 2026 · INDEX-UNIVERSE

Description: S&P 500 dual-class shares history: ten companies held two index lines at once since 1980; a slot per line moves equal-weight returns 10.6 bp in affected years.

Ten companies have held two S&P 500 index lines at the same time since 1980, with up to five at once (December 2016-2021). An equal-weight backtest giving each line its own slot drifts from one giving each company one slot by 10.6 basis points a year on average over the 20 affected years, 38.5 at most; $1 invested over 1996-2025 ends at $22.26 per line against $22.22 per company.

## Why the S&P 500 can list one company twice

The S&P 500 counts 500 companies but holds listed stock lines, each with its own ticker and price: Alphabet’s Class A trades as GOOGL and its Class C as GOOG. The end-2025 roster has 503 lines for 500 companies.

In March 2014 S&P Dow Jones Indices said it would [keep both Google lines and include every eligible class line from September 2015](https://press.spglobal.com/2014-03-11-S-P-Dow-Jones-Indices-Announces-Changes-in-Treatment-of-Multiple-Share-Classes-in-U-S-Indices-and-Revises-Previously-Announced-Treatment-of-Google-Stock-Split), taking the index [from 502 trading lines to 505](https://press.spglobal.com/2015-08-10-S-P-U-S-Indices-Methodology-Update-and-Constituent-Additions). On 31 July 2017 it [stopped adding companies with multiple share class structures](https://press.spglobal.com/2017-07-31-S-P-Dow-Jones-Indices-Announces-Decision-on-Multi-Class-Shares-and-Voting-Rules), grandfathering existing members, and on 17 April 2023 it [announced that they were eligible again](https://www.davispolk.com/insights/client-update/sp-dow-jones-reopens-its-indices-companies-multiple-share-classes).

Dual-class voting alone does not add a line: Morningstar [counted 22 S&P 500 companies with dual class share structures](https://corpgov.law.harvard.edu/2025/02/11/shareholder-democracy-and-the-challenge-of-dual-class-share-structures/) in the 2024 proxy season. A backtest that weights roster rows equally gives a two-line company two slots, and only each year’s roster shows how many there were.

## Counting lines and companies on 46 December rosters

1. Rosters: the S&P 500 on the last business day of each December, 1980-2025, from `xfl.index("sp500", as_of=...)` ([index membership reference](/docs/index-membership)). Lines sharing a company name on one roster count as one company. Names opening with the same initials (U S X, I T T, A T & T) were screened and each match ruled on by hand; only USX’s two lines were joined.
2. Dated changes from `xfl.index_events` find two-line spells between year-ends.
3. Backtest: each December roster from 1995 to 2024 is bought at equal weight in January and held to December, once with a slot per line (1/N) and once with a slot per company (1/M, split across its lines). Monthly total returns come from `xfl.prices`; a delisted line takes its recorded delisting return, then cash.

## Code

```python
import re

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# S&P 500 roster on the last business day of each December, 1980-2025
days = pd.date_range("1980-12-01", "2025-12-31", freq="BYE")
ros = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in days}

# One company = index lines that share a name on the same roster, or that the override joins.
# Tracking stocks can carry different names, so lines whose names open with the same run of
# initials ("U S X", "I T T", "A T & T") are screened below and ruled on one by one.
SAME_COMPANY = {"U S X MARATHON GROUP": "U S X CORP",       # USX tracking stocks, 1991-05-07 to
                "U S X US STEEL GROUP INC": "U S X CORP"}   # the 2001-12-31 separation
PAIR_END = {"U S X CORP": "2001-12-31"}  # separation date; no index removal is recorded for either line
initials = lambda n: (m.group(1).replace(" ", "") if (m := re.match(r"((?:[A-Z&] )+[A-Z&]) ", n or "")) else None)
for r in ros.values():
    r["company"] = r["entity_name"].map(lambda n: SAME_COMPANY.get(n, n))
screen = {}
for y, r in ros.items():
    r = r.assign(stem=r["entity_name"].map(initials)).dropna(subset=["stem"])
    for st, g in r.groupby("stem"):
        if g["entity_name"].nunique() > 1:
            key = (st, " | ".join(sorted(set(g["entity_name"]))), "one company" if g["company"].nunique() == 1 else "separate")
            screen.setdefault(key, []).append(y)
count = pd.DataFrame({y: {"lines": len(r), "companies": r["company"].nunique(),
                          "multi": (r["company"].value_counts() > 1).sum()} for y, r in ros.items()}).T
pairs = {}
for y, r in ros.items():
    for name, g in r[r.duplicated("company", keep=False)].groupby("company"):
        pairs[frozenset(g["entity_id"])] = name  # latest name

# Dated additions and removals catch two-line spells that start and end between year-ends
ev = pd.concat([xfl.index_events("sp500", start="1980-01-01", end="2025-12-31", offset=o) for o in (0, 1000, 2000)])
add = ev[ev["event_type"] == "added"]
for name, g in add.groupby("entity_name"):
    ids = sorted(set(g["entity_id"]))
    for i in ids:
        for j in ids:
            if i < j and not any({i, j} <= set(k) for k in pairs):
                pairs[frozenset({i, j})] = name
spell = {}
for y, r in ros.items():
    spell.update({e: (a, d if isinstance(d, str) else None)
                  for e, a, d in r[["entity_id", "added_date", "removed_date"]].itertuples(index=False)})
for e, g in ev.groupby("entity_id"):
    if e not in spell:
        spell[e] = (g.loc[g["event_type"] == "added", "effective_date"].min(),
                    g.loc[g["event_type"] == "removed", "effective_date"].max())
both = []
for ids, name in pairs.items():
    a, b = sorted(ids)
    start = max(spell[a][0], spell[b][0])
    end = min(spell[a][1] or "9999", spell[b][1] or "9999", PAIR_END.get(name, "9999"))
    if start <= end:
        both.append({"company": name, "a": a, "b": b, "from": start, "to": end.replace("9999", "present")})
both = pd.DataFrame(both).sort_values("from")

# Monthly total returns for every line on the 1995-2024 rosters, delisting returns included
ids = sorted(set().union(*[set(ros[y]["entity_id"]) for y in range(1995, 2025)]))
px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start="1996-01-01", end="2025-12-31", interval="1mo",
                           fields=["return_daily", "close"], max_rows=500_000) for i in range(0, len(ids), 100)])
px["m"] = px["date"].dt.to_period("M")
R = px.drop_duplicates(["entity_id", "m"]).pivot(index="m", columns="entity_id", values="return_daily")
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2025-12", "M")]
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    secs = g.index.tolist()
    for c in [secs[i:i + 100] for i in range(0, len(secs), 100)]:
        d = xfl.prices(entity_id=c, start=f"{y}-01-01", end="2025-12-31", interval="1mo", fields=["return_daily"])
        for e, v in (d.attrs["meta"].get("delisting") or {}).items():
            e, dr = int(e), v.get("delisting_return")
            if dr is not None and not np.isnan(dr):
                m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
                if m <= R.index[-1]:
                    R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def held(w, y):
    """Weights w bought in January of y+1 and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12"].reindex(columns=w.index)
    return float((w * (1 + r.fillna(0)).prod()).sum() / w.sum() - 1)

res = []
for y in range(1995, 2025):
    r = ros[y]
    by_row = pd.Series(1.0, index=r["entity_id"])
    by_co = pd.Series((1 / r.groupby("company")["entity_id"].transform("size")).values, index=r["entity_id"])
    res.append({"year": y + 1, "row_ew": held(by_row, y), "company_ew": held(by_co, y)})
res = pd.DataFrame(res).set_index("year")
def label(e, start):
    """Ticker on the first year-end roster after the two lines overlap, else at the addition."""
    on = [r.set_index("entity_id")["ticker"] for y, r in sorted(ros.items()) if str(y) >= start[:4] and e in set(r["entity_id"])]
    return on[0][e] if on else add.loc[add["entity_id"] == e, "ticker"].iloc[-1]
both["lines"] = [f"{label(a, f)}/{label(b, f)}" for a, b, f in zip(both["a"], both["b"], both["from"])]
def corr(a, b, start, end):
    if a not in R or b not in R:
        return np.nan
    r = R.loc[pd.Period(start, "M") + 1:pd.Period(min(end, "2025-12"), "M") - 1, [a, b]].dropna()
    return r[a].corr(r[b]) if len(r) >= 12 else np.nan
both["monthly_corr"] = [corr(a, b, f, t) for a, b, f, t in zip(both["a"], both["b"], both["from"], both["to"])]
res["gap_bp"] = (res["row_ew"] - res["company_ew"]) * 1e4

for (st, names, ruling), ys in sorted(screen.items(), key=lambda kv: kv[1][0]):
    print(f"screen {st}: {names} ({ys[0]}-{ys[-1]}): {ruling}")
same = count[count["lines"] == count["companies"]]
print(f"{len(same)} of {len(count)} year-ends: lines == companies ({same['lines'].min()}-{same['lines'].max()})")
print(count.loc[count["lines"] != count["companies"]].T.to_string())
print(both[["company", "lines", "from", "to", "monthly_corr"]].round(3).to_string(index=False))
dup = res[res["gap_bp"] != 0]
print(dup.round(4).T.to_string())
growth = (1 + res[["row_ew", "company_ew"]]).prod()
print(f"{len(dup)} of {len(res)} holding years differ; mean gap {dup['gap_bp'].mean():+.1f} bp, "
      f"mean absolute {dup['gap_bp'].abs().mean():.1f} bp, largest {dup['gap_bp'].abs().max():.1f} bp in {dup['gap_bp'].abs().idxmax()}")
print(f"$1 in 1996-2025: row-weighted {growth['row_ew']:.2f}, company-weighted {growth['company_ew']:.2f}")
t = pd.concat(ros.values())
print("Berkshire and Brown-Forman lines on any roster:",
      sorted(set(t.loc[t["ticker"].str.match(r"^(BRK|BF)[-.]", na=False), "ticker"])))
```

Full script with formatting and visualisation: [sp500-dual-class-shares-history-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-dual-class-shares-history-python.py)

## Output

![Two-panel chart of S&P 500 December rosters 1980-2025: extra index lines (lines minus companies) are 0 until 1990, 1 from 1991 to 1994 and 2 from 1995 to 2001 for the USX, U S West and Sprint tracking stocks, 1 in 2002-2003, 0 from 2004 to 2013, then 2 in 2014, 4 in 2015, 5 from 2016 to 2021 and 3 from 2022 to 2025; below, the per-line minus per-company equal-weight return in the following year ranges from −19.8 to +38.5 basis points](/blog-images/sp500-dual-class-shares-history-python.png)

```
screen US: U S HOME CORP | U S WEST INC NEW (1983-1983): separate
screen US: U S HOME CORP | U S WEST INC (1984-1990): separate
screen USX: U S X MARATHON GROUP | U S X US STEEL GROUP INC (1991-2001): one company
screen ITT: I T T CORP NEV | I T T HARTFORD GROUP INC | I T T INDUSTRIES INC IND (1995-1996): separate
screen ITT: I T T CORP NEV | I T T INDUSTRIES INC IND (1997-1997): separate
screen US: U S AIRWAYS GROUP INC | U S BANCORP DEL | U S WEST INC (1997-1997): separate
screen US: U S AIRWAYS GROUP INC | U S BANCORP DEL | U S WEST INC NEW (1998-1999): separate
screen US: U S AIRWAYS GROUP INC | U S BANCORP DEL (2000-2001): separate
screen AT&T: A T & T CORP | A T & T WIRELESS SVCS INC (2001-2003): separate
screen FMC: F M C CORP | F M C TECHNOLOGIES INC (2009-2016): separate
screen GE: G E HEALTHCARE TECHNOLOGIES INC | G E VERNOVA INC (2024-2025): separate
21 of 46 year-ends: lines == companies (500-500)
           1991  1992  1993  1994  1995  1996  1997  1998  1999  2000  2001  2002  2003  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024  2025
lines       500   500   500   500   500   500   500   500   500   500   500   500   500   502   504   505   505   505   505   505   505   503   503   503   503
companies   499   499   499   499   498   498   498   498   498   498   498   499   499   500   500   500   500   500   500   500   500   500   500   500   500
multi         1     1     1     1     2     2     2     2     2     2     2     1     1     2     4     5     5     5     5     5     5     3     3     3     3
                     company       lines       from         to  monthly_corr
                  U S X CORP       MRO/X 1991-05-07 2001-12-31         0.270
                U S WEST INC     UMG/USW 1995-11-01 1998-06-12         0.240
                 SPRINT CORP     FON/PCS 1998-11-24 2004-04-22         0.669
                ALPHABET INC  GOOG/GOOGL 2014-04-03    present         0.997
               DISCOVERY INC DISCA/DISCK 2014-08-07 2022-04-08         0.986
               NEWS CORP NEW    NWSA/NWS 2015-09-21    present         0.988
TWENTY FIRST CENTURY FOX INC    FOXA/FOX 2015-09-21 2019-03-19         0.994
            COMCAST CORP NEW CMCSA/CMCSK 2015-09-21 2015-12-11           NaN
            UNDER ARMOUR INC      UA/UAA 2016-04-08 2022-06-17         0.987
                    FOX CORP    FOXA/FOX 2019-03-19    present         0.989
year          1996    1997    1998     1999     2000    2001    2002    2003    2004    2015     2016     2017     2018    2019    2020    2021     2022     2023     2024     2025
row_ew      0.1923  0.2833  0.1392   0.1377   0.1046  0.0047 -0.1805  0.4131  0.1695 -0.0183   0.1376   0.1880  -0.0698  0.2870  0.0962  0.2904  -0.1047   0.1479   0.1311   0.1168
company_ew  0.1928  0.2829  0.1393   0.1338   0.1066  0.0044 -0.1797  0.4135  0.1685 -0.0188   0.1387   0.1892  -0.0713  0.2876  0.0971  0.2909  -0.1034   0.1469   0.1295   0.1153
gap_bp     -5.1652  3.9978 -0.9340  38.4519 -19.8382  3.4413 -8.8409 -3.6869  9.2974  5.0723 -11.8670 -11.8081  15.4773 -5.5635 -8.5550 -4.8253 -12.9327  10.3731  15.9724  15.0126
20 of 30 holding years differ; mean gap +1.2 bp, mean absolute 10.6 bp, largest 38.5 bp in 1999
$1 in 1996-2025: row-weighted 22.26, company-weighted 22.22
Berkshire and Brown-Forman lines on any roster: ['BF-B', 'BRK-B']
```

## Tracking stocks in the 1990s, share classes after 2014

The roster has 500 lines for 500 companies every December of 1980-1990 and 2004-2013. In between, tracking stocks (classes of one company’s stock that each follow one business) cut the count. USX’s Marathon and U.S. Steel lines ran from [7 May 1991](https://sec.gov/Archives/edgar/data/101778/000010177801500013/ex991424.txt) to [the separation on 31 December 2001](https://www.sec.gov/Archives/edgar/data/0000101778/000095017202000013/s273688.txt); U S West [had two such stocks from 1 November 1995 to 12 June 1998](https://www.sec.gov/Archives/edgar/data/0001054522/000104746999010985/0001047469-99-010985.txt); Sprint’s FON and PCS from their [creation in November 1998](https://www.sec.gov/Archives/edgar/data/0000101830/000095013199001360/0000950131-99-001360.txt) to their [recombination on 23 April 2004](https://www.sec.gov/Archives/edgar/data/0000101830/000119312505047941/d10k.htm). That leaves 499 companies in 1991-1994 and 2002-2003, and 498 in 1995-2001. The pairs’ monthly returns correlated at only 0.270 (USX, measured over 1996-2001), 0.240 and 0.669.

From 2014 the extra lines are share classes. Google’s Class C joined on 3 April 2014 and Discovery’s on 7 August 2014. News Corp, Twenty-First Century Fox and Comcast added second lines on 21 September 2015; Comcast’s ended on 11 December 2015, when it [folded its Class A Special shares into Class A](https://corporate.comcast.com/news-information/news-feed/comcast-shareholders-approve-the-reclassification-of-each-share-of-class-a-special-common-stock-into-one-share-of-class-a-common-stock). Under Armour’s second line, from 8 April 2016, lifted the count to 505 through 2021, and Fox Corp kept both lines in March 2019 as [the surviving entity of Twenty-First Century Fox](https://press.spglobal.com/2019-03-14-Fox-Set-to-Join-S-P-500-Adobe-to-Join-S-P-100). Discovery left on 8 April 2022, the day its [merger with WarnerMedia completed](https://www.upi.com/Top_News/US/2022/04/08/warner-bros-discovery-merger-completed-leadership-shakeup/3531649471665/), and Under Armour on 17 June 2022, ahead of its [move to the S&P MidCap 400](https://press.spglobal.com/2022-06-03-Keurig-Dr-Pepper,-VICI-Properties-and-ON-Semiconductor-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400,-and-S-P-SmallCap-600).

Share-class lines move almost as one (correlations 0.986 to 0.997), so a second slot is a 0.2-percentage-point overweight in one company, and the yearly gap is that overweight times its return against the rest of the roster. Share-class years therefore stay between −12.9 and +16.0 basis points. The extremes, −19.8 in 2000 and +38.5 in 1999, came from rosters holding the USX and Sprint tracking stocks, whose lines moved far less in step.

## Equal weight by company, prices by line

In an equal-weight S&P 500 backtest the duplicate-line error is small and cheap to remove. Group roster lines by company before weighting or counting members, and keep each line’s own `entity_id` for prices; cap weighting needs no change, since each line carries only its own shares. A year-end grid misses spells such as Comcast’s; the [historical S&P 500 constituents](/historical-sp500-constituents) page gives the roster for any day back to 1925. [A year-by-year count of members by listing exchange](/blog/sp500-companies-nasdaq-vs-nyse-history-python) counts members on year-end rosters with one line per share class, and [a comparison of cap-weighted and equal-weight backtests](/blog/survivorship-bias-cap-weighted-vs-equal-weight-python) covers weighting and survivorship together.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
