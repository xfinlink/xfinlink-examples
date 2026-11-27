# S&P 500 Stock Splits by Year: 105 at the Peak, 1 in 2009

November 27, 2026 · INDEX-UNIVERSE

Description: S&P 500 stock split count by year since 1980: 105 members split in 1987, 1 in 2009, 16 in 2024; today's roster gives a higher split rate in 44 of the 46 years.

S&P 500 members split their stock most often in 1986 and 1987, 105 companies in each year or one member in five, and least often in 2009, when one company did. On the roster in force on each ex-date, splits averaged 58 a year in the 1980s and 1990s, 26 in the 2000s and 8 from 2010 to 2025. Today's roster traced back gives a higher split rate in 44 of the 46 years, by 3.0 percentage points on average.

## How often do S&P 500 companies split their stock?

A stock split gives holders more shares at a proportionally lower price: in a 2-for-1 split one $100 share becomes two $50 shares. Companies tend to split after a large rise in the share price.

A split belongs to the index's history only if the company was a member on the ex-date, the first day the shares trade on the new basis. Microsoft split five times before joining in June 1994. Today's member list counts such splits and drops the companies removed since: survivorship bias, counting only the companies that lasted.

## Matching every split to the roster on its ex-date

1. Membership: S&P 500 rosters on the last business day of each December 1979-2025 and today, from `xfl.index("sp500", as_of=...)`, plus dated changes from `xfl.index_events` for spells between year-ends. First and last days both count. Lines sharing a name on one roster (GOOG and GOOGL) are one company.
2. Splits: monthly bars for 1,592 securities from `xfl.prices` ([prices reference](/docs/prices)). A recorded `split_ratio` counts when a second signal confirms it: the step in `close` over `adj_close` within 3%, or the change in shares outstanding within 15%. The factor agreed on 3,339 of 3,370 candidates.
3. Definitions: a forward split is a plain ratio (3-for-2, 4-for-1) of 1.25 or more, which excludes small stock dividends and 25 merger or recapitalisation ratios. A reverse split is a plain ratio of 0.8 or less, confirmed by the factor or, where a spin-off fell on the same day, by the company's filing. Share dividends filed as splits count: Google's [two-for-one split effected as a stock dividend](https://www.sec.gov/Archives/edgar/data/0001652044/000165204416000022/exhibit992.htm) in April 2014, Discovery's [share dividend of August 2014](https://www.sec.gov/Archives/edgar/data/1437107/000143710715000004/disca-2014123110k.htm), Under Armour's [Class C dividend of 2016](https://www.sec.gov/Archives/edgar/data/0001336917/000133691718000009/ua-20171231x10k.htm) and Wrigley's [5-for-4 Class B dividend of 2006](https://www.sec.gov/Archives/edgar/data/0000108601/000095013707002108/c12224exv13.htm).
4. Where a member joined or left in the split's month, a daily pull fixes the ex-date.
5. Rates divide splitting members by companies in the index at any time that year; for today's roster, by its companies trading that year.

## Code

```python
import pandas as pd
import xfinlink as xfl
from fractions import Fraction

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Membership spells: December rosters, today's roster, and short spells from the event log
days = pd.date_range("1979-12-01", "2025-12-31", freq="BYE")
ros = [xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in days]
today = xfl.index("sp500")
ev = pd.concat([xfl.index_events("sp500", start="1979-01-01", end="2026-11-26", limit=1000, offset=o)
                for o in (0, 1000, 2000)], ignore_index=True)
sp = pd.concat(ros + [today]).drop_duplicates(["entity_id", "added_date"])
seen = set(zip(sp.entity_id, sp.added_date))
for a in ev[(ev.event_type == "added") & ev.entity_id.notna()].itertuples():
    if (a.entity_id, a.effective_date) not in seen:
        out = ev[(ev.entity_id == a.entity_id) & (ev.event_type == "removed") & (ev.effective_date >= a.effective_date)]
        sp.loc[len(sp) + 10**6] = {"entity_id": a.entity_id, "entity_name": a.entity_name,
                                   "added_date": a.effective_date, "removed_date": out.effective_date.min()}
sp = sp.dropna(subset=["entity_id"]).astype({"entity_id": int})
sp["a"], sp["r"] = pd.to_datetime(sp.added_date), pd.to_datetime(sp.removed_date).fillna(pd.Timestamp("2099-12-31"))
sp = sp[sp.r >= "1980-01-01"]
ids = sorted(set(sp.entity_id))

# Monthly bars: split_ratio (product of the month's ratios) and the close/adj_close factor
px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start="1979-12-01", end="2026-11-26", interval="1mo",
                           fields=["close", "adj_close", "split_ratio", "shares_outstanding"], max_rows=250_000)
                for i in range(0, len(ids), 100)], ignore_index=True).sort_values(["entity_id", "date"])
g = px.groupby("entity_id")
px["factor"] = (px.close / px.adj_close).groupby(px.entity_id).shift(1) / (px.close / px.adj_close)
px["shares"] = px.shares_outstanding / g.shares_outstanding.shift(1)
# Google, Discovery, Under Armour, Wrigley share dividends
CLASS_C = {(10030636, "2014-04"): 2.0, (10031087, "2014-08"): 2.0, (10031256, "2016-04"): 2.0, (10004401, "2006-04"): 1.25}
px["ratio"] = [CLASS_C.get((e, d.strftime("%Y-%m")), r) for e, d, r in zip(px.entity_id, px.date, px.split_ratio)]

def plain(r):  # 5-for-4, 3-for-2, 2-for-1 ... or their reverse; not merger exchange ratios
    x = r if r >= 1 else 1 / r
    return abs(float(Fraction(x).limit_denominator(4)) - x) < 0.002

s = px[(px.date >= "1980-01-01") & (px.ratio >= 1.25)].copy()
s = s[s.ratio.map(plain) & (((s.factor / s.ratio - 1).abs() < 0.03) | ((s.shares / s.ratio - 1).abs() < 0.15))]

def member_on_ex_date(row):
    m0, m1 = row.date.to_period("M").start_time, row.date.to_period("M").end_time.normalize()
    sp_e = sp[sp.entity_id == row.entity_id]
    if ((sp_e.a <= m0) & (sp_e.r >= m1)).any():
        return True
    if not ((sp_e.a <= m1) & (sp_e.r >= m0)).any():
        return False
    d = xfl.prices(entity_id=int(row.entity_id), start=str(m0.date()), end=str(m1.date()),
                   fields=["close", "adj_close", "split_ratio", "return_daily", "shares_outstanding"])
    x = d.loc[d.split_ratio.notna(), "date"].min()
    return bool(((sp_e.a <= x) & (sp_e.r >= x)).any())

s["member"] = [member_on_ex_date(r) for r in s.itertuples()]

# One company = lines with the same name on the same roster (GOOG and GOOGL count once)
co = {e: e for e in ids}
for r in ros + [today]:
    for _, grp in r.dropna(subset=["entity_id"]).groupby("entity_name"):
        e = sorted(set(grp.entity_id.astype(int)) & set(ids))
        co.update({x: co[e[0]] for x in e})
s["co"], sp["co"], px["co"] = s.entity_id.map(co), sp.entity_id.map(co), px.entity_id.map(co)
now = set(today.entity_id.dropna().astype(int).map(co))
for y in range(1980, 2026):
    members = set(sp[(sp.a <= f"{y}-12-31") & (sp.r >= f"{y}-01-01")].co)
    pit = set(s[s.member & (s.date.dt.year == y)].co)
    traced = now & set(px[px.date.dt.year == y].co)
    split_traced = traced & set(s[s.date.dt.year == y].co)
    print(y, len(pit), f"{len(pit) / len(members):.1%}", len(split_traced), f"{len(split_traced) / len(traced):.1%}")
```

Full script with formatting and visualisation: [sp500-stock-splits-by-year-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-stock-splits-by-year-python.py)

## Output

![Share of S&P 500 companies splitting their stock 5-for-4 or more each year, 1980-2025: on the roster in force on each ex-date 19.9% (105 companies) in 1986 and 1987, 18.5% (97) in 1997, 0.2% (1) in 2009 and 3.1% (16) in 2024; today's roster traced back shows a higher share in 44 of the 46 years, 27.6% in 1986](/blog-images/sp500-stock-splits-by-year-python.png)

```
47 December rosters 1979-2025, today's roster (503 lines), 2,289 index events, 1,592 securities
Ratios of 1.25+ or 0.8- on member and non-member lines, 1980-2025: 3,370; adj_close factor agrees on 3,339, share count alone on 26; set aside: 25 not a plain ratio (merger exchange, recapitalisation), 4 unconfirmed by either

 year  members  splits  share  still in  today  splits  share  reverse
 1980      516      53  10.3%         7    174      25  14.4%        0
 1981      520      63  12.1%        13    177      32  18.1%        0
 1982      528      20   3.8%         6    179      23  12.8%        0
 1983      520      79  15.2%        23    192      48  25.0%        0
 1984      530      47   8.9%         8    199      23  11.6%        0
 1985      528      47   8.9%         8    206      32  15.5%        0
 1986      528     105  19.9%        30    214      59  27.6%        0
 1987      527     105  19.9%        25    221      50  22.6%        0
 1988      525      23   4.4%         7    224      12   5.4%        0
 1989      530      40   7.5%        14    230      22   9.6%        0
 1990      513      40   7.8%        12    233      26  11.2%        0
 1991      509      35   6.9%         9    243      29  11.9%        1
 1992      507      58  11.4%        20    250      51  20.4%        0
 1993      512      48   9.4%        16    265      39  14.7%        1
 1994      518      40   7.7%        20    275      41  14.9%        0
 1995      526      47   8.9%        21    291      43  14.8%        0
 1996      521      55  10.6%        22    296      39  13.2%        0
 1997      525      97  18.5%        50    312      74  23.7%        0
 1998      538      87  16.2%        38    322      61  18.9%        0
 1999      539      78  14.5%        37    335      59  17.6%        1
 2000      553      62  11.2%        30    345      60  17.4%        0
 2001      528      28   5.3%        14    358      28   7.8%        0
 2002      522      18   3.4%        10    364      28   7.7%        1
 2003      508      13   2.6%         7    364      18   4.9%        0
 2004      518      35   6.8%        21    375      41  10.9%        0
 2005      516      36   7.0%        22    382      45  11.8%        1
 2006      531      32   6.0%        17    388      27   7.0%        3
 2007      537      26   4.8%        14    400      26   6.5%        2
 2008      534       7   1.3%         4    404       6   1.5%        0
 2009      526       1   0.2%         1    408       2   0.5%        2
 2010      516       4   0.8%         2    418      12   2.9%        1
 2011      520      12   2.3%         9    425      15   3.5%        3
 2012      517      10   1.9%         8    432      12   2.8%        2
 2013      518      14   2.7%         6    441      11   2.5%        0
 2014      514      10   1.9%         8    446       8   1.8%        0
 2015      526      11   2.1%        10    453      14   3.1%        0
 2016      538       7   1.3%         6    455       8   1.8%        1
 2017      529       4   0.8%         4    461       5   1.1%        1
 2018      528       5   0.9%         5    468       7   1.5%        0
 2019      523       2   0.4%         2    474       3   0.6%        1
 2020      518       6   1.2%         6    480       7   1.5%        0
 2021      519       8   1.5%         8    486       9   1.9%        1
 2022      517       9   1.7%         9    487      10   2.1%        0
 2023      518       4   0.8%         4    490       4   0.8%        0
 2024      517      16   3.1%        16    494      18   3.6%        0
 2025      520       5   1.0%         5    497       6   1.2%        0

1980-1989: 582 splits, 58.2 a year (11.1% of members); today's roster 32.6 a year (16.3%)
1990-1999: 585 splits, 58.5 a year (11.2% of members); today's roster 46.2 a year (16.1%)
2000-2009: 258 splits, 25.8 a year (4.9% of members); today's roster 28.1 a year (7.6%)
2010-2019: 79 splits, 7.9 a year (1.5% of members); today's roster 9.5 a year (2.2%)
2020-2025: 48 splits, 8.0 a year (1.5% of members); today's roster 9.0 a year (1.8%)
Most: 105 in 1986, 1987; fewest: 1 in 2009
Today's-roster share minus point-in-time share: mean +3.0 points, above zero in 44 of 46 years; 1980-1999 +5.1, 2000-2025 +1.4
Split ratios 1980-1999: 66% two-for-one, 26% below it, 2% four-for-one or more (1167 splits)
Split ratios 2000-2019: 77% two-for-one, 15% below it, 3% four-for-one or more (337 splits)
Split ratios 2020-2025: 25% two-for-one, 12% below it, 46% four-for-one or more (48 splits)
Within the real roster, mean yearly split rate: 8.3% for companies still in the index today, 5.3% for companies removed since (higher in 41 of 46 years)
Point-in-time splitters 1980-2025 still in the index today: 634 of 1552

Reverse splits by members on the ex-date, 1980-2025: 22
  1991-08  VAT   VARITY CORP                       1-for-10
  1993-07  NAV   NAVISTAR INTERNATIONAL CORP       1-for-10
  1999-11  ATI   ALLEGHENY TELEDYNE INC            1-for-2
  2002-11  T     A T & T CORP                      1-for-5
  2005-05  ADCT  A D C TELECOMMUNICATIONS INC      1-for-7
  2006-02  PMTC  PARAMETRIC TECHNOLOGY CORP        2-for-5
  2006-09  CIEN  CIENA CORP                        1-for-7
  2006-10  JDSU  J D S UNIPHASE CORP               1-for-8
  2007-07  TYC   TYCO INTERNATIONAL LTD NEW        1-for-4
  2007-11  JAVA  SUN MICROSYSTEMS INC              1-for-4
  2009-03  TWX   TIME WARNER INC NEW               1-for-3
  2009-07  AIG   AMERICAN INTERNATIONAL GROUP INC  1-for-20
  2010-06  ETFC  E TRADE FINANCIAL CORP            1-for-10
  2011-01  MSI   MOTOROLA INC                      1-for-7
  2011-05  C     CITIGROUP INC                     1-for-10
  2011-12  EXPE  EXPEDIA INC DE                    1-for-2
  2012-07  DUK   DUKE ENERGY CORP NEW              1-for-3
  2012-10  THC   TENET HEALTHCARE CORP             1-for-4
  2016-10  AA    ALCOA INC                         1-for-3
  2017-06  XRX   XEROX CORP                        1-for-4
  2019-06  DD    DOWDUPONT INC                     1-for-3
  2021-08  GE    GENERAL ELECTRIC CO               1-for-8

Forward splits while a member, spot checks:
  APPLE INC      1987-06 2, 2000-06 2, 2005-02 2, 2014-06 7, 2020-08 4
  MICROSOFT CORP 1996-12 2, 1998-02 2, 1999-03 2, 2003-02 2
  WALMART INC    1983-07 2, 1985-10 2, 1987-07 2, 1990-07 2, 1993-02 2, 1999-04 2, 2024-02 3
  NVIDIA CORP    2006-04 2, 2007-09 3/2, 2021-07 4, 2024-06 10
  AMAZON COM INC 2022-06 20
  ALPHABET INC   2014-04 2, 2022-07 20
  TESLA INC      2022-08 3
```

## From one member in nine to one in sixty-seven

About 11% of members split each year through the 1980s and 1990s, peaking in the bull markets of 1986-1987 and 1997-1999. The 2001-2003 bear market cut the count to 13, and after 36 in 2005 the 2008 crash took it to 7, then to one split in 2009 (AmerisourceBergen's 2-for-1). Since 2010 the rate has held near 1.5% a year.

The 2020s revival is narrow: 8.0 splits a year against 7.9 in the 2010s, with 2024's 16 the most since 2007. The splits are larger. In 1980-1999, 66% were 2-for-1 and 2% were 4-for-1 or more; in 2020-2025, 46% were 4-for-1 or more, up to Chipotle's 50-for-1 in 2024.

Members made 22 reverse splits from 1980 to 2025, including AT&T's 1-for-5 in 2002, AIG's 1-for-20 in 2009, Citigroup's 1-for-10 in 2011 and GE's 1-for-8 in 2021.

## Why does a survivor list overstate the split rate?

On each year's real roster, companies still in the index today split at 8.3% a year on average and those removed since at 5.3%; the survivors split more in 41 of 46 years. A rising share price leads to splits, and it also keeps a company large enough to stay in the index. Today's roster keeps that group and adds companies that split before joining: in 1986 it shows 27.6% splitting against 19.9%.

Counts move the other way: only 174 of today's companies traded in 1980, so the traced-back 1980s average 32.6 splits a year against the real 58.2. A survivor list understates how many splits the index saw and overstates how often a member split.

## Using split counts without a survivor list

An event study of split announcements needs the roster on each ex-date and a fixed definition of a split, since mergers and spin-offs also change the share basis. The [historical S&P 500 constituents](/historical-sp500-constituents) page lists the members on any date back to 1925. [A year-by-year count of S&P 500 dividend payers](/blog/sp500-dividend-payers-by-year-python) applies the same roster method to cash dividends, and [a test of split-adjusted prices in a backtest](/blog/lookahead-bias-split-adjusted-prices-python) covers how later splits leak into earlier prices.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
