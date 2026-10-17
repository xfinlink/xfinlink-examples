# How Long After an IPO Can a Company Join the S&P 500?

October 17, 2026 · INDEX-UNIVERSE

No company that listed through its own IPO joined the S&P 500 within two years of its first trading day in 2020–2025; the quickest, Moderna, took 2.6 years. Of the 14 first-time members in those years that joined with less than two years of trading, 13 were spin-offs from existing members and one, Kenvue, was a carve-out Johnson & Johnson was separating. Across 736 first entries from 1990 to 2025, the median new member had traded for 11.2 years.

## How long must a company trade before it can join the S&P 500?

S&P Dow Jones Indices' current rules for the S&P Composite 1500, which contains the S&P 500, state that "Initial public offerings should be traded on an eligible exchange for at least 12 months", and that "Spin-offs or in-specie distributions from existing constituents are not required to have 12 months of trading". A consultation proposed cutting the seasoning period to six months for very large companies; its results, published on 4 June 2026, left it unchanged (S&P Dow Jones Indices press release, checked 17 October 2026).

Twelve months is a floor. The typical wait can only be measured on the dated record of past additions, since most companies that joined in the 1990s have left the index and are missing from today's member list.

## Dating 736 first entries against their first trading day

1. Take the 859 additions effective 1990 to 2025 from the [index events log](/docs/index-events).
2. Set aside 7 second share classes, 46 securities returning after an earlier membership and 24 stays of under 30 days, 21 of them spin-offs removed within three days of joining.
3. Date each listing from the security's first daily price bar. For 75 additions that bar is one of the three dates on which the daily series begin (2 January 1962, 2 July 1962, 14 December 1972), so the age is a lower bound; each decade's shortest bound exceeds its median.
4. Classify by hand every first entry with under five years of trading: spin-off, carve-out IPO (a parent sold shares in an IPO, then the rest), own IPO or direct listing (including returns after a buyout or bankruptcy, such as HCA and General Motors), or a new security created when a listed business merged or reorganised, such as Kraft Heinz or Linde. The 46 in the last group are set aside.

The full event log is on paid plans; `as_of` rosters work on every plan.

## Code

```python
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose

# Second share classes of companies already in the index (Alphabet C, Under Armour C, Fox B, ...)
SECOND_CLASS = {10003590, 10032892, 10006698, 10003102, 10030753, 10004763, 10029970}
EARLIER_CLASS = {10025221: 10006133}  # Berkshire's index line (class B) -> class A, listed earlier
COVERAGE_START = {"1962-01-02", "1962-07-02", "1972-12-14"}  # daily bars begin on these dates

# Hand classification of every first entry listed under five years (ticker:year of addition).
# Default: under 0.1 years = spin-off, 0.1-5 years = own IPO or direct listing.
ROUTE = {
    "carve-out IPO": """SFR:1990 KRB:1992 DWD:1993 ATI:1994 FDC:1994 GLD:1994 ALL:1995 CSE:1996
        LU:1996 UPR:1996 GDT:1996 AFS:1998 COF:1998 DPH:1999 COC:1999 TSG:2000 A:2000 CVG:2000
        CIT:2000 PALM:2000 MIR:2001 PBG:2001 MON:2002 TAPB:2002 COH:2004 CIT:2004 FSLB:2004
        GNW:2005 AIZ:2007 MJN:2009 ZTS:2013 SYF:2015 CFG:2016 KVUE:2023""",
    "spin-off": """AGN:1992 LEH:1998 NCR:2000 CNXT:2000 RJR:2002 EXPE:2007 DPS:2008 DISCA:2010
        MNK:2014 FBHS:2016 KEYS:2018 LW:2018 SNDK:2025""",
    "other": "FRE:1992 VIA:1994 NDAQ:2008 TWC:2009",
    "reorganisation": """CVN:1990 X:1991 TMC:1995 FCX:1995 UMG:1995 PNU:1995 FD:1995 GRA:1996
        ROK:1996 TEN:1996 SEE:1998 MAR:1998 UMG:1998 SRE:1998 EDS:1998 PCS:1998 MCK:1999 WLP:1999
        DVN:2000 DYN:2000 AET:2000 FLR:2000 XL:2001 CMCSA:2002 NWSA:2004 SHLD:2005 HOT:2006
        DTV:2006 NYX:2007 IVZ:2008 ESV:2012 LYB:2012 PNR:2012 KHC:2015 FTI:2017 DWDP:2017
        LIN:2018 AMCR:2019 VTRS:2020 WBD:2022 KDP:2022 SW:2024 TPL:2024 EXE:2025 PSKY:2025 CRH:2025""",
}
ROUTE_OF = {k: r for r, keys in ROUTE.items() for k in keys.split()}


def events(kind):
    pages = [xfl.index_events("sp500", event_type=kind, start="1925-12-31")]
    while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
        pages.append(xfl.index_events("sp500", event_type=kind, start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True).dropna(subset=["entity_id"]).astype({"entity_id": int})


def first_bar(entity_id):
    p = xfl.prices(entity_id=EARLIER_CLASS.get(entity_id, entity_id), start="1962-01-01", limit=1, fields=["close"])
    return p["date"].iloc[0].strftime("%Y-%m-%d")


adds, rems = events("added"), events("removed")

# each addition's last day in the index; a stay under 30 days is a placement, not a membership
adds["last_day"] = [rems.loc[(rems["entity_id"] == r.entity_id) & (rems["effective_date"] >= r.effective_date),
                             "effective_date"].min() for r in adds.itertuples()]
adds["days_in"] = (pd.to_datetime(adds["last_day"]) - pd.to_datetime(adds["effective_date"])).dt.days
adds["placement"] = adds["days_in"] < 30
placement_exits = set(zip(adds.loc[adds["placement"], "entity_id"], adds.loc[adds["placement"], "last_day"]))
real_adds = adds[~adds["placement"]]
real_exits = rems[[(e, d) not in placement_exits for e, d in zip(rems["entity_id"], rems["effective_date"])]]

window = adds[adds["effective_date"].between("1990-01-01", "2025-12-31")].copy()
window["second_class"] = window["entity_id"].isin(SECOND_CLASS)
window["returning"] = [((real_adds["entity_id"] == r.entity_id) & (real_adds["effective_date"] < r.effective_date)).any()
                       or ((real_exits["entity_id"] == r.entity_id) & (real_exits["effective_date"] < r.effective_date)).any()
                       for r in window.itertuples()]
first = window[~window["second_class"] & ~window["placement"] & ~window["returning"]].copy()

with ThreadPoolExecutor(8) as ex:
    first["first_bar"] = list(ex.map(first_bar, first["entity_id"]))
first["years"] = ((pd.to_datetime(first["effective_date"]) - pd.to_datetime(first["first_bar"])).dt.days / 365.25).clip(lower=0)
first["before_start"] = first["first_bar"].isin(COVERAGE_START)  # listed before daily bars begin: age is a floor
first["key"] = first["ticker"] + ":" + first["effective_date"].str[:4]
young = (first["years"] < 5) & ~first["before_start"]
default = np.where(first["years"] < 0.1, "spin-off", "own IPO")
first["route"] = np.where(young, [ROUTE_OF.get(k, d) for k, d in zip(first["key"], default)], "listed 5y+")
first["decade"] = first["effective_date"].str[:3] + "0s"

reorg = first["route"] == "reorganisation"
sample = first[~reorg]
print(f"S&P 500 additions effective 1990-2025: {len(window)}")
print(f"  second share classes: {window['second_class'].sum()}; stays under 30 days: {window['placement'].sum()}; "
      f"securities returning to the index: {(window['returning'] & ~window['second_class'] & ~window['placement']).sum()}")
print(f"  first entries: {len(first)}; new securities from mergers and reorganisations set aside: {reorg.sum()}")
print(f"  analysed: {len(sample)}; first bar on a coverage start date (listed before it): {sample['before_start'].sum()}\n")

rows = []
for dec, g in sample.groupby("decade"):
    u2 = g[g["years"] < 2]
    rows.append({"decade": dec, "adds": len(g), "median_yrs": g["years"].median(),
                 "median_ex_spinoffs": g.loc[g["route"] != "spin-off", "years"].median(),
                 "min_floor": g.loc[g["before_start"], "years"].min(),
                 "under_2y": len(u2), "spin_off": (u2["route"] == "spin-off").sum(),
                 "carve_out": (u2["route"] == "carve-out IPO").sum(), "own_ipo": (u2["route"] == "own IPO").sum(),
                 "own_ipo_under_1y": ((g["years"] < 1) & (g["route"] == "own IPO")).sum(),
                 "20y_plus": ((g["years"] >= 20) | g["before_start"]).sum()})
table = pd.DataFrame(rows).set_index("decade")
print(table.round(1).to_string())
print(f"all decades: median {sample['years'].median():.1f} years, under 2 years {(sample['years'] < 2).mean():.1%}\n")

print("Own IPO or direct listing, joined within 12 months:")
print(sample[(sample["route"] == "own IPO") & (sample["years"] < 1)][["effective_date", "ticker", "first_bar", "years"]].round(2).to_string(index=False))
print("\nFastest own listings by decade:")
fast = sample[sample["route"] == "own IPO"].sort_values("years").groupby("decade").head(2)
print(fast.sort_values(["decade", "years"])[["decade", "effective_date", "ticker", "first_bar", "years"]].round(2).to_string(index=False))
print("\nJoined within 12 months, any route, since 2003:")
late = sample[(sample["effective_date"] >= "2003-01-01") & (sample["years"] < 1)]
print(late["route"].value_counts().to_string())
```

Full script with formatting and visualisation: [how-long-after-ipo-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/how-long-after-ipo-sp500-python.py)

## Output

![Stacked bar chart of first-time S&P 500 additions by decade, 1990s to 2020s, split by how long each company had traded when it joined; spin-offs under two years rise from 8% to 14% of entrants while IPOs and carve-outs under two years fall from 6% to 1%](/blog-images/how-long-after-ipo-sp500-python.png)

```
S&P 500 additions effective 1990-2025: 859
  second share classes: 7; stays under 30 days: 24; securities returning to the index: 46
  first entries: 782; new securities from mergers and reorganisations set aside: 46
  analysed: 736; first bar on a coverage start date (listed before it): 75

        adds  median_yrs  median_ex_spinoffs  min_floor  under_2y  spin_off  carve_out  own_ipo  own_ipo_under_1y  20y_plus
decade                                                                                                                     
1990s    194        10.2                11.4       17.8        28        16         11        1                 0        58
2000s    266        10.2                11.9       28.2        44        26          9        9                 4        59
2010s    186        14.1                16.6       39.0        31        24          3        4                 0        61
2020s     90        11.0                15.1       47.4        14        13          1        0                 0        28
all decades: median 11.2 years, under 2 years 15.9%

Own IPO or direct listing, joined within 12 months:
effective_date ticker  first_bar  years
    2000-12-11    MET 2000-04-05   0.68
    2002-07-22    PFG 2001-10-23   0.74
    2002-07-22    PRU 2001-12-13   0.61
    2002-07-25    ATH 2001-10-30   0.73

Fastest own listings by decade:
decade effective_date ticker  first_bar  years
 1990s     1999-09-29   GBLX 1998-08-14   1.13
 1990s     1995-02-24    BSX 1992-05-19   2.77
 2000s     2002-07-22    PRU 2001-12-13   0.61
 2000s     2000-12-11    MET 2000-04-05   0.68
 2010s     2012-12-24   DLPH 2011-11-17   1.10
 2010s     2012-05-25    KMI 2011-02-11   1.28
 2020s     2021-07-21   MRNA 2018-12-07   2.62
 2020s     2023-09-18   ABNB 2020-12-10   2.77

Joined within 12 months, any route, since 2003:
route
spin-off         55
carve-out IPO     4
```

## Spin-offs now fill the young end of the index

The share of young entrants barely moved: in every decade 14% to 17% of first entries had traded for less than two years. Their route to market changed. In the 1990s, 12 of the 28 came through an IPO, 11 of them carve-outs such as Lucent and Delphi. In the 2000s nine companies that floated on their own joined inside two years, Google and Visa among them, and four newly demutualised insurers (MetLife, Principal, Prudential and Anthem) joined within 12 months. Since 2003, every first entry with under a year of trading has been a spin-off (55) or a carve-out its parent was separating (4: Freescale, Mead Johnson, Zoetis, Kenvue).

The 2010s had four own IPOs inside two years (Delphi Automotive, Kinder Morgan, Michael Kors, Facebook); the 2020s had none, and Moderna (2.6 years) and Airbnb (2.8) were the fastest.

The current rules fit this split. A spin-off comes from a parent already in the index: 12 of the 14 young entrants since 2020 joined on their first day of trading, Sandisk nine months after Western Digital spun it off. An IPO has to sit out the seasoning period and pass the financial viability screens, and the companies that floated on their own arrived years later. Without spin-offs, the median entrant had traded 11.4 years in the 1990s, 16.6 in the 2010s and 15.1 in 2020–2025.

## Planning an index-inclusion screen around listing age

A model that ranks recent IPOs as candidates for the next addition should treat the 12-month seasoning period as a minimum, not a forecast: on 2020–2025 evidence, a company that floats on its own becomes a realistic candidate no earlier than its third year of trading. Spin-offs from large members are the opposite case and can enter on their first day, sometimes for only a few sessions, so an event study should hold those stays apart. [An event study of S&P 500 additions](/blog/sp500-index-effect-event-study-python) measures returns around the effective date, and [a post on new members' market value](/blog/sp500-additions-market-cap-at-entry-python) sizes each one on its first day in the index.

The [guide to S&P 500 additions and deletions history](/blog/sp500-additions-and-deletions-history) explains which date each row of the log carries, and the [historical S&P 500 constituents lookup](/historical-sp500-constituents) lists the members on any past date, companies that later left included.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
