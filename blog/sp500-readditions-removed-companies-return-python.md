# Removed From the S&P 500: How Many Companies Come Back?

October 18, 2026 · INDEX-UNIVERSE

Removal from the S&P 500 is almost always final: of the 850 removals since 1990, 33 (3.9%) ended with the same security added back later, after a median 5.9 years outside the index. Seven of the 33 returned under a different ticker. A membership table that stores one first-added and one last-removed date per security counts those absences as membership, placing an average of 6.1 non-members inside the index per weekday from 1990 to 2025.

## Can a company rejoin the S&P 500 after being removed?

Yes, after a wait. S&P’s U.S. index rules, as reproduced in an [index-linked note filed with the SEC in December 2024](https://www.sec.gov/Archives/edgar/data/19617/000121390024109873/ea0225071-01_424b2.htm), say a company removed from the S&P 500, MidCap 400 or SmallCap 600 “must wait a minimum of one year from its removal date before being reconsidered as a replacement candidate for the S&P 500” (checked 18 October 2026).

Returns break shortcuts that assume a stable list: one span per company keeps a company in the index while it was out, and a join on symbols misses one that comes back under a new ticker. Counting them needs the dated record of every removal.

## Pairing every removal with the same security’s next addition

1. Rebuild each security’s stays in the index from the 3,612 dated S&P 500 changes in the [index events log](/docs/index-events).
2. Set aside stays shorter than 30 days, such as Lamb Weston’s single day as a spin-off in November 2016; these are placements, not memberships.
3. Count a gap between two stays of one security as a removal followed by a return, for removals from 1990 onward. A company that returns as a new security, as General Motors did after its 2009 bankruptcy, starts a new history and is not counted.
4. Mark every weekday from 1990 to 2025 on which one of these securities sat outside the index between two stays.

The full event log is on paid plans; `as_of` rosters work on every plan. On six test dates from 1995 to 2024, no security the log places between stays appears in that day’s `as_of` roster.

## Code

```python
import warnings

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose

pages = [xfl.index_events("sp500", start="1925-12-31")]
while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
    pages.append(xfl.index_events("sp500", start="1925-12-31", offset=1000 * len(pages)))
ev = pd.concat(pages, ignore_index=True).dropna(subset=["entity_id"]).astype({"entity_id": int})
ev["effective_date"] = pd.to_datetime(ev["effective_date"])
ev = ev.sort_values(["entity_id", "effective_date"], kind="stable")

# Rebuild membership spells per security; founding members open before their first event
spells = []
for eid, g in ev.groupby("entity_id"):
    start = None if g["event_type"].iloc[0] == "added" else pd.Timestamp("1925-12-31")
    tick_in = g["ticker"].iloc[0]
    for r in g.itertuples():
        if r.event_type == "added":
            start, tick_in = r.effective_date, r.ticker
        elif start is not None:
            spells.append((eid, r.entity_name, start, r.effective_date, tick_in, r.ticker))
            start = None
    if start is not None:
        spells.append((eid, g["entity_name"].iloc[-1], start, pd.NaT, tick_in, None))
spells = pd.DataFrame(spells, columns=["entity_id", "name", "added", "removed", "ticker_in", "ticker_out"])
# a stay under 30 days is a placement (a spin-off or merger vehicle), not a membership
spells = spells[~((spells["removed"] - spells["added"]).dt.days < 30)]

# Each gap between two memberships of the same security is a removal followed by a return
spells["next_added"] = spells.groupby("entity_id")["added"].shift(-1)
spells["ticker_back"] = spells.groupby("entity_id")["ticker_in"].shift(-1)
gaps = spells.dropna(subset=["next_added"]).copy()
gaps["years_out"] = (gaps["next_added"] - gaps["removed"]).dt.days / 365.25

exits = spells[spells["removed"] >= "1990-01-01"]
back = gaps[gaps["removed"] >= "1990-01-01"].sort_values("removed")
renamed = back[back["ticker_out"] != back["ticker_back"]]
print(f"Removals since 1990: {len(exits)}; later re-added as the same security: {len(back)} ({len(back) / len(exits):.1%})")
print(f"Years outside: median {back['years_out'].median():.1f}, shortest {back['years_out'].min():.1f}, "
      f"longest {back['years_out'].max():.1f}; over 5 years: {(back['years_out'] > 5).sum()}")
print(f"Returned under a different ticker: {len(renamed)}  "
      + ", ".join(f"{a}->{b}" for a, b in zip(renamed["ticker_out"], renamed["ticker_back"])))
print("Returns by decade:", back.groupby(back["next_added"].dt.year // 10 * 10).size().to_dict())

# A table with one first-added / last-removed span per security counts each of these gaps as membership
days = pd.bdate_range("1990-01-01", "2025-12-31")
wrong = pd.Series(0, index=days)
for g in back.itertuples():
    wrong[(days > g.removed) & (days < g.next_added)] += 1
print(f"One-span table, 1990-2025: {wrong.mean():.1f} non-members counted per weekday, "
      f"{(wrong > 0).mean():.0%} of weekdays affected, peak {wrong.max()} on {wrong.idxmax():%Y-%m-%d}")
```

Full script with formatting and visualisation: [sp500-readditions-removed-companies-return-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-readditions-removed-companies-return-python.py)

## Output

![Timeline of the 33 S&P 500 members removed since 1990 that were later re-added, each drawn as a line from removal to return; most absences run five to seventeen years, and a cluster of short ones starts in 2008 and 2009](https://xfinlink.com/blog-images/sp500-readditions-removed-companies-return-python.png)

```
Removals since 1990: 850; later re-added as the same security: 33 (3.9%)
Years outside: median 5.9, shortest 1.3, longest 17.3; over 5 years: 19
Returned under a different ticker: 7  MHS->HST, Z->FL, AMR->AAL, PMTC->PTC, PCS->TMUS, SAI->LDOS, CSC->DXC
Returns by decade: {1990: 1, 2000: 5, 2010: 19, 2020: 8}
One-span table, 1990-2025: 6.1 non-members counted per weekday, 91% of weekdays affected, peak 17 on 2009-12-21

Removed since 1990 and re-added:
                         name ticker_out ticker_back       left   returned  years_out
                   HUMANA INC        HUM         HUM 1993-03-05 1995-12-01        2.7
    HOST HOTELS & RESORTS INC        MHS         HST 1993-10-11 2007-03-20       13.4
              FOOT LOCKER INC          Z          FL 1998-12-31 2016-04-04       17.3
    L3HARRIS TECHNOLOGIES INC        HRS         HRS 1999-11-05 2008-09-22        8.9
        HELMERICH & PAYNE INC         HP          HP 1999-12-30 2010-03-01       10.2
         KANSAS CITY SOUTHERN        KSU         KSU 2000-07-12 2013-05-24       12.9
                O I GLASS INC         OI          OI 2000-12-08 2009-01-02        8.1
             MASSEY ENERGY CO        MEE         MEE 2000-12-21 2008-06-23        7.5
                ONEOK INC NEW        OKE         OKE 2001-08-29 2010-03-15        8.5
                   F M C CORP        FMC         FMC 2001-12-31 2009-08-19        7.6
  AMERICAN AIRLINES GROUP INC        AMR         AAL 2003-03-13 2015-03-23       12.0
                    P T C INC       PMTC         PTC 2006-12-29 2021-04-20       14.3
MAXIM INTEGRATED PRODUCTS INC       MXIM        MXIM 2007-09-26 2018-12-03       11.2
                    CHUBB LTD        ACE         ACE 2008-07-17 2010-07-15        2.0
 BROOKFIELD PROPERTY REIT INC        GGP         GGP 2008-11-12 2013-12-10        5.1
               TRANSOCEAN LTD        RIG         RIG 2008-12-18 2013-10-29        4.9
    JOHNSON CONTROLS INTL PLC        TYC         TYC 2009-03-16 2010-08-27        1.4
               NOBLE CORP PLC         NE          NE 2009-03-26 2011-01-18        1.8
                 COVIDIEN PLC        COV         COV 2009-06-04 2011-03-01        1.7
         T E CONNECTIVITY PLC        TEL         TEL 2009-06-25 2011-10-17        2.3
       TRANE TECHNOLOGIES PLC         IR          IR 2009-06-30 2010-11-17        1.4
        COOPER INDUSTRIES PLC        CBE         CBE 2009-09-08 2011-11-23        2.2
                   CIENA CORP       CIEN        CIEN 2009-12-18 2026-02-09       16.1
             T MOBILE U S INC        PCS        TMUS 2013-04-30 2019-07-15        6.2
   ADVANCED MICRO DEVICES INC        AMD         AMD 2013-09-20 2017-03-20        3.5
          LEIDOS HOLDINGS INC        SAI        LDOS 2013-09-20 2019-08-09        5.9
                 TERADYNE INC        TER         TER 2013-12-20 2020-09-21        6.8
                    JABIL INC        JBL         JBL 2014-11-04 2023-12-18        9.1
          D X C TECHNOLOGY CO        CSC         DXC 2015-11-30 2017-04-04        1.3
              FIRST SOLAR INC       FSLR        FSLR 2017-03-17 2022-12-19        5.8
                   E Q T CORP        EQT         EQT 2018-11-12 2022-10-03        3.9
                 P G & E CORP        PCG         PCG 2019-01-17 2022-10-03        3.7
                 ILLUMINA INC       ILMN        ILMN 2024-06-21 2026-09-21        2.3
```

## Which removed companies came back, and after how long?

Fewer than one removal in 25 has been reversed, and 19 of the 33 returns took more than five years. The security that left at the end of 1998 as Venator Group, the renamed Woolworth, under the symbol Z came back in 2016 as Foot Locker (FL), 17.3 years later; Ciena left in December 2009 and rejoined on 9 February 2026, replacing Dayforce (S&P Dow Jones Indices release, 4 February 2026). Nobody returned inside a year, consistent with the one-year minimum.

The quick returns cluster in one episode: seven of the nine absences under two and a half years began between July 2008 and September 2009. S&P announced on 11 December 2008 that Transocean, “in the process of redomesticating to Switzerland”, was “ineligible for continued inclusion in the S&P 500” and would leave after the close on 18 December, the date in the log. ACE left in July 2008 after moving its holding company from the Cayman Islands to Switzerland, and was readmitted in July 2010 once S&P had revised its definition of a U.S. company that May (*The Royal Gazette*, 12 July 2010). Tyco, Noble, Covidien, TE Connectivity, Ingersoll-Rand and Cooper Industries, all removed in 2009, were back between August 2010 and November 2011.

Long absences make the one-span error persistent. Foot Locker alone is wrongly counted for 17 years and Ciena for 16, so at least one false member sits in the index on 91% of weekdays, peaking at 17 on 21 December 2009. Symbols add a second failure: American Airlines left as AMR and came back as AAL, and SAIC’s security left as SAI and returned as Leidos (LDOS).

## Building an S&P 500 membership table that handles returns

Store membership as stays, one row per entry and exit, never as one span per company. Rosters drawn with `as_of` on each rebalance date avoid the problem by construction, and the [historical S&P 500 constituents lookup](/historical-sp500-constituents) shows the roster for any single date.

Key the history on a security identifier rather than the symbol, since seven of the 33 came back under a new ticker. Symbols also move the other way, as [a test of ticker recycling in a mean-reversion backtest](/blog/ticker-recycling-mean-reversion-backtest-python) examines.

Flag returning members in any study built on index changes. The dated list of [S&P 500 additions and deletions](/blog/sp500-additions-and-deletions-history) shows each entry and exit in sequence, and an analysis of [what happens to stocks after removal](/blog/sp500-replacement-pairs-deletion-returns-python) should treat a later return as part of the same history.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
