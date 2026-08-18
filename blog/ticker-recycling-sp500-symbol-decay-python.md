**Do Old Ticker Symbols Still Point to the Same Company? S&P 500 Ticker Recycling in Python**

August 18, 2026 · INDEX-UNIVERSE

**What's the question?**

A ticker symbol is a lease on three or four letters, not a permanent name. Once the holder is acquired, taken private or renamed, the exchange is free to hand those letters to somebody else, and the handover leaves no mark in a price file.

Saved symbol lists get reused constantly. A screen published in a 2010 paper arrives as a table of tickers; a universe file is copied forward from one project to the next. Years later those symbols go back into a data request, and it answers.

Three things can come back. The symbol still belongs to the same company, so the answer matches the intent. It belongs to nobody, so nothing is returned and the name drops out of the sample, which costs coverage but announces itself in a row count. Or it now belongs to a different company, and the request returns a clean, entirely plausible price series for a business the researcher never meant to study. The third case does the real damage, because nothing in the output flags it.

**The approach**

Six S&P 500 rosters, as they stood at the end of 1996, 2000, 2005, 2010, 2015 and 2020. Each is addressed by permanent entity identifier, so the company is fixed before any symbol enters the analysis.

1. Pull the point-in-time membership list for each of the six dates.
2. For every member, read the symbol its price series carried in the closing trading days of that year, the symbol a researcher would have copied down at the time.
3. Send all 1,049 distinct symbols to the price endpoint for June to mid-August 2026 and record which company answers.
4. Classify every symbol as the same company, a different company, or no answer.
5. For the 2005 list, check whether the new owner of each reassigned symbol was already quoted in June 2010, inside a 2006 to 2016 study window.

Members without a quoted price in the closing weeks of the year drop from the sample; the retained counts appear in the output.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

VINTAGES = ["1996-12-31", "2000-12-31", "2005-12-31", "2010-12-31",
            "2015-12-31", "2020-12-31"]


def price_batch(start, end, ids=None, syms=None):
    keys = ids if ids is not None else syms
    parts = []
    for i in range(0, len(keys), 50):
        chunk = keys[i:i + 50]
        parts.append(
            xfl.prices(entity_id=chunk, start=start, end=end, fields=["close"])
            if ids is not None else
            xfl.prices(chunk, start=start, end=end, fields=["close"]))
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


# the symbol each member actually traded under at each vintage
vintage_frames = {}
for as_of in VINTAGES:
    roster = xfl.index("sp500", as_of=as_of).dropna(subset=["entity_id"])
    roster["entity_id"] = roster["entity_id"].astype(int)
    px = price_batch((pd.Timestamp(as_of) - pd.Timedelta(days=20)).date().isoformat(),
                     as_of, ids=roster["entity_id"].tolist())
    vintage_frames[as_of] = (
        px.groupby("entity_id").agg(symbol=("ticker", "last")).reset_index()
        .merge(roster[["entity_id", "entity_name"]], on="entity_id")
        .drop_duplicates("symbol"))

# which company answers to that symbol today
all_syms = sorted({s for f in vintage_frames.values() for s in f["symbol"]})
now = price_batch("2026-06-01", "2026-08-14", syms=all_syms)
cur = now.groupby("ticker").agg(now_id=("entity_id", "last"),
                                now_name=("entity_name", "last")).reset_index()

rows = []
for as_of, then in vintage_frames.items():
    j = then.merge(cur, left_on="symbol", right_on="ticker", how="left")
    j["outcome"] = np.where(j["now_id"].isna(), "no data",
                            np.where(j["now_id"] == j["entity_id"],
                                     "same company", "different company"))
    rows.append(j.assign(vintage=as_of[:4]))

res = pd.concat(rows, ignore_index=True)
print(res.pivot_table(index="vintage", columns="outcome",
                      values="symbol", aggfunc="count"))
```

Full script with formatting and visualisation: [ticker-recycling-sp500-symbol-decay-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/ticker-recycling-sp500-symbol-decay-python.py)

**Output**

![Stacked bars showing what a saved S&P 500 ticker list returns when replayed in August 2026: the 2020 list returns the same company for 89 percent of symbols, falling to 37 percent for the 1996 list, while the share answering for a different company rises from under 1 percent to 15 percent](/blog-images/ticker-recycling-sp500-symbol-decay-python.png)

```
1996 roster: 498 members, 433 with a traded symbol on the tape
2000 roster: 497 members, 458 with a traded symbol on the tape
2005 roster: 498 members, 472 with a traded symbol on the tape
2010 roster: 499 members, 479 with a traded symbol on the tape
2015 roster: 500 members, 492 with a traded symbol on the tape
2020 roster: 501 members, 500 with a traded symbol on the tape

Distinct symbols tested: 1049   answering in 2026-06-2026-08: 673

What a saved ticker list returns when replayed in August 2026
 list from  symbols           same co      different co           no data
      1996      433    162 (37.4%)     65 (15.0%)    206 (47.6%)
      2000      458    218 (47.6%)     43 ( 9.4%)    197 (43.0%)
      2005      472    263 (55.7%)     31 ( 6.6%)    178 (37.7%)
      2010      479    314 (65.6%)     21 ( 4.4%)    144 (30.1%)
      2015      492    372 (75.6%)     11 ( 2.2%)    109 (22.2%)
      2020      500    447 (89.4%)      2 ( 0.4%)     51 (10.2%)

2005 symbols now answering for a different company: 31
  symbol held in 2005 by (entity, name today)  answering today
  AA     HOWMET AEROSPACE INC                  ALCOA CORP
  ADCT   A D C TELECOMMUNICATIONS INC          A D C THERAPEUTICS S A
  APC    ANADARKO PETROLEUM CORP               ARKO PETROLEUM CORP
  ASO    AMSOUTH BANCORPORATION                ACADEMY SPORTS & OUTDOORS INC
  BR     BURLINGTON RESOURCES INC              BROADRIDGE FINANCIAL SOLUTNS INC
  BUD    ANHEUSER BUSCH COS INC                ANHEUSER BUSCH INBEV SA NV
  CC     CIRCUIT CITY STORES INC               CHEMOURS CO
  CD     CENDANT CORP                          Chaince Digital Holdings Inc.
  CEG    CONSTELLATION ENERGY GROUP INC        CONSTELLATION ENERGY CORP
  DELL   DELL INC                              DELL TECHNOLOGIES INC
  DOW    DOW CHEMICAL CO                       DOW INC
  EC     ENGELHARD CORP                        ECOPETROL S A
  EMC    E M C CORP MA                         GLOBAL X FUNDS
  EP     EL PASO CORP                          EMPIRE PETROLEUM CORP
  ET     E TRADE FINANCIAL CORP                ENERGY TRANSFER L P
  GM     GENERAL MOTORS CORP                   General Motors Company
  IGT    INTERNATIONAL GAME TECHNOLOGY         Brightstar Lottery PLC
  LU     LUCENT TECHNOLOGIES INC               LUFAX HOLDING LTD
  MEDI   MEDIMMUNE INC                         HARBOR E T F TRUST
  MI     MARSHALL & ILSLEY CORP NEW            NFT Ltd
  PD     PHELPS DODGE CORP                     PAGERDUTY INC
  PX     PRAXAIR INC                           P10 INC
  S      SPRINT NEXTEL CORP                    SENTINELONE INC
  SGP    SCHERING PLOUGH CORP                  SPYGLASS PHARMA INC
  SHLD   SEARS HOLDINGS CORP                   GLOBAL X FUNDS
  SLE    SARA LEE CORP                         SUPER LEAGUE ENTERPRISE INC
  STI    SUNTRUST BANKS INC                    SOLIDION TECHNOLOGY INC
  SUN    SUNOCO INC                            SUNOCO LP
  TEK    TEKTRONIX INC                         BLACKROCK E T F TRUST
  UST    U S T INC                             PROSHARES TRUST
  VIA    PARAMOUNT GLOBAL                      Via Transportation, Inc.

Of those 31 symbols, 19 were already quoted in June 2010 under the new owner.
A 2006-2016 backtest keyed on the 2005 symbol list would have returned prices for 19 companies it never meant to hold, and nothing at all for 178 it did.
```

**What this tells us**

Accuracy falls at roughly two percentage points for every year of list age, and the slope does not flatten. The 2020 list, five and a half years old, still returns the right company for 89.4% of its symbols. Ten years out the figure is 75.6%, at twenty years 55.7%, at thirty years 37.4%. A ticker list is perishable, and its shelf life is shorter than the horizon of most long-run studies.

Composition matters more than the total. Silent substitutions climb from 2 symbols on the 2020 list to 65 on the 1996 list, 15.0% of everything on it, and the share of failures that are silent climbs alongside: 3.8% for 2020, 14.8% for 2005, 24.0% for 1996. An old list does not simply lose more names. A larger fraction of what it loses has been quietly replaced rather than dropped.

The 2005 detail shows what replacement looks like. Each 2005 holder is named there by the label its identifier carries today, since a symbol has no memory of the company but an identifier does. APC belonged to Anadarko Petroleum until Occidental bought the company in 2019, and now belongs to ARKO Petroleum, a fuel distributor listed on Nasdaq; both are petroleum companies, which is exactly the coincidence that survives a sanity check. PD was Phelps Dodge, a copper miner Freeport acquired in 2007, and now answers for PagerDuty; MEDI was MedImmune until AstraZeneca bought it, and now returns an exchange traded fund rather than a company at all.

BR shows the timing problem. Burlington Resources left the index when ConocoPhillips bought it in March 2006, and Broadridge Financial Solutions took the symbol after its 2007 spin-off from ADP. A study of the 2005 roster running to 2016 receives nine years of Broadridge under a heading that reads Burlington Resources, and every one of those prices is real. Since 19 of the 31 reassigned symbols already had a new owner quoted by June 2010, most substitutions were live inside an ordinary backtest window.

**So what?**

Store universes as identifiers rather than symbols. The conversion costs one call at the moment the list is built and never expires; attempted later it cannot be done properly, because the symbol no longer knows which company it meant.

When a legacy ticker list is all that exists, run the classification above first. The empty responses are the safe failures, since a shrinking row count is visible. The populated ones need the work: confirm that the company answering today is the one that sat in the index on the date the list was made, then drop or re-map the rest.

Size that effort by the age of the list. Under five years, one symbol in 250 has changed hands and a spot check will do. At twenty years the rate is one in fifteen, roughly 30 wrong companies in a 500-name screen, enough to move a decile sort. At thirty years almost two thirds of the list is wrong, and one symbol in seven is wrong without saying so.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
