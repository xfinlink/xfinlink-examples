# S&P 500 Members: 43 of 499 Launch-Day Listings Never Left

October 11, 2026 · INDEX-UNIVERSE

Forty-three of the 499 listings on the S&P 500's launch-day roster of 4 March 1957 are still in the index without ever having been removed, and 15 of them now trade under a different ticker. Across the whole record, which opens with the index's predecessor at the end of 1925, 1,972 listed securities have been S&P 500 members; today's 503 are about a quarter of them.

## How many companies have ever been in the S&P 500?

The published answers are old. When the index turned 50, S&P said that 86 of its original constituents had survived (PLANSPONSOR, 2 March 2007). A Reuters timeline of 10 April 2013 put the figure at 69 of the 500 original companies. Neither gives a total for the companies that have passed through the index, nor says how a merged or renamed company was counted.

Today's member list cannot answer the question: companies that were acquired, failed or were demoted leave no trace in it, the mechanism behind [survivorship bias in backtesting](/blog/what-is-survivorship-bias-in-backtesting).

The unit counted here is the listed security (one share class). Alphabet, with two classes in the index, counts twice. A reorganisation that created a new listed security counts as a new listing: Union Pacific Railroad had its last day in the index on 15 July 1969 and Union Pacific Corporation joined on 16 July. A count that follows company lineage would merge such pairs, giving fewer companies in total and more surviving originals.

## Counting listings from the event log and two rosters

1. Page through the full event log with `index_events`: 3,612 dated additions and removals; [the guide to S&P 500 additions and deletions history](/blog/sp500-additions-and-deletions-history) explains the log. Founding members carry no addition row, so the opening roster of 31 December 1925 comes from `as_of`.
2. Count distinct listings by `entity_id`, which stays with a security through renames and ticker changes (by ticker where a row has none).
3. Take the launch-day roster with `as_of="1957-03-04"`: 499 listings. Each row's `removed_date` closes the membership open that day, so rows with a blank `removed_date`, or one on or after the date checked, are the listings still in without a break.
4. Join that roster to today's on `entity_id`; a current membership that began after 4 March 1957 marks a listing removed and later re-added.
5. Repeat the counts on the dates of the two published figures.

## Code

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LAUNCH = "1957-03-04"  # first day of the 500-stock index

def all_events():
    pages = [xfl.index_events("sp500", start="1925-12-31")]
    while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
        pages.append(xfl.index_events("sp500", start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True)

def listings(df):  # one key per listed security: entity_id, or ticker where no id is linked
    linked = set(df["entity_id"].dropna().astype(int))
    return linked | set("ticker " + df.loc[df["entity_id"].isna(), "ticker"])

ev = all_events()
opening = xfl.index("sp500", as_of="1925-12-31")  # the opening roster has no addition rows
launch = xfl.index("sp500", as_of=LAUNCH)
today = xfl.index("sp500")

ever = listings(ev) | listings(opening)
since_launch = listings(ev[ev["effective_date"] >= LAUNCH]) | listings(launch)

# removed_date ends the membership that was open on launch day
launch["last_day"] = launch["removed_date"].fillna("9999-12-31")
unbroken = pd.Series({y: int((launch["last_day"] >= f"{y}-03-04").sum()) for y in range(1957, 2027)})

now = launch.merge(today[["entity_id", "ticker", "added_date"]], on="entity_id", suffixes=("_1957", "_now"))
stayed = now[now["added_date_now"] <= LAUNCH]  # current membership began on or before launch

print(len(ever), len(since_launch), len(today), len(launch), len(now), len(stayed),
      (stayed["ticker_1957"] != stayed["ticker_now"]).sum())
print(unbroken[[1967, 1977, 1987, 1997, 2007, 2017, 2026]].to_dict())
for day in ("2007-03-02", "2013-04-10"):  # dates of the S&P and Reuters counts
    on_day = set(xfl.index("sp500", as_of=day)["entity_id"].dropna())
    print(day, (launch["last_day"] >= day).sum(), len(on_day & set(launch["entity_id"])))
print(today["added_date"].str[:3].add("0s").value_counts().sort_index().to_dict())
```

Full script with formatting and visualisation: [sp500-members-1957-originals-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-members-1957-originals-python.py)

## Output

![Line chart of the 499 listings on the S&P 500 launch-day roster of 4 March 1957 that were still in the index without a break on 4 March of each year to 2026, falling below half by 1977 and to 43 today, with the published original-company counts of 86 in 2007 and 69 in 2013 marked for comparison](https://xfinlink.com/blog-images/sp500-members-1957-originals-python.png)

```
S&P 500 event log: 3,612 rows, 1928-08-24 to 2026-10-06
Listings ever in the index (record opens 1925-12-31): 1,972
Listings in the index on or after 1957-03-04:          1,944
Members today: 503 (25.5% of all listings ever)

Launch-day roster (1957-03-04): 499 listings
  in today's roster:                  53
  of which never left:                43
  removed and later re-added:         10
  never-left listings on a new ticker: 15

Launch-day listings still in without a break, on 4 March of:
  1967: 380
  1977: 247
  1987: 168
  1997: 126
  2007:  69
  2017:  51
  2026:  44
  today:  43
  left since 4 March 2026: CPB (CAMPBELLS CO, last day 2026-06-18)
  first 4 March below half the roster: 1977
  largest one-day exit: 21 on 1976-06-30 (46 removals that day, 46 additions on 1976-07-01)

On 2007-03-02: never left 69, in the index (re-additions counted) 76
On 2013-04-10: never left 55, in the index (re-additions counted) 65

Today's members by decade their current membership began:
decade
1920s      7
1940s      4
1950s     34
1960s      9
1970s     21
1980s     46
1990s     60
2000s     88
2010s    133
2020s    101

Never left since 1957, ticker then -> now (earliest current membership first):
  SD   -> CVX   CHEVRON CORP NEW                   since 1925-12-31  (new ticker)
  DTE  -> DTE   D T E ENERGY CO                    since 1925-12-31
  ED   -> ED    CONSOLIDATED EDISON INC            since 1925-12-31
  SCE  -> EIX   EDISON INTERNATIONAL               since 1925-12-31  (new ticker)
  ACD  -> HON   HONEYWELL INTERNATIONAL INC        since 1925-12-31  (new ticker)
  IP   -> IP    INTERNATIONAL PAPER CO             since 1925-12-31
  J    -> XOM   EXXON MOBIL CORP                   since 1925-12-31  (new ticker)
  PE   -> EXC   EXELON CORP                        since 1944-06-07  (new ticker)
  SO   -> SO    SOUTHERN CO                        since 1944-06-07
  PEG  -> PEG   PUBLIC SERVICE ENTERPRISE GP INC   since 1948-07-07
  NSP  -> XEL   X C E L ENERGY INC                 since 1949-07-06  (new ticker)
  AEP  -> AEP   AMERICAN ELECTRIC POWER CO INC     since 1951-09-06
  ABT  -> ABT   ABBOTT LABORATORIES                since 1957-03-01
  BA   -> BA    BOEING CO                          since 1957-03-01
  BMY  -> BMY   BRISTOL MYERS SQUIBB CO            since 1957-03-01
  CTR  -> CAT   CATERPILLAR INC                    since 1957-03-01  (new ticker)
  CL   -> CL    COLGATE PALMOLIVE CO               since 1957-03-01
  P    -> COP   CONOCOPHILLIPS                     since 1957-03-01  (new ticker)
  MES  -> CVS   C V S HEALTH CORP                  since 1957-03-01  (new ticker)
  DE   -> DE    DEERE & CO                         since 1957-03-01
  ENX  -> ETN   EATON CORP PLC                     since 1957-03-01  (new ticker)
  MSU  -> ETR   ENTERGY CORP NEW                   since 1957-03-01  (new ticker)
  F    -> F     FORD MOTOR CO DEL                  since 1957-03-01
  GD   -> GD    GENERAL DYNAMICS CORP              since 1957-03-01
  GE   -> GE    GENERAL ELECTRIC CO                since 1957-03-01
  GLW  -> GLW   CORNING INC                        since 1957-03-01
  HAL  -> HAL   HALLIBURTON CO                     since 1957-03-01
  HSY  -> HSY   HERSHEY CO                         since 1957-03-01
  AA   -> HWM   HOWMET AEROSPACE INC               since 1957-03-01  (new ticker)
  IBM  -> IBM   INTERNATIONAL BUSINESS MACHS COR   since 1957-03-01
  KMB  -> KMB   KIMBERLY CLARK CORP                since 1957-03-01
  KO   -> KO    COCA COLA CO                       since 1957-03-01
  KR   -> KR    KROGER CO                          since 1957-03-01
  MMM  -> MMM   3M CO                              since 1957-03-01
  MO   -> MO    ALTRIA GROUP INC                   since 1957-03-01
  MRK  -> MRK   MERCK & CO INC NEW                 since 1957-03-01
  MOT  -> MSI   MOTOROLA SOLUTIONS INC             since 1957-03-01  (new ticker)
  PEP  -> PEP   PEPSICO INC                        since 1957-03-01
  PFE  -> PFE   PFIZER INC                         since 1957-03-01
  PG   -> PG    PROCTER & GAMBLE CO                since 1957-03-01
  PPG  -> PPG   P P G INDUSTRIES INC               since 1957-03-01
  UA   -> RTX   R T X CORP                         since 1957-03-01  (new ticker)
  MHP  -> SPGI  S & P GLOBAL INC                   since 1957-03-01  (new ticker)

Re-added after leaving:
  SLB  -> SLB   SLB LTD                            back since 1965-03-04
  GIS  -> GIS   GENERAL MILLS INC                  back since 1969-02-13
  ADM  -> ADM   ARCHER DANIELS MIDLAND CO          back since 1981-07-30
  LK   -> LMT   LOCKHEED MARTIN CORP               back since 1984-07-19
  CMS  -> CMS   C M S ENERGY CORP                  back since 1999-05-03
  HI   -> LHX   L3HARRIS TECHNOLOGIES INC          back since 2008-09-22
  ONG  -> OKE   ONEOK INC NEW                      back since 2010-03-15
  IR   -> TT    TRANE TECHNOLOGIES PLC             back since 2010-11-17
  EQT  -> EQT   E Q T CORP                         back since 2022-10-03
  PCG  -> PCG   P G & E CORP                       back since 2022-10-03
```

## Why 43 is lower than the published counts

The roster thinned fastest early: 380 of the 499 were still in by March 1967 and 247, below half, by March 1977. The largest single exit came on 30 June 1976, when 21 launch-day listings left in a change that swapped 46 members. Between 2017 and 2026 only seven more left, and the 44th, Campbell's, had its last day on 18 June 2026.

Neither source states its unit, so the gap most likely reflects definition rather than a different history. On 2 March 2007, 69 launch-day listings had never left and 76 were in the index when re-additions count, against S&P's 86. On 10 April 2013 the same counts are 55 and 65, against Reuters' 69. A count that follows company lineage would treat a successor as the original carrying on, as with Union Pacific in 1969; a listing count does not.

Ticker changes are a separate trap. Fifteen of the 43 trade under a new symbol: Standard Oil Company (New Jersey), listed as J, is now Exxon Mobil (XOM); Melville Shoe's MES is CVS Health. Matching the 1957 roster to today's tickers would record all fifteen as departures. The ten re-added listings, Schlumberger (now SLB) and Lockheed (now Lockheed Martin) among them, are members today but not unbroken ones.

Today's index is young: 234 of its 503 members began their current membership in 2010 or later, and 45 in the 1950s or earlier. Membership length across every cohort of entrants is the subject of [a Kaplan-Meier survival analysis of S&P 500 membership](/blog/sp500-membership-kaplan-meier-hazard-python).

## Starting a 1957 universe without losing renamed members

A backtest or longevity study that begins in 1957 should start from the 499 listings on the launch-day roster, not from today's members filtered by join date, and key them by a persistent identifier rather than ticker or name. Keyed by `entity_id`, the fifteen renamed survivors keep their history and the listings that left stay in the sample until their last day.

A count of original members should state its unit, listed security or company lineage, unbroken or on the date; on one day in March 2007 the two listing counts give 69 and 76, against S&P's published 86.

Past rosters come from `as_of` on the [index membership endpoint](/docs/index-membership), and [the S&P 500 members on any date since 1925](/historical-sp500-constituents) can be looked up without code.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
