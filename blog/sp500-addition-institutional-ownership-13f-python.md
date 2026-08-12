**Does Joining the S&P 500 Bring New Institutional Owners? 13F Event Study in Python**

August 12, 2026 · INDEX-UNIVERSE

**What's the question?**

Joining the S&P 500 forces a purchase, and the trade is usually studied through its effect on price. The ownership side gets less attention, and it is the part that persists after the price effect fades.

Form 13F measures it. Managers holding more than $100 million in US equities file their positions each quarter, one row per security. The count of managers reporting a position is the breadth of a company's institutional owner base; the shares reported by the three firms that run most of the world's index money are its passive core.

That splits the question in two: whether membership widens the owner base or merely reshuffles it, and how much of the company the index complex already owned beforehand. A total-market fund holds a mid-cap long before the S&P 500 committee notices it.

**The approach**

Additions come from the index event log, read by entity identifier rather than ticker, so a later symbol change cannot misroute the filing history.

1. Take every S&P 500 addition effective between 1 January 2018 and 31 December 2024, which gives 141 events.
2. Anchor each event on the last quarter end at least ten days before the effective date, the final snapshot filed while the company sits outside the index. Measure two quarters either side.
3. Per company-quarter, count the institutions reporting a common-stock position and compute the share of reported institutional shares held by BlackRock, Vanguard and State Street.
4. For each addition quarter, draw eight control companies that sat in the index at both ends of the window and never entered or left it between 2016 and 2026, measured over the same calendar quarters.
5. Keep a name when the five-quarter window is complete and at least 50 institutions report in every quarter. Companies filing across two share classes drop out, since share counts are not comparable between classes.

Neither measure breaks on a stock split.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

BIG3 = {2432, 10139, 486}          # BlackRock, Vanguard, State Street
OFFSETS = [-2, -1, 0, 1, 2]

ev = xfl.index_events("sp500", start="2018-01-01", end="2024-12-31",
                      event_type="added", limit=1000)
ev["effective_date"] = pd.to_datetime(ev["effective_date"])
ev["q0"] = ev["effective_date"].apply(
    lambda d: pd.Period(d - pd.Timedelta(days=10), freq="Q") - 1)

def snapshot(eid, q):
    h = xfl.holdings(entity_id=eid, quarter=q, security_class="COM",
                     fields=["manager_id", "shares"], max_rows=5000)
    if len(h) == 0 or h["manager_id"].duplicated().any() or not 50 <= len(h) < 5000:
        return None
    big3 = h.loc[h["manager_id"].isin(BIG3), "shares"].sum()
    return len(h), big3 / h["shares"].sum()

rows = []
for _, r in ev.iterrows():
    vals = [snapshot(int(r["entity_id"]), str((r["q0"] + o).end_time.date()))
            for o in OFFSETS]
    if all(v is not None for v in vals):
        rows.append({f"h{o}": v[0] for o, v in zip(OFFSETS, vals)})

a = pd.DataFrame(rows)
inclusion = np.log(a["h1"] / a["h0"])
prior = np.log(a["h0"] / a["h-1"])
print(100 * (np.exp(inclusion.median()) - 1), 100 * (np.exp(prior.median()) - 1))
```

Full script with formatting and visualisation: [sp500-addition-institutional-ownership-13f-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-addition-institutional-ownership-13f-python.py)

**Output**

```
S&P 500 additions 2018-2024: 141 events
Sample with a complete five-quarter filing window: 107 additions, 199 continuing-member controls

Median number of institutions reporting a position, by quarter
  quarter   additions   members
    -2           480       933
    -1           498       946
     0           534       933
    +1           603       971
    +2           625      1029

Median quarter-on-quarter change in the number of reporting institutions
  window        additions   members
  -2 -> -1         +4.55%    +1.44%
  -1 ->  0         +5.82%    +2.04%
   0 -> +1         +9.49%    +1.79%   <- inclusion quarter
  +1 -> +2         +3.76%    +1.58%

Additions gaining holders in the inclusion quarter: 93.5%   (members: 68.3%)
Wilcoxon, additions: inclusion quarter vs prior quarter   p = 2.83e-05
Mann-Whitney, inclusion quarter: additions vs members     p = 2.62e-23

BlackRock + Vanguard + State Street share of reported institutional shares (median)
  quarter   additions   members
    -2          24.44%    26.51%
    -1          24.35%    26.31%
     0          24.98%    26.74%
    +1          25.98%    26.94%
    +2          26.25%    26.81%
  median change 0 -> +1:  additions +0.48pp   members +0.10pp

Largest and smallest holder gains in the inclusion quarter
  IR    INGERSOLL RAND INC           2020-03-03    207 ->   570  (+175.4%)
  VTRS  VIATRIS INC                  2020-11-17    459 ->  1233  (+168.6%)
  WBD   WARNER BROS DISCOVERY INC    2022-04-11    590 ->  1483  (+151.4%)
  WAB   WABTEC CORP                  2019-02-27    430 ->   862  (+100.5%)
  FOXA  FOX CORP                     2019-03-19    729 ->   578  ( -20.7%)
  CE    CELANESE CORP DEL            2018-12-24    567 ->   556  (  -1.9%)
  MXIM  MAXIM INTEGRATED PRODUCTS IN 2018-12-03    548 ->   538  (  -1.8%)
  LVS   LAS VEGAS SANDS CORP         2019-10-03    643 ->   632  (  -1.7%)
```

**What this tells us**

The owner base widens, and the widening lands in one quarter. The median addition is held by 534 institutions in its last filing outside the index and by 603 in the first filing inside it. Within each company, median growth in that quarter is 9.49 percent, against 5.82 percent in the quarter before and 3.76 percent in the quarter after; continuing members grew 1.79 percent over the same calendar quarters.

The raw gap overstates the effect: additions were already gaining holders faster than members beforehand, which is what the committee selects for. The usable figure is the change in the gap, from 3.8 percentage points in the quarter before inclusion to 7.7 points in the inclusion quarter, leaving roughly four points traceable to membership. The effect is also broad: 93.5 percent of additions gained institutions in that quarter against 68.3 percent of members, and the paired test on within-company differences returns p = 2.83e-05.

The passive side is quieter. Two quarters before joining, the three index managers already held 24.44 percent of the median addition's reported institutional shares, close to the 26.51 percent they held of long-standing members; two quarters after, the readings are 26.25 and 26.81 percent, so a two-point gap has nearly closed. The median company-level move across the inclusion quarter is 0.48 percentage points against 0.10 for members. Total-market and mid-cap funds owned these companies well before the S&P 500 did, so inclusion shifts passive weight rather than creating it.

The extremes carry a caveat. Ingersoll Rand, Viatris and Warner Bros Discovery each took their index place through a merger completed inside the window, so the pre-inclusion snapshot describes a predecessor with a smaller register. Fox Corp is that shape reversed, down 20.7 percent after being carved out of a larger parent. Medians and rank tests absorb these cases; a mean would not.

**So what?**

Model an index addition as a change in breadth first, passive weight second. The index complex adds under half a percentage point of the company's institutional shares at the median, a thin basis for a trade. The register gains hundreds of managers, and that is what moves borrow availability, quoted depth and the shareholder vote. The window is short: holder growth falls back to 3.76 percent in the quarter after inclusion, near the pace before the event.

Two habits follow for research design. Build the sample on entity identifiers and point-in-time membership, since a name that changed symbol after joining will otherwise attach to whoever holds that symbol today. Carry a control group of continuing members too: without one, the pre-inclusion drift of 5.82 percent per quarter counts as an index effect and the measured impact roughly doubles.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
