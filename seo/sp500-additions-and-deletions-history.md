# S&P 500 Additions and Deletions: Full History by Date

S&P Dow Jones Indices changes the S&P 500 by press release, naming the companies that join and leave and the day the change takes effect before the open of trading. Some changes are timed to the quarterly rebalance; the rest follow an acquisition or a spin-off and happen when the deal closes. The full history of S&P 500 additions and deletions is therefore a dated event log, one row per addition and one per deletion, and it is most useful next to a roster showing who was in the index on any past date. xfinlink serves both: `index_events()` returns every change since the first one on record, in August 1928, nearly three decades before the index widened to 500 stocks in March 1957, and `index(as_of=...)` returns the membership on any date back to the end of 1925, companies later removed, acquired or delisted included.

## How Do S&P 500 Additions and Deletions Work?

Each change arrives in an announcement on press.spglobal.com. The September 2026 release, dated 4 September, added Bloom Energy, Everpure and Illumina and removed Molson Coors, The Trade Desk and Builders FirstSource, "effective prior to the open of trading on Monday, September 21, 2026, to coincide with the quarterly rebalance." The stated reason was that the changes "ensure that each index is more representative of its market capitalization range." The March, September and December 2025 releases and the June 2024 release tie their changes to the quarterly rebalance in the same words (all checked on press.spglobal.com, 8 October 2026).

Changes between rebalances follow corporate events. Honeywell's spin-off of Solstice Advanced Materials, completed on 30 October 2025, put Solstice into the index in place of CarMax, and DuPont's spin-off of Qnity Electronics replaced Eastman Chemical four days later; the December 2025 rebalance then moved Solstice out to the S&P SmallCap 600 (S&P Dow Jones Indices releases of 27 October and 5 December 2025). A corporate event can also decide who leaves at a scheduled date. Tesla joined on 21 December 2020 in place of Apartment Investment and Management, which the release of 11 December 2020 said would no longer be representative of the S&P Composite 1500 indices market cap ranges after its own spin-off.

In 2025 the two kinds were equal in number. Of the 40 S&P 500 events that year, 20 fell on the Friday and Monday of the March, September and December rebalances, and 20 came between them.

## Which Date Does an Addition or Deletion Carry?

Three dates attach to every change, and a backtest has to know which one it is reading. The announcement is the day the market learns of it. S&P's own summary tables date a deletion by the first session without the stock. A membership record dates it by the last session the stock was in. For additions the last two agree.

| Change | Announced | S&P table date | xfinlink `effective_date` |
|---|---|---|---|
| Tesla added | 16 Nov 2020 | 21 Dec 2020 | 2020-12-21 (first day in) |
| Solstice added | 27 Oct 2025 | 30 Oct 2025 | 2025-10-30 (first day in) |
| CarMax deleted | 27 Oct 2025 | 31 Oct 2025 | 2025-10-30 (last day in) |
| Solstice deleted | 5 Dec 2025 | 22 Dec 2025 | 2025-12-19 (last day in) |

Announcement and effective dates are from the S&P Dow Jones Indices releases named above and, for Tesla, the release of 16 November 2020 that first announced its addition, checked on 8 October 2026. In all five quarterly releases checked for this guide (June 2024, March, September and December 2025, September 2026), the announcement came 17 calendar days before the effective date. Spin-off changes moved faster: Solstice was announced three days before it joined. A strategy that trades on the news needs the announcement date; a strategy that only holds index members needs the effective date, because that is the day index funds own the stock.

## Where Can You Get the Full History of S&P 500 Changes?

The press archive is the primary source and costs nothing, which makes it the right place to confirm a single change. It is a set of announcements rather than a table, though, so a decade of history means reading and transcribing release after release.

Free trackers fill part of the gap. Chartrow, which ranks for this search, builds its list from the quarterly holdings reports of an S&P 500 fund: it dates each change to the window between two filings rather than to a day, opens its record with the 30 September 2019 filing, and states that companies joining and leaving between two filings do not appear (chartrow.com, as of 8 October 2026). For a glance at recent years that is enough. Solstice, a member for seven weeks, is exactly the kind of spell such a list cannot see.

A backtest needs the day-level record. The xfinlink event log holds more than 3,600 S&P 500 events from 24 August 1928 onward, keyed by an `entity_id` that stays with the company through ticker changes, the same key that prices and fundamentals use. Here is the fourth quarter of 2025:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# 1. The dated log: every S&P 500 addition and deletion in Q4 2025
ev = xfl.index_events("sp500", start="2025-10-01", end="2025-12-31")
print(ev[["effective_date", "event_type", "ticker"]].to_string(index=False))

# 2. Two quarter-end snapshots against the log
sep = set(xfl.index("sp500", as_of="2025-09-30")["entity_id"].dropna())
dec = set(xfl.index("sp500", as_of="2025-12-31")["entity_id"].dropna())
print(len(dec - sep), "joined between snapshots;", (ev["event_type"] == "added").sum(), "additions in the log")

# 3. Membership flips on the effective date (Tesla, December 2020)
for d in ["2020-12-18", "2020-12-21"]:
    r = set(xfl.index("sp500", as_of=d)["ticker"])
    print(d, "TSLA" in r, "AIV" in r)
```

```
effective_date event_type ticker
    2025-10-30      added   SOLS
    2025-10-30    removed    KMX
    2025-11-03      added      Q
    2025-11-03    removed    EMN
    2025-11-26    removed    IPG
    2025-11-28      added   SNDK
    2025-12-10    removed      K
    2025-12-11      added   ARES
    2025-12-19    removed    LKQ
    2025-12-19    removed    MHK
    2025-12-19    removed   SOLS
    2025-12-22      added   CVNA
    2025-12-22      added    FIX
    2025-12-22      added    CRH
6 joined between snapshots; 7 additions in the log
2020-12-18 False True
2020-12-21 True False
```

The two quarter-end rosters differ by six companies; the log records seven additions. The seventh is Solstice, in on 30 October and out after 19 December, invisible to any method that compares snapshots. The Tesla lines show the date convention at work: on Friday 18 December 2020 the roster still holds Apartment Investment and Management, and on Monday 21 December it holds Tesla.

Parameters, the founding roster and the treatment of companies that rejoin are documented on the [index events reference](/docs/index-events). Illumina, for one, left in June 2024 for the S&P MidCap 400 and came back in September 2026 under the same `entity_id`.

## How Do You Use Additions and Deletions in a Backtest?

Pick the universe on each rebalance date from the roster as it stood that day, and let the event log tell you when it changed. Building the universe from today's constituents instead hands the strategy knowledge of which companies would survive; [the survivorship bias explainer](/blog/what-is-survivorship-bias-in-backtesting) covers why that distorts results. In practice that comes down to four rules:

- Call `xfl.index("sp500", as_of=date)` for each rebalance date and trade only those `entity_id` values. A deleted company stays in every roster dated before its exit.
- Treat `effective_date` on a removal as the last day the stock counts. A position held under an index-membership rule closes at that session, not the next one.
- Use the announcement date only for strategies that trade the news, and never let a signal see a change before it was announced.
- Join on `entity_id`, not ticker, so that a reassigned symbol does not attach another company's prices to a past member.

The price behaviour around these dates is a separate question with its own posts. [An event study of S&P 500 additions](/blog/sp500-index-effect-event-study-python) measures returns either side of the effective date, and the deleted side is followed in [a comparison of removed stocks with the companies that replaced them](/blog/sp500-replacement-pairs-deletion-returns-python).

On plans: the roster with `as_of` reaches the start of the record on every plan, Free included, which covers the universe-selection step on its own. The event log returns the last year on Free and the full history from 1928 on paid plans, listed on the [pricing](/pricing) page. The [index membership reference](/docs/index-membership) documents the roster call for the S&P 500, the Nasdaq-100 and the Dow Jones Industrial Average.

## FAQ

**How often does the S&P 500 change companies?** At most quarterly rebalances (the June 2025 rebalance changed no members), and between rebalances whenever a member is acquired or spins off a business. The event log holds 20 additions and 20 deletions for calendar 2025.

**When are S&P 500 additions announced?** For quarterly changes, 17 days before they take effect in every release checked here. Changes caused by a deal are announced closer to the date: the release of 27 October 2025 came three days before Solstice joined and seven before Qnity did.

**Does a deleted company disappear from the history?** No. Its removal is an event in the log, and it appears in every roster dated before its exit, so `index("sp500", as_of="2020-12-18")` still lists Apartment Investment and Management.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
