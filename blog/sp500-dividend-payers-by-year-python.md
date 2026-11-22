# S&P 500 Dividend Payers by Year: 95% in 1980, 81% in 2025

November 22, 2026 · INDEX-UNIVERSE

Description: S&P 500 dividend payers by year: 474 of 500 companies paid in 1980, 356 in 2002 and 406 in 2025; today's roster traced back shows a lower share every year.

On the S&P 500 roster in force each December, 406 of 500 companies (81.2%) paid a cash dividend in 2025, down from 474 (94.8%) in 1980, with a low of 356 (71.3%) in 2002. Rebuilt from today's member list, the history shows a smaller share of payers in all 46 years, by 6.1 percentage points on average and 13.7 points in 1997.

## How many S&P 500 companies pay dividends, and on which roster?

The share of members that pay a dividend is a base rate for sizing a dividend universe or comparing payers with non-payers. The S&P 500 has never been a fixed list: only 86 of the 500 companies on the December 1980 roster are in the index today.

The shortcut is to take today's constituents and look up each one's dividend record in past years. That answers a different question, because companies removed since drop out of the count and companies then outside the index come in. The two changes can pull in opposite directions, so the net effect has to be measured.

## Counting payers on 47 year-end rosters

1. Rosters: the S&P 500 on the last business day of each December, 1979-2025, from `xfl.index("sp500", as_of=...)` ([index membership reference](/docs/index-membership)), and today's roster of 22 November 2026. Share classes of one company (GOOG and GOOGL) and tracking stocks count once.
2. Payer: a company that paid at least one cash dividend with an ex-date in the calendar year, from the monthly `dividend` field of `xfl.prices`. Special dividends count, so the test does not depend on how a payment is labelled; at most four companies a year paid only specials. A lone payment under 2 cents a share, with no payment in the year before or after, is treated as a rights-plan redemption, not a dividend (12 cases, such as Navistar's [$0.01 per right in October 2001](https://www.sec.gov/Archives/edgar/data/0000808450/000080845001500013/nic4q-8a.htm)).
3. Denominator: members with a December price that year (in 1983, eight members added that month start trading in 1984 and are set aside).
4. Today's roster traced back: each current member's own payment history, counted in years when it traded in December.
5. The real roster splits into members still in the index and members removed since; today's roster into the same survivors and companies not yet in the index. Stopped and started count members on consecutive rosters that paid in one year and not the other.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# S&P 500 rosters on the last business day of each December 1979-2025, and today's
dates = pd.date_range("1979-12-01", "2025-12-31", freq="BYE")
rosters = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in dates}
rosters["today"] = xfl.index("sp500")

def companies(r):
    """One entry per company: index lines that share a company name count once."""
    r = r.dropna(subset=["entity_id"])
    return r.assign(e=r["entity_id"].astype(int)).groupby("entity_name")["e"].apply(frozenset).tolist()

cos = {k: companies(r) for k, r in rosters.items()}
ids = sorted(set().union(*[set(r["entity_id"].dropna().astype(int)) for r in rosters.values()]))

# Monthly bars with each month's cash dividends, 1979-2025
px = pd.concat(xfl.prices(entity_id=ids[i:i + 50], start="1979-01-01", end="2025-12-31", interval="1mo",
                          fields=["close", "dividend"], max_rows=250_000)
               for i in range(0, len(ids), 50)).reset_index(drop=True)
px["year"] = px["date"].dt.year
dec = set(zip(px.loc[px["date"].dt.month == 12, "entity_id"], px.loc[px["date"].dt.month == 12, "year"]))
pay = px[px["dividend"] > 0]
years_paid = set(zip(pay["entity_id"], pay["year"]))
alone = pay.groupby(["entity_id", "year"])["dividend"].transform("size").eq(1) & np.array(
    [(e, y - 1) not in years_paid and (e, y + 1) not in years_paid for e, y in zip(pay["entity_id"], pay["year"])])
rights = pay[alone & (pay["dividend"] < 0.02)]
pay = pay.drop(rights.index)
P = set(zip(pay["entity_id"], pay["year"]))
only_special = pay.groupby(["entity_id", "year"])["dividend_type"].agg(lambda s: set(s) == {"special_dividends"})
SPECIAL = set(only_special[only_special].index)

traded = lambda c, y: any((e, y) in dec for e in c)           # had a December bar that year
paid = lambda c, y: any((e, y) in P for e in c)
def share(group, y):
    g = [c for c in group if traded(c, y)]
    return len(g), sum(paid(c, y) for c in g)

today = cos["today"]
in_today = set().union(*today)
rows = []
for y in range(1980, 2026):
    roster = cos[y]
    on_roster = set().union(*roster)
    stay = [c for c in roster if c & in_today]                 # still in the index today
    gone = [c for c in roster if not c & in_today]             # removed since
    later = [c for c in today if not c & on_roster]            # today's members, not yet in the index
    both = [c for c in roster if c & set().union(*cos[y - 1]) and traded(c, y - 1)]
    n, p = share(roster, y)
    nt, pt = share(today, y)
    row = {"year": y, "companies": n, "payers": p, "share": p / n, "today_n": nt, "today_share": pt / nt}
    for k, g in [("stay", stay), ("gone", gone), ("later", later)]:
        gn, gp = share(g, y)
        row[k + "_n"], row[k] = gn, gp / gn if gn else np.nan
    row["stopped"] = sum(paid(c, y - 1) and not paid(c, y) for c in both)
    row["started"] = sum(paid(c, y) and not paid(c, y - 1) for c in both)
    row["special_only"] = sum(paid(c, y) and all((e, y) not in P or (e, y) in SPECIAL for e in c)
                              for c in roster if traded(c, y))
    rows.append(row)
t = pd.DataFrame(rows).set_index("year")
t["gap"] = t["today_share"] - t["share"]

# ---------------- report ----------------
print(f"{len(dates)} December rosters 1979-2025 plus today's, {len(ids):,} securities; "
      f"{len(rights)} lone sub-2-cent payments set aside as rights redemptions")
print(f"\n{'year':>4}{'members':>9}{'payers':>8}{'share':>8}{'today':>8}{'gap':>7}"
      f"{'still in':>10}{'removed':>9}{'not yet':>9}{'stopped':>9}{'started':>9}")
for y in [1980, 1985, 1990, 1995, 2000, 2001, 2002, 2005, 2008, 2009, 2010, 2015, 2019, 2020, 2021, 2025]:
    r = t.loc[y]
    print(f"{y:>4}{r.companies:9.0f}{r.payers:8.0f}{r.share:8.1%}{r.today_share:8.1%}{r.gap * 100:+7.1f}"
          f"{r.stay:10.1%}{r.gone:9.1%}{r.later:9.1%}{r.stopped:9.0f}{r.started:9.0f}")
low = t["payers"].idxmin()
print(f"\nPayers: {t.loc[1980, 'payers']:.0f} of {t.loc[1980, 'companies']:.0f} in 1980, low {t.loc[low, 'payers']:.0f} "
      f"({t.loc[low, 'share']:.1%}) in {low}, {t.loc[2025, 'payers']:.0f} of {t.loc[2025, 'companies']:.0f} in 2025")
print(f"Today's roster traced back: {t.loc[1980, 'today_n']:.0f} companies trading in 1980 ({t.loc[1980, 'today_share']:.1%} payers), "
      f"low {t['today_share'].min():.1%} in {t['today_share'].idxmin()}")
print(f"Gap (today's roster minus real roster): mean {t['gap'].mean() * 100:+.1f} points, "
      f"widest {t['gap'].min() * 100:+.1f} in {t['gap'].idxmin()}, below zero in {(t['gap'] < 0).sum()} of {len(t)} years")
print(f"Years whose only payments were special dividends: at most {t['special_only'].max():.0f} companies a year")
for y in [1980, 2000]:
    r = t.loc[y]
    print(f"{y}: real roster = {r.stay_n:.0f} still in the index ({r.stay:.1%}) + {r.gone_n:.0f} removed since ({r.gone:.1%}); "
          f"today's roster = {r.stay_n:.0f} still in + {r.later_n:.0f} not yet members ({r.later:.1%})")
ids95, ids02 = set().union(*cos[1995]), set().union(*cos[2002])
joined = [c for c in cos[2002] if not c & ids95 and traded(c, 2002)]
left = [c for c in cos[1995] if not c & ids02 and traded(c, 1995)]
print(f"1995 to 2002: {len(joined)} companies joined ({np.mean([paid(c, 2002) for c in joined]):.1%} paid in 2002), "
      f"{len(left)} left ({np.mean([paid(c, 1995) for c in left]):.1%} paid in 1995)")
s = t.loc[2001:2025]
print(f"2001-2025 among members on consecutive rosters: {s['stopped'].sum():.0f} stopped, {s['started'].sum():.0f} started")
```

Full script with formatting and visualisation: [sp500-dividend-payers-by-year-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-dividend-payers-by-year-python.py)

## Output

![Share of S&P 500 companies that paid a dividend, 1980-2025: on the roster in force each December 94.8% in 1980, a low of 71.3% in 2002 and 81.2% in 2025; today's roster traced back shows 90.2% in 1980 and a lower share in every year](/blog-images/sp500-dividend-payers-by-year-python.png)

```
47 December rosters 1979-2025 plus today's, 1,564 securities; 12 lone sub-2-cent payments set aside as rights redemptions

year  members  payers   share   today    gap  still in  removed  not yet  stopped  started
1980      500     474   94.8%   90.2%   -4.6     96.5%    94.4%    84.1%        5        1
1985      500     449   89.8%   85.0%   -4.8     96.5%    87.8%    70.7%        6        2
1990      500     442   88.4%   81.5%   -6.9     96.1%    85.7%    63.5%        6        4
1995      499     435   87.2%   74.2%  -12.9     91.9%    85.2%    55.9%        1        4
2000      499     377   75.6%   68.7%   -6.9     85.7%    69.0%    46.3%        1        0
2001      499     361   72.3%   66.2%   -6.1     82.8%    65.2%    44.5%        7        4
2002      499     356   71.3%   65.7%   -5.7     81.3%    63.9%    43.3%       10        4
2005      500     388   77.6%   71.7%   -5.9     84.3%    72.0%    52.9%        1       10
2008      500     383   76.6%   71.0%   -5.6     82.3%    70.1%    49.3%        7        2
2009      500     367   73.4%   69.4%   -4.0     79.4%    65.9%    48.1%       13        5
2010      500     376   75.2%   69.9%   -5.3     80.7%    67.9%    46.6%        3       13
2015      500     422   84.4%   78.1%   -6.3     87.3%    78.6%    52.9%        0        4
2019      500     422   84.4%   79.1%   -5.3     85.5%    80.0%    44.6%        0        6
2020      500     414   82.8%   78.3%   -4.5     83.9%    77.5%    44.9%        2        2
2021      500     392   78.4%   75.5%   -2.9     80.5%    67.1%    43.1%       17        4
2025      500     406   81.2%   80.1%   -1.1     81.9%    63.2%    25.0%        2        6

Payers: 474 of 500 in 1980, low 356 (71.3%) in 2002, 406 of 500 in 2025
Today's roster traced back: 174 companies trading in 1980 (90.2% payers), low 65.7% in 2002
Gap (today's roster minus real roster): mean -6.1 points, widest -13.7 in 1997, below zero in 46 of 46 years
Years whose only payments were special dividends: at most 4 companies a year
1980: real roster = 86 still in the index (96.5%) + 414 removed since (94.4%); today's roster = 86 still in + 88 not yet members (84.1%)
2000: real roster = 196 still in the index (85.7%) + 303 removed since (69.0%); today's roster = 196 still in + 149 not yet members (46.3%)
1995 to 2002: 198 companies joined (49.5% paid in 2002), 198 left (83.3% paid in 1995)
2001-2025 among members on consecutive rosters: 84 stopped, 188 started
```

## Why a survivor list shows fewer payers than the real index

The payer share eased from 94.8% in 1980 to 87.2% in 1995, then fell 15.9 points in seven years to 71.3% in 2002, mostly through turnover rather than members stopping. The 198 companies that joined between the 1995 and 2002 rosters paid at a rate of 49.5% in 2002, against 83.3% in 1995 for the 198 that left. The share recovered to 84.4% by 2015, with 188 starts against 84 stops among continuing members in 2001-2025. Crises show as clusters of stops: 13 in 2009, and 17 in 2021, most of them airlines, cruise lines, hotels and casinos that suspended payments during 2020.

These counts sit close to the figures [S&P Dow Jones Indices reports via First Trust](https://www.ftportfolios.com/blogs/MarketBlog/2022/2/3/sp-500-index-dividend-payout-profile): a year-end high of 469 dividend-paying stocks in 1980 and a low of 351 in 2001 and 2002, against 474, 361 and 356 here. A company that paid early in a year and stopped before December counts here, so this measure should run slightly higher.

Today's roster understates the past share because of what it adds. The removed members it leaves out paid less often than the survivors (69.0% against 85.7% in 2000), so dropping them alone would raise the share. The companies it brings in, trading then but not yet in the index, paid far less often: 84.1% in 1980 and 46.3% in 2000. For 1980, today's roster offers 174 trading companies, 88 of them not yet members, in place of 500. The gap narrows to 1.1 points in 2025, when the two lists largely overlap.

## Payer counts for research without hindsight

A study that splits the S&P 500 into payers and non-payers in a past year needs that year's roster. Today's list makes the past index look less income-oriented than it was, by about 6 points in a typical year, and fills the non-payer side with companies that later grew into the index. The [historical S&P 500 constituents](/historical-sp500-constituents) page lists the members on any date back to 1925, including companies that later left. [A count of how many members stay in the index for 20 years](/blog/sp500-index-turnover-survivorship-python) covers the turnover behind these rosters, and [a year-by-year count of members by listing exchange](/blog/sp500-companies-nasdaq-vs-nyse-history-python) tracks the same decades from the exchange side.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
