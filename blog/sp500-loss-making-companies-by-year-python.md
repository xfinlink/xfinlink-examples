# S&P 500 Companies Losing Money by Year, 2008-2025

November 24, 2026 · INDEX-UNIVERSE

Description: S&P 500 companies losing money by year: 87 members reported a net loss in 2008, 79 in 2020 and 17 in 2013; today's roster shows more in 16 of 18 years.

In fiscal 2008, 87 of the 478 S&P 500 members with an annual figure on file (18.2%) reported a net loss, the most in any year from 2008 to 2025; 2020 came next with 79 (15.9%), and 2013 had the fewest, 17 (3.5%). Rebuilt from today's member list, the same history shows a higher share of loss-makers in 16 of the 18 years, by 1.5 percentage points on average, because the companies it adds, counted before they joined, lost money far more often.

## How many S&P 500 companies report a net loss in a typical year?

A net loss is annual net income below zero, after interest, taxes, write-downs and one-off charges. The number of members in that position is a base rate for anyone who screens out unprofitable companies or builds a profitability factor. S&P Dow Jones Indices lists [positive GAAP earnings](https://press.spglobal.com/2026-06-04-S-P-Dow-Jones-Indices-Consultation-on-Treatment-of-MegaCap-Companies-Results) among the criteria for joining the index, and [a post on new additions and that earnings test](/blog/sp500-additions-profitability-rule-python) covers entry; this count covers every member, every year.

The roster matters because the S&P 500 is not a fixed list: today's constituents traced back drop the members removed since and add companies then outside the index.

## Matching 18 year-end rosters to annual net income

1. Rosters: the S&P 500 on 31 December 2008-2025 from `xfl.index("sp500", as_of=...)`, and today's roster of 24 November 2026. Tickers resolve to the company that files the statements, so GOOGL and GOOG count once.
2. Earnings: annual net income attributable to the parent (`net_income`). Fiscal years ending June to December count in that calendar year, January to May in the year before (Walmart's year to January 2021 is 2020).
3. As on file a year later: figures on file at the end of the following year, via `as_of` ([point-in-time reference](/docs/point-in-time)), or today's figure where none was on file by then (30 company-years, mostly 2008-2009). Each company-year is the fiscal year ending in the company's usual month; a 12-month window ending in another month within 300 days of a neighbouring year-end is not a fiscal year.
4. Sample: fiscal 2008 to 2025. Members without a usable annual figure leave that year's count, which keeps 478 to 500 a year.
5. Today's roster traced back uses each current member's own figures. The real roster splits into members still in the index and members removed since, today's into the same survivors and companies not yet in the index; each share column is that group's percentage of loss-makers.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE, YEARS = "2026-11-24", range(2008, 2026)
chunks = lambda xs, n=100: [xs[i:i + n] for i in range(0, len(xs), n)]

# S&P 500 rosters on each 31 December, and today's
rosters = {y: xfl.index("sp500", as_of=f"{y}-12-31") for y in YEARS}
rosters["today"] = xfl.index("sp500")

# Share class -> company, from each ticker resolved on the first roster date it appears
issuer = {}
for k, r in rosters.items():
    d = RUN_DATE if k == "today" else f"{k}-12-31"
    new = r.dropna(subset=["entity_id", "ticker"])
    new = new[~new["entity_id"].astype(int).isin(issuer)]
    for c in chunks(sorted(set(new["ticker"]))):
        try:
            meta = xfl.fundamentals(c, period_type="annual", start=d[:4] + "-01-01", end=d,
                                    fields=["net_income"]).attrs["meta"]
        except xfl.XfinlinkError:
            continue
        res = meta.get("tickers_resolved") or {}
        for sec, t in zip(new["entity_id"].astype(int), new["ticker"]):
            if t in res and res[t] != sec:
                issuer.setdefault(sec, int(res[t]))
left = {int(s): t for r in rosters.values() for s, t in zip(r["entity_id"], r["ticker"])
        if pd.notna(s) and pd.notna(t) and int(s) not in issuer}
for c in chunks(sorted(set(left.values())), 10):
    for v in xfl.resolve(c, include=["events"])["data"].values():
        for e in v.get("entities", []):
            if e.get("entity_id") in left and e.get("issuer_entity_id"):
                issuer[e["entity_id"]] = int(e["issuer_entity_id"])

def companies(r):
    out = {}
    for s, n in zip(r["entity_id"].dropna().astype(int), r.dropna(subset=["entity_id"])["entity_name"]):
        out.setdefault(issuer.get(s, n), set()).add(n)
    return out

cos = {k: companies(r) for k, r in rosters.items()}
ids_of = lambda k: {c for c in cos[k] if isinstance(c, int)}
today = ids_of("today")

# Annual net income as on file at the end of the following year (src 0), and as on file today (src 1)
pit = [xfl.fundamentals(entity_id=c, period_type="annual", as_of=min(f"{y + 1}-12-31", RUN_DATE),
                        start=f"{y}-06-01", end=f"{y + 1}-05-31", fields=["net_income", "revenue"]).assign(src=0)
       for y in YEARS for c in chunks(sorted(ids_of(y) | today))]
every = sorted(set().union(*[ids_of(k) for k in cos]))
now = [xfl.fundamentals(entity_id=c, period_type="annual", start="2008-06-01", end="2026-05-31",
                        fields=["net_income", "revenue"], max_rows=100_000).assign(src=1) for c in chunks(every)]
F = pd.concat(pit + now, ignore_index=True)
F["year"] = (F["period_end"] - pd.DateOffset(months=5)).dt.year   # Jan-May year-ends count in the year before

# One row per company-year, at the usual year-end month (no stray 12-month windows)
F["m"] = F["period_end"].dt.month
usual = F.groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax())
usual.update(F[F["revenue"].notna()].groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax()))
F["off"] = ((F["m"] - F["entity_id"].map(usual)) % 12).map(lambda d: min(d, 12 - d))
ends = F.drop_duplicates(["entity_id", "period_end"]).sort_values(["entity_id", "period_end"])
g = ends.groupby("entity_id")["period_end"]
ends["near"] = np.minimum((ends["period_end"] - g.shift()).dt.days.fillna(999),
                          (g.shift(-1) - ends["period_end"]).dt.days.fillna(999))
F = F.merge(ends[["entity_id", "period_end", "near"]], on=["entity_id", "period_end"])
F = F[F["net_income"].notna() & ~((F["off"] >= 2) & (F["near"] < 300))]
NI = (F.sort_values(["src", "off", "period_end"], ascending=[True, True, False])
       .drop_duplicates(["entity_id", "year"]).set_index(["entity_id", "year"]))

def tally(group, y):
    have = [c for c in group if (c, y) in NI.index]
    return len(have), sum(NI.loc[(c, y), "net_income"] < 0 for c in have)

for y in YEARS:
    shared = {c for c, n in cos[y].items() if isinstance(c, int) and len(n) > 1}
    members = [c for c in cos[y] if c not in shared]
    n, k = tally(members, y)
    nt, kt = tally(list(today), y)
    ns, ks = tally([c for c in members if c in today], y)
    ng, kg = tally([c for c in members if c not in today], y)
    nl, kl = tally([c for c in today if c not in cos[y]], y)
    print(y, len(cos[y]), n, k, f"{k / n:.1%} today {kt / nt:.1%} still in {ks / ns:.1%} "
          f"removed {kg / ng:.1%} not yet {kl / nl:.1%} removed loss-makers {kg}")
```

Full script with formatting and visualisation: [sp500-loss-making-companies-by-year-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-loss-making-companies-by-year-python.py)

## Output

![Bar chart of the share of S&P 500 members reporting a net loss each fiscal year, 2008-2025, on the roster in force each 31 December: 87 companies (18.2%) in 2008, 75 in 2009, 47 and 53 in 2015-2016, 79 (15.9%) in 2020 and 27 (5.4%) in 2025; a line for today's roster traced back sits above the bars in 16 of 18 years](/blog-images/sp500-loss-making-companies-by-year-python.png)

```
18 year-end rosters 2008-2025 plus today's (2026-11-24); 835 companies; figures on file a year later, else today's (30 company-years)

year  members  covered  losses   share   today    gap  still in  removed  not yet  removed, losses
2008      499      478      87   18.2%   17.8%   -0.4     15.4%    21.7%    21.8%       46 of 87
2009      500      485      75   15.5%   16.4%   +1.0     12.9%    19.0%    22.8%       39 of 75
2010      499      486      28    5.8%    8.5%   +2.8      3.5%     9.0%    17.6%       18 of 28
2011      498      485      28    5.8%    6.9%   +1.1      3.4%     9.5%    13.5%       18 of 28
2012      499      486      36    7.4%    9.0%   +1.6      6.2%     9.4%    14.6%       17 of 36
2013      499      487      17    3.5%    5.8%   +2.3      2.6%     5.1%    12.6%        9 of 17
2014      500      487      21    4.3%    6.9%   +2.5      2.5%     7.7%    16.2%       13 of 21
2015      500      486      47    9.7%   10.9%   +1.2      6.9%    15.6%    20.6%       24 of 47
2016      501      492      53   10.8%   11.6%   +0.8      8.5%    16.4%    20.5%       23 of 53
2017      501      494      41    8.3%    9.4%   +1.1      5.9%    15.6%    21.5%       19 of 41
2018      501      496      28    5.6%    7.9%   +2.2      3.4%    13.4%    25.0%       15 of 28
2019      501      497      33    6.6%    8.6%   +2.0      4.0%    17.5%    29.9%       17 of 33
2020      501      497      79   15.9%   16.8%   +0.9     13.9%    25.6%    32.1%       22 of 79
2021      501      498      20    4.0%    7.2%   +3.1      3.8%     5.2%    27.9%        4 of 20
2022      500      500      38    7.6%    9.7%   +2.1      6.2%    16.9%    35.6%       11 of 38
2023      500      500      32    6.4%    7.2%   +0.8      4.9%    19.2%    28.0%       10 of 32
2024      500      500      24    4.8%    6.0%   +1.2      4.1%    13.5%    30.6%        5 of 24
2025      500      498      27    5.4%    5.4%   +0.0      5.2%    11.1%    11.1%        2 of 27

Coverage: 478 to 500 companies a year (95.8% to 100.0% of members)
Most loss-makers: 2008 87 (18.2%), 2020 79 (15.9%), 2009 75 (15.5%)
Fewest: 2013 17 (3.5%); median 32 a year
Today's roster traced back: loss share above the real roster in 16 of 18 years, mean gap +1.5 points, widest +3.1 in 2021
Loss-makers later removed from the index: 312 of 714 company-years (43.7%); removed members are 25.9% of member-years
2008 largest losses among covered members ($bn): American International Group Inc -99.3, Citigroup Inc -27.7, Merrill Lynch & Co Inc -27.6
2009 largest losses among covered members ($bn): American International Group Inc -10.9, Chesapeake Energy Corp -5.8, Twenty First Century Fox Inc -3.4
2015 largest losses among covered members ($bn): A P A Corp -23.1, Chesapeake Energy Corp -14.7, Devon Energy Corp New -14.5
2020 largest losses among covered members ($bn): Exxon Mobil Corp -22.4, Occidental Petroleum Corp -14.8, Delta Air Lines Inc -12.4
```

## Crisis years, and the companies a survivor list adds

Losses cluster in three episodes. The financial crisis put 87 members in the red in 2008, including AIG's $99.3 billion net loss, Citigroup's $27.7 billion and Merrill Lynch's $27.6 billion, and 75 in 2009. The oil and gas price slump lifted the count to 47 in 2015 and 53 in 2016; Apache (now APA) lost $23.1 billion in 2015, followed by Chesapeake and Devon. The pandemic year produced 79 loss-makers, among them Exxon Mobil ($22.4 billion), Occidental and Delta. Outside those years the count stayed between 17 and 41, or 3.5% to 8.3% of members.

Survivorship bias usually flatters a history. Members later removed lost money more often than those still in the index (17.5% against 4.0% in 2019), and leaving them out alone would lower the count. But today's roster also brings in companies that joined later, and 11.1% to 35.6% of them lost money by year. That effect wins in 16 of 18 years: the traced-back list runs 1.5 points above the real roster on average and 3.1 points above in 2021 (7.2% against 4.0%); 2008 is 0.4 points below, and in 2025 the two lists nearly coincide.

Losses also gather among later leavers: 312 of the 714 loss-making member-years (43.7%) belong to companies removed since, against 25.9% of all member-years.

## A loss base rate for screens and backtests

In an ordinary year 3.5% to 8.3% of S&P 500 members report a net loss; in 2008, 2009 and 2020 it was 15.5% to 18.2%. A profitability screen removes a thin slice in most years and a much larger one in a downturn, when the largest losses came from financial companies (2008), oil and gas producers (2015) and energy companies and airlines (2020), so a backtest using one should expect its sector weights to move in those years.

Count on the roster in force on each date, which the [historical S&P 500 constituents](/historical-sp500-constituents) page lists back to 1925, with earnings as on file at the time; today's list swaps removed companies for future ones and makes loss-makers look more common than they were. [A year-by-year count of dividend payers](/blog/sp500-dividend-payers-by-year-python) applies the same roster method to another company trait, and [a point-in-time study of index profit](/blog/sp500-earnings-concentration-point-in-time-python) covers how earnings are spread across members.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
