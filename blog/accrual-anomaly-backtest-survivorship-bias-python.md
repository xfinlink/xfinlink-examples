# Accrual Anomaly Backtest: Survivorship Bias in the S&P 500

November 28, 2026 · INDEX-UNIVERSE

Description: Accrual anomaly backtest, S&P 500 2011-2025: today's roster lifts the low-minus-high accrual spread to 7.41 points a year; rosters in force give 1.26 (t 0.60).

On today's S&P 500 list traced back, an accrual anomaly backtest finds low-accrual stocks beating high-accrual stocks by 7.41 points a year over 2011-2025 (t 2.63); on the roster in force at each sort date, the gap is 1.26 points (t 0.60). Survivorship bias adds 6.15 points a year to the spread, mostly through the lowest-accrual quintile, which held the most members later removed and the most companies that joined the index afterwards.

## Does the accrual anomaly hold on the index as it stood?

Accruals are the part of earnings not received as cash: net income minus operating cash flow, here scaled by total assets. A negative value means cash ran ahead of profit, usually from depreciation or write-downs. Richard Sloan reported in The Accounting Review in 1996 that how far current earnings persist depends on the relative size of their cash and accrual components, and that stock prices act as if investors "fixate" on earnings. The accrual anomaly trades on it: buy low accruals, avoid high ones.

Today's index list holds only companies that later grew into the index or stayed in it; [the guide to survivorship bias in backtesting](/blog/what-is-survivorship-bias-in-backtesting) explains how that enters a test. The question here is whether it lifts all five quintiles alike, leaving the spread intact.

## Sorting on accruals as filed at each December close

1. Universes: on the last business day of each December from 2010 to 2024, either the S&P 500 roster in force that day (`xfl.index("sp500", as_of=...)`; the [historical S&P 500 constituents](/historical-sp500-constituents) page lists it) or today's roster of 502 lines traced back, each company that traded on the sort date. [The dated history of S&P 500 additions and deletions](/blog/sp500-additions-and-deletions-history) lists the changes between them.
2. Accruals: net income minus operating cash flow, over the average of the year's opening and closing total assets, from the latest fiscal year on record by the sort date, read with `as_of` (the [point-in-time reference](/docs/point-in-time)). Companies without usable filings for a sort date drop from that year; 95.4% to 99.4% of roster lines have the inputs.
3. Financial companies (SIC codes 6000-6799) are left out. That leaves 387-405 companies a year on the roster in force and 329-396 on today's roster. Three companies are set aside for two to six years each.
4. Quintiles are cut within each universe and year, Q1 the lowest accruals, and held the next calendar year, equal weight; a stock that stops trading takes its recorded delisting return, then sits in cash. Biases are paired over 15 years.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = pd.Timestamp("2026-11-28")  # the day today's roster was read
forms = pd.date_range("2010-12-01", "2024-12-31", freq="BYE")  # last business day of December
ymd = lambda d: d.strftime("%Y-%m-%d")

FIELDS = ["net_income", "operating_cash_flow", "total_assets", "revenue", "sic"]

def statements(d, tickers=None, issuers=None):
    """Annual statements as on file at the close of d (as_of), fiscal years ending up to three years back."""
    kw = dict(period_type="annual", as_of=ymd(d), start=ymd(d - pd.DateOffset(years=3)), end=ymd(d),
              fields=FIELDS, max_rows=50_000)
    if tickers is not None:  # tickers resolve on the as_of date; rows carry the company id
        return xfl.fundamentals(tickers, **kw)
    return pd.concat([xfl.fundamentals(entity_id=issuers[i:i + 500], **kw) for i in range(0, len(issuers), 500)])

# Today's roster: security ids for prices, company ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
now = xfl.fundamentals(today["ticker"].tolist(), period_type="annual", fields=["net_income"])
sec_of = dict(zip(now["entity_id"], now["ticker"].map(dict(zip(today["ticker"], today["entity_id"])))))

F, rows = [], []
for d in forms:
    roster = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = statements(d, tickers=roster["ticker"].tolist())
    a["sec"] = a["ticker"].map(dict(zip(roster["ticker"], roster["entity_id"])))
    b = statements(d, issuers=sorted(set(sec_of) - set(a["entity_id"])))  # today's names outside it
    b["sec"] = b["entity_id"].map(sec_of)
    rows.append(pd.concat([a.assign(honest=True), b.assign(honest=False)])[["entity_id", "ticker", "entity_name", "sec", "honest"]]
                .drop_duplicates("entity_id").assign(form=d, roster=len(roster)))
    F.append(pd.concat([a, b]).drop_duplicates(["entity_id", "period_end"]).assign(form=d))
S = pd.concat(rows, ignore_index=True).dropna(subset=["sec"])
S["sec"] = S["sec"].astype(int)
S["naive"] = S["entity_id"].isin(set(sec_of))
F = pd.concat(F, ignore_index=True)

# Keep true fiscal years: the company's usual year-end month, keyed on rows that carry revenue;
# a 12-month window ending in another month within 300 days of a neighbouring year-end is dropped
F["m"] = F["period_end"].dt.month
usual = F.groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax())
usual.update(F[F["revenue"].notna()].groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax()))
F["off"] = ((F["m"] - F["entity_id"].map(usual)) % 12).map(lambda x: min(x, 12 - x))
ends = F.drop_duplicates(["entity_id", "period_end"]).sort_values(["entity_id", "period_end"])
g = ends.groupby("entity_id")["period_end"]
ends["near"] = np.minimum((ends["period_end"] - g.shift()).dt.days.fillna(999),
                          (g.shift(-1) - ends["period_end"]).dt.days.fillna(999))
F = F.merge(ends[["entity_id", "period_end", "near"]], on=["entity_id", "period_end"])
F = F[~((F["off"] >= 2) & (F["near"] < 300))]

# Latest fiscal year on record at the sort date (ended within 18 months), and the year before it
cur = F[((F["form"] - F["period_end"]).dt.days <= 548) & F[["net_income", "operating_cash_flow", "total_assets"]].notna().all(axis=1)]
cur = cur.sort_values("period_end")
cur = cur.drop_duplicates(["form", "entity_id"], keep="last")
pri = F.dropna(subset=["total_assets"])[["form", "entity_id", "period_end", "total_assets"]].rename(columns={"period_end": "pe0", "total_assets": "ta0"})
A = cur.merge(pri, on=["form", "entity_id"])
A = A[(A["period_end"] - A["pe0"]).dt.days.between(300, 430)].sort_values("pe0").drop_duplicates(["form", "entity_id"], keep="last")
A["accruals"] = (A["net_income"] - A["operating_cash_flow"]) / ((A["total_assets"] + A["ta0"]) / 2)
S = S.merge(A[["form", "entity_id", "period_end", "accruals", "sic"]], on=["form", "entity_id"], how="left")

# Monthly total returns, December 2010 to December 2025, by security
ids = sorted(S["sec"].unique())
px = pd.concat(xfl.prices(entity_id=ids[i:i + 200], start="2010-12-01", end="2025-12-31", interval="1mo",
                          fields=["return_daily", "close"], max_rows=250_000) for i in range(0, len(ids), 200))
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2010-12", "2025-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(index=R.index, columns=ids).notna()

# Exits: recorded delisting return added to the last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2025-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    g = g.index.tolist()
    for i in range(0, len(g), 300):
        d = xfl.prices(entity_id=g[i:i + 300], start=f"{y}-01-01", end="2025-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

S["y"] = S["form"].dt.year
S = S[[traded.at[pd.Period(f"{y}-12", "M"), s] for y, s in zip(S["y"], S["sec"])]]
S["sic"] = pd.to_numeric(S["sic"], errors="coerce")
n_cover = S[S["honest"]].groupby("y").agg(lines=("roster", "first"), have=("accruals", lambda s: s.notna().sum()))
# Companies sharing a company record with a merger partner before the merger, set aside for those
# years: MeadWestvaco (WestRock's record) to 2014; Tyco and Johnson Controls Inc to 2015
SET_ASIDE = {8097: 2014, 4074: 2015}
SET_ASIDE_YEARS = {670: (2012, 2013)}  # AGL Resources 2012-2013: an annual record dated ahead of its fiscal year
S = S[~(S["y"] <= S["entity_id"].map(SET_ASIDE))]
S = S[[not (e in SET_ASIDE_YEARS and SET_ASIDE_YEARS[e][0] <= y <= SET_ASIDE_YEARS[e][1])
       for e, y in zip(S["entity_id"], S["y"])]]
S = S[S["accruals"].notna() & S["sic"].notna() & ~S["sic"].between(6000, 6799)].copy()  # financials (SIC 6000-6799) left out

def held(secs, y):
    """Equal weight at the start of y+1, bought and held to December; cash after an exit."""
    return (1 + R.loc[f"{y + 1}-01":f"{y + 1}-12", secs].fillna(0)).prod() - 1

for y, g in S.groupby("y"):
    S.loc[g.index, "r"] = g["sec"].map(held(sorted(set(g["sec"])), y)).values

def quintiles(u):
    """1 = lowest accruals (earnings backed by cash) ... 5 = highest accruals."""
    return u.groupby("y")["accruals"].transform(lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)

U = {"Roster in force": S[S["honest"]].copy(), "Today's roster": S[S["naive"]].copy()}
tab, spread = {}, {}
for k, u in U.items():
    u["q"] = quintiles(u)
    tab[k] = u.groupby(["y", "q"])["r"].mean().unstack()
    spread[k] = tab[k][1] - tab[k][5]
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
for k in U:
    print(k, f"Q1-Q5 {spread[k].mean() * 100:+.2f} points a year, t {tstat(spread[k]):.2f}")
```

Full script with formatting and visualisation: [accrual-anomaly-backtest-survivorship-bias-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/accrual-anomaly-backtest-survivorship-bias-python.py)

## Output

![Average annual return by accrual quintile, S&P 500 2011-2025: roster in force 14.2%, 11.1%, 12.8%, 12.5%, 13.0%; today's roster 26.5%, 16.3%, 17.0%, 16.0%, 19.0%. Share of company-years later removed: 34%, 25%, 22%, 23%, 28%; not yet in the index: 33%, 22%, 16%, 13%, 17%](/blog-images/accrual-anomaly-backtest-survivorship-bias-python.png)

```
15 December sorts 2010-12-31 to 2024-12-31, holding years 2011-2025; today's roster 502 lines on 2026-11-28; 787 securities, 177 exit records
Roster lines with accruals on the sort date: 95.4%-99.4% a year
Roster in force: 387-405 non-financial companies a year, 5,957 company-years
Today's roster: 329-396 non-financial companies a year, 5,494 company-years

universe            Q1 low     Q2     Q3     Q4  Q5 high   Q1-Q5     t  Q1>Q5
Roster in force      14.2%  11.1%  12.8%  12.5%    13.0%   +1.26  0.60   8/15
Today's roster       26.5%  16.3%  17.0%  16.0%    19.0%   +7.41  2.63  10/15
Median accruals, roster in force: Q1 -10.7%  Q2 -6.3%  Q3 -4.3%  Q4 -2.7%  Q5 +0.1%

Survivorship effect (today's roster minus roster in force), points a year, paired over 15 years
  Q1            +12.21  paired t  4.47  higher in 13/15 years
  Q2             +5.22  paired t  5.47  higher in 14/15 years
  Q3             +4.21  paired t  4.65  higher in 14/15 years
  Q4             +3.58  paired t  4.35  higher in 12/15 years
  Q5             +6.06  paired t  5.63  higher in 15/15 years
  Quintile average +6.26  paired t  7.18  higher in 15/15 years
  Q1-Q5 spread    +6.15  paired t  2.26  higher in 11/15 years

Roster in force: members since removed: 1,567 of 5,957 company-years (26.3%)
  quintile   names     removed  share      return, removed  return, rest
  Q1          1185         401  33.8%                 3.3%         19.9%
  Q2          1190         296  24.9%                 6.0%         12.8%
  Q3          1195         265  22.2%                 5.9%         14.9%
  Q4          1190         272  22.9%                 9.5%         13.3%
  Q5          1197         333  27.8%                 4.9%         16.1%

Today's roster: companies not yet in the index: 1,104 of 5,494 company-years (20.1%)
  quintile   names  not yet in  share   return, not yet in  return, rest
  Q1          1092         361  33.1%                40.2%         19.9%
  Q2          1100         238  21.6%                25.5%         13.6%
  Q3          1098         178  16.2%                32.1%         14.1%
  Q4          1100         141  12.8%                28.8%         14.0%
  Q5          1104         186  16.8%                36.2%         15.6%
```

## Where the removed and the later members sit

The honest test finds little: Q1 returned 14.2% a year, Q5 13.0% and Q2 11.1%, and low accruals won in 8 of 15 years. Today's roster raises every quintile, by 3.58 points (Q4) to 12.21 (Q1); the uneven lift supplies most of the spread. The bias on the spread, +6.15 points, has a paired t of 2.26.

Two groups account for the Q1 lift. On the roster in force, 33.8% of Q1 company-years belong to companies since removed, and they returned 3.3% a year against 19.9% for the rest. Many are oil and gas producers such as Noble Energy and Hess, whose depletion charges, and in the oil price slumps their impairment write-downs, push earnings below cash flow. Today's roster drops them and adds companies not yet in the index, 33.1% of its Q1, returning 40.2% a year: young loss-makers whose operating cash flow ran ahead of their earnings, such as Tesla in 2013 and 2020 or Palantir and AppLovin in 2024, each sorted before it joined.

The expected mechanism, aggressive accruers that fail and vanish, is present but smaller: removed companies make up 27.8% of Q5 and returned 4.9% against 16.1%. Removed names gather in both tails, more in Q1, so deleting them lifts the low-accrual side most.

## What an accrual screen needs before it is backtested

Test an accrual screen on the roster in force at each sort date, with statements as filed by then. On that basis the S&P 500 spread over 2011-2025 is 1.26 points a year (t 0.60), no reliable case for a large-cap accruals strategy; the naive version reports 7.41 points, enough to pass a standard significance screen.

Before trusting a factor sort, count how many names in each tail later left the index or had not yet joined it; where they cluster, a survivor list distorts most. [A price-to-book backtest on the same rosters](/blog/value-investing-backtest-survivorship-bias-python) applies the roster comparison to book value.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
