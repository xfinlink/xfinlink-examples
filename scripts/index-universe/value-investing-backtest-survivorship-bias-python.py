# Full write-up: https://xfinlink.com/blog/value-investing-backtest-survivorship-bias-python
#
# Does survivorship bias inflate a price-to-book value backtest? At each December close
# from 1999 to 2023, S&P 500 members are sorted into quintiles by book-to-market: book
# equity from the latest balance sheet filed by that date, over market value on that
# close. Cheap minus expensive is held for the next calendar year, equal-weighted and
# rebalanced monthly. The sort runs once on the roster in force each December
# (point-in-time), once on the point-in-time members still in the index today, and once
# on today's roster, every company in it that traded then. Negative book equity is its
# own group. A holding that stops trading takes its recorded delisting return in its
# exit month; an exit with no recorded delisting return keeps its last traded price.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = pd.Timestamp("2026-11-16")  # the day today's roster was read
FIELDS = ["market_cap", "pb_ratio"]
forms = pd.date_range("1999-12-01", "2023-12-31", freq="BYE")  # last business day of December
ymd = lambda d: d.strftime("%Y-%m-%d")

def snapshot(d, tickers=None, issuers=None):
    """Market value (all share classes) on the close of d and P/B from statements filed by d."""
    kw = dict(period_type="ttm", fields=FIELDS, as_of=ymd(d))
    if tickers is not None:  # start/end only date the tickers' resolution; as_of fixes the snapshot
        m = xfl.metrics(tickers, start=ymd(d - pd.Timedelta(days=300)), end=ymd(d), **kw)
    else:
        m = pd.concat([xfl.metrics(entity_id=issuers[i:i + 500], **kw) for i in range(0, len(issuers), 500)])
    return m[["entity_id", "ticker", "entity_name", "priced_on", "period_end"] + FIELDS]

# Today's roster: security ids for prices, issuer ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
today_iss = snapshot(RUN_DATE, tickers=today["ticker"].tolist())
sec_of = dict(zip(today_iss["entity_id"], today_iss["ticker"].map(dict(zip(today["ticker"], today["entity_id"])))))

snaps = []
for d in forms:
    roster = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = snapshot(d, tickers=roster["ticker"].tolist()).assign(pit=True)
    a["sec"] = a["ticker"].map(dict(zip(roster["ticker"], roster["entity_id"])))
    extra = sorted(set(sec_of) - set(a["entity_id"]))  # today's companies outside that roster
    b = snapshot(d, issuers=extra).assign(pit=False)
    b["sec"] = b["entity_id"].map(sec_of)
    snaps.append(pd.concat([a, b]).assign(form=d, roster_size=len(roster)))
S = pd.concat(snaps, ignore_index=True).dropna(subset=["sec", "market_cap"])
S["sec"] = S["sec"].astype(int)
S["today"] = S["entity_id"].isin(set(sec_of))

# Book equity: P/B where the snapshot has it, else the latest equity filed by the date
eq = []
for d, g in S.groupby("form"):
    f = xfl.fundamentals(entity_id=sorted(g["entity_id"].unique()), as_of=ymd(d),
                         start=ymd(d - pd.DateOffset(years=2)), fields=["total_equity"], max_rows=50_000)
    f = f.dropna(subset=["total_equity"]).sort_values("period_end").drop_duplicates("entity_id", keep="last")
    eq.append(f[["entity_id", "total_equity"]].assign(form=d))
S = S.merge(pd.concat(eq), on=["form", "entity_id"], how="left")
S["book"] = (S["market_cap"] / S["pb_ratio"]).fillna(S["total_equity"])
S = S.dropna(subset=["book"])
S["bm"] = S["book"] / S["market_cap"]
S["year"] = S["form"].dt.year + 1

# Monthly total returns 2000-2024 by security
ids = sorted(S["sec"].unique())
px = pd.concat(xfl.prices(entity_id=ids[i:i + 200], start="2000-01-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily"], max_rows=250_000) for i in range(0, len(ids), 200))
px["m"] = px["date"].dt.to_period("M")
R = px.drop_duplicates(["entity_id", "m"]).pivot(index="m", columns="entity_id", values="return_daily")
R = R.reindex(index=pd.period_range("2000-01", "2024-12", freq="M"), columns=ids)

# Exits: recorded delisting return in the exit month (single-page calls, one exit year each)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    g = g.index.tolist()
    for i in range(0, len(g), 300):
        d = xfl.prices(entity_id=g[i:i + 300], start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def universe(mask):
    u = S[mask].drop_duplicates(["form", "entity_id"]).copy()
    pos = u["book"] > 0
    u["q"] = 0  # 0 = negative book equity, its own group
    u.loc[pos, "q"] = u[pos].groupby("form")["bm"].transform(
        lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)  # 1 = most expensive, 5 = cheapest
    return u

def year_return(g, y, weights=None):  # equal weight (or formation market value), rebalanced monthly
    r = R.loc[f"{y}-01":f"{y}-12", g["sec"].tolist()]
    w = pd.Series(1.0 if weights is None else g[weights].values, index=r.columns)
    return (1 + ((r * w).sum(axis=1) / r.notna().mul(w).sum(axis=1)).fillna(0)).prod() - 1

U = {"Point-in-time rosters": universe(S["pit"]),
     "Point-in-time, still members today": universe(S["pit"] & S["today"]),
     "Today's roster": universe(S["today"])}
res = {k: pd.DataFrame([{"year": y, "q": q, "n": len(g), "ew": year_return(g, y),
                         "vw": year_return(g, y, "market_cap")}
                        for (y, q), g in u.groupby(["year", "q"])]).pivot(index="year", columns="q")
       for k, u in U.items()}
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
spread = {k: t["ew"][5] - t["ew"][1] for k, t in res.items()}
print(f"{len(forms)} December sorts, {forms[0].date()} to {forms[-1].date()}; holding years 2000-2024; "
      f"today's roster: {len(today)} members on {RUN_DATE.date()}; {len(ids)} securities, "
      f"{len(delist)} exit records")

# ---------------- report (formatting) ----------------
P, T = "Point-in-time rosters", "Today's roster"
print("\nMean annual return by book-to-market quintile, equal weight, 2000-2024")
print(f"{'universe':36}{'names':>6}{'neg BE':>7}{'Q1 exp':>8}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5 cheap':>9}"
      f"{'Q5-Q1':>7}{'t':>6}{'Q5>Q1':>7}")
for k, t in res.items():
    s = spread[k]
    print(f"{k:36}{t['n'].sum(axis=1).mean():6.0f}{t['ew'][0].mean():7.1%}"
          + "".join(f"{t['ew'][q].mean():{8 if q == 1 else 9 if q == 5 else 7}.1%}" for q in range(1, 6))
          + f"{s.mean() * 100:+7.2f}{tstat(s):6.2f}{int((s > 0).sum()):>4}/{len(s)}")
d = spread[T] - spread[P]
print(f"\nToday's roster minus point-in-time, value spread: {d.mean() * 100:+.2f} points a year, paired t {tstat(d):.2f}; "
      f"larger in {int((d > 0).sum())} of {len(d)} years")
lvl = res[T]["ew"][[1, 2, 3, 4, 5]].mean(axis=1) - res[P]["ew"][[1, 2, 3, 4, 5]].mean(axis=1)
print(f"Today's roster minus point-in-time, average of the five quintiles: {lvl.mean() * 100:+.2f} points a year")
for k in (P, T):
    s = spread[k]
    print(f"{k}: Q5-Q1 2000-2012 {s.loc[:2012].mean() * 100:+.2f}, 2013-2024 {s.loc[2013:].mean() * 100:+.2f}; "
          f"weighted by market value {(res[k]['vw'][5] - res[k]['vw'][1]).mean() * 100:+.2f} "
          f"(t {tstat(res[k]['vw'][5] - res[k]['vw'][1]):.2f})")

print("\nPoint-in-time quintiles: who left, and what they earned (mean annual return)")
u = U[P]
u["stopped"] = [e in ended.index and ended[e] <= pd.Period(f"{y}-12", "M") for e, y in zip(u["sec"], u["year"])]
u["failed"] = [s and delist.get(e, {}).get("event_type") == "dropped" for e, s in zip(u["sec"], u["stopped"])]
for q, name in [(1, "Q1 expensive"), (5, "Q5 cheap"), (0, "negative BE")]:
    g = u[u["q"] == q]
    stay = pd.Series([year_return(h, y) for y, h in g[g["today"]].groupby("year")]).mean()
    gone = pd.Series([year_return(h, y) for y, h in g[~g["today"]].groupby("year")]).mean()
    print(f"  {name:13} median P/B {g['pb_ratio'].median():5.2f}  not in today's index {1 - g['today'].mean():5.1%}  "
          f"stopped trading within the year {g['stopped'].mean():4.1%} (dropped {g['failed'].mean():4.1%})  "
          f"still members {stay:6.1%}  since removed {gone:6.1%}")

print("\nKnown collapses in the point-in-time sort (P/B at the December before, quintile, return that year)")
for tk, y in [("ENE", 2001), ("BSC", 2008), ("LEH", 2008), ("WM", 2008), ("SIVB", 2023), ("FRC", 2023)]:
    g = u[(u["ticker"] == tk) & (u["year"] == y)]
    for _, r in g.iterrows():
        print(f"  {tk:5}{y}  P/B {r['market_cap'] / r['book']:5.2f}  Q{r['q']}  {year_return(g, y):+.1%}")
t = U[T]
print("\nToday's roster: share of each quintile not yet an index member at the sort, "
      + ", ".join(f"Q{q} {1 - t.loc[t['q'] == q, 'pit'].mean():.1%}" for q in range(6)))
print("Names a year by group: " + "; ".join(f"{k}: neg BE {res[k]['n'][0].mean():.0f}" for k in res))

# Chart: mean annual return by quintile and the cheap-minus-expensive spread, three universes
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.array([0, 1, 2, 3, 4, 5.6])
for off, (k, color) in zip([-0.27, 0, 0.27], [(P, "#3b82f6"), ("Point-in-time, still members today", "#0d9488"),
                                               (T, "#c2780a")]):
    vals = [res[k]["ew"][q].mean() * 100 for q in range(1, 6)] + [spread[k].mean() * 100]
    bars = ax.bar(x + off, vals, width=0.25, color=color, label=k)
    ax.bar_label(bars, fmt="%.1f", padding=2, color="#e0e0e0", fontsize=7.5)
ax.axhline(0, color="#3f3f3f", linewidth=1)
ax.axvline(5.0, color="#3f3f3f", linewidth=1, linestyle=":")
ax.set_xticks(x, ["Q1\nexpensive", "Q2", "Q3", "Q4", "Q5\ncheap", "Cheap minus\nexpensive"])
ax.set_xlabel("Book-to-market quintile at each December sort")
ax.set_ylabel("Mean annual return, 2000-2024 (%)")
ax.set_title("Value Investing Backtest: Survivorship Bias in the S&P 500")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(top=24.5)
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig("value-investing-backtest-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
