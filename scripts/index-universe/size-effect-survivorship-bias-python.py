# Full write-up: https://xfinlink.com/blog/size-effect-survivorship-bias-python
#
# The size effect inside the S&P 500, 2000-2024: does a backtest on today's roster
# make small members look better than they were? Each December from 1999 to 2023,
# members are sorted into five groups by company market value that day and held
# equal weight, bought and held, for the next calendar year. Three universes: the
# roster in force that December (point-in-time), the members of that roster still
# in the index today, and today's roster valued on the same date. A holding that
# stops trading takes its recorded delisting return, then sits in cash.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Second share classes count toward the company's first index line; BRK-A trades but is not an index line
CLASS_OF = {10003590: 10030636, 10032892: 10031087, 10006698: 10006697, 10003102: 10003101,
            10030753: 10030752, 10004763: 10031256, 10029970: 10029932, 10006133: 10025221}
co = lambda e: CLASS_OF.get(e, e)

# S&P 500 rosters on the last business day of each December 1999-2023, and today's roster
forms = pd.date_range("1999-12-01", "2023-12-31", freq="BYE")
member = {d.year: set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))["entity_id"].dropna().astype(int))
          for d in forms}
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))
ids = sorted(set().union(*member.values()) | today | {10006133})

# Monthly total returns and month-end market values, December 1999 to December 2024
px = pd.concat(xfl.prices(entity_id=ids[i:i + 50], start="1999-11-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily", "market_cap"], max_rows=250_000)
               for i in range(0, len(ids), 50)).reset_index(drop=True)
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
cap = px.pivot(index="m", columns="entity_id", values="market_cap")
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2000-01", "2024-12", freq="M"))

# Each exit's recorded delisting return, added to its last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    d = xfl.prices(entity_id=g.index.tolist(), start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                   fields=["return_daily"])
    delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def sort(names, y):
    """Company market value on the December sort date (share classes summed), split into quintiles."""
    v = cap.loc[pd.Period(f"{y}-12", "M")].reindex(sorted(names | {10006133} if 10025221 in names else names))
    v = v.groupby(co).sum(min_count=1).dropna()
    v = v[v.index.isin(names)]
    return v, np.ceil(v.rank(pct=True) * 5).astype(int)           # 1 = smallest fifth

def held(names, y):
    """Equal weight in January of y+1, bought and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12", sorted(names)]
    return ((1 + r.fillna(0)).prod() - 1)

books = {"Point-in-time rosters": lambda y: member[y],
         "Members then, still in today's index": lambda y: member[y] & today,
         "Today's roster": lambda y: today}
years = range(1999, 2024)
res, info = {}, {}
for name, f in books.items():
    rows = []
    for y in years:
        v, q = sort(f(y), y)
        r = held(set(q.index), y)
        rows.append({"year": y + 1, "n": len(q), **{k: r[q.index[q == k]].mean() for k in range(1, 6)}})
        info[name, y] = (v, q, r)
    res[name] = pd.DataFrame(rows).set_index("year")
cagr = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5

print(f"{len(member)} December rosters 1999-2023, today's roster {len(today)} lines, {len(ids):,} securities, "
      f"{len(delist)} exit records; holding years 2000-2024")
print(f"\n{'universe':38}{'names':>6}" + "".join(f"{'Q' + str(k):>8}" for k in range(1, 6))
      + f"{'Q1-Q5':>8}{'t':>6}{'Q1 ahead':>10}")
smb = {}
for name, df in res.items():
    smb[name] = df[1] - df[5]
    print(f"{name:38}{df['n'].mean():6.0f}" + "".join(f"{cagr(df[k]):8.2%}" for k in range(1, 6))
          + f"{smb[name].mean() * 100:+8.2f}{tstat(smb[name]):6.2f}{(smb[name] > 0).sum():>7}/25")
P, M, T = books
for a, b, label in [(T, P, "today's roster minus point-in-time (total bias)"),
                    (M, P, "  of which members since removed"),
                    (T, M, "  of which names not yet in the index")]:
    x = smb[a] - smb[b]
    print(f"{label:50}{x.mean() * 100:+6.2f} points a year in Q1-Q5, paired t {tstat(x):5.2f}, "
          f"ahead {(x > 0).sum()}/25")
for a, b in [(2000, 2012), (2013, 2024)]:
    print(f"  {a}-{b} Q1-Q5 a year: " + ", ".join(f"{n.split(',')[0]} {smb[n].loc[a:b].mean() * 100:+.2f}"
                                                  for n in books))

# Who sits in the smallest fifth of each universe
print("\nSmallest fifth (Q1), averaged over the 25 sort dates")
for name in books:
    med, outside, later, rq1 = [], [], [], []
    for y in years:
        v, q, r = info[name, y]
        q1 = q.index[q == 1]
        med.append(v[q1].median() / 1e9)
        outside.append(np.mean([e not in member[y] for e in q1]))
        later.append(np.mean([e not in today for e in q1]))
    print(f"  {name:38} median value ${np.mean(med):5.1f}bn, not an S&P 500 member that December "
          f"{np.mean(outside):6.1%}, not in today's index {np.mean(later):6.1%}")
fate = {"still in today's index": 0, "left the index, still trading": 0}
ret_by = {k: [] for k in fate}
for y in years:
    v, q, r = info[P, y]
    for e in q.index[q == 1]:
        k = "still in today's index" if e in today else (
            f"stopped trading ({delist.get(e, {}).get('event_type', 'no exit record')})" if e in ended.index
            else "left the index, still trading")
        fate[k] = fate.get(k, 0) + 1
        ret_by.setdefault(k, []).append(r[e])
n_q1 = sum(fate.values())
print(f"\nPoint-in-time Q1 holdings, 25 years: {n_q1} company-years, by status at end-2024")
for k, c in sorted(fate.items(), key=lambda x: -x[1]):
    print(f"  {k:34}{c:5} ({c / n_q1:5.1%}), mean return that year {np.mean(ret_by[k]):7.2%}")
v, q, r = info[T, 1999]
top = r[q.index[q == 1]].sort_values(ascending=False).head(5)
tick = px[px["m"] == pd.Period("1999-12", "M")].set_index("entity_id")["ticker"]
print("Today's roster, Q1 of December 1999: " + ", ".join(
    f"{tick[e]} ${v[e] / 1e9:.1f}bn {r[e]:+.0%}" for e in top.index) + " in 2000")

# Chart: annual return by size quintile in the three universes
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 6)
for off, (name, color) in zip([-0.27, 0, 0.27], [(P, "#3b82f6"), (M, "#a3a3a3"), (T, "#f59e0b")]):
    bars = ax.bar(x + off, [cagr(res[name][k]) * 100 for k in x], width=0.27, color=color, label=name)
    ax.bar_label(bars, fmt="%.1f", padding=2, color="#e0e0e0", fontsize=8)
ax.set_xticks(x, ["Q1\nsmallest", "Q2", "Q3", "Q4", "Q5\nlargest"])
ax.set_xlabel("Size quintile on each December sort date")
ax.set_ylabel("Annual return, 2000-2024 (%)")
ax.set_title("Size Effect Backtest: Survivorship Bias in the S&P 500 (2000-2024)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("size-effect-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
