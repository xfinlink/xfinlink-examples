# Full write-up: https://xfinlink.com/blog/lookahead-bias-vs-survivorship-bias-earnings-yield-python
#
# Look-ahead bias against survivorship bias in one earnings-yield (E/P) backtest. At each
# December close from 2010 to 2023, S&P 500 companies are sorted into quintiles by annual
# net income over market value (all share classes) on that close; loss-makers form their own
# group. Each portfolio is held the next calendar year, equal weight, bought and held, with
# monthly total returns; a holding that stops trading takes its recorded delisting return
# (an exit with none keeps its last traded price), then sits in cash. Returns end on 2024-12-31. The sort runs four ways, a 2x2:
#   roster:  the S&P 500 roster in force on the sort date, or today's roster traced back
#   timing:  point-in-time, the latest fiscal year filed by the sort date, or look-ahead,
#            the latest fiscal year that had ended by 31 December, filed or not
# A fiscal year carries the same net income figure in both timings: the figure on file at
# the first year-end snapshot (as_of) that contains it.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = pd.Timestamp("2026-11-26")  # the day today's roster was read
forms = pd.date_range("2010-12-01", "2023-12-31", freq="BYE")  # last business day of December
snaps = list(forms) + [pd.Timestamp("2024-12-31")]             # statement snapshots, one a year
ymd = lambda d: d.strftime("%Y-%m-%d")

def mcap(d, tickers=None, issuers=None):
    """Market value (all share classes, millions USD) on the close of d."""
    kw = dict(period_type="ttm", fields=["market_cap"], as_of=ymd(d))
    if tickers is not None:  # start/end only date the tickers' resolution; as_of fixes the snapshot
        m = xfl.metrics(tickers, start=ymd(d - pd.Timedelta(days=300)), end=ymd(d), **kw)
    else:
        m = pd.concat([xfl.metrics(entity_id=issuers[i:i + 500], **kw) for i in range(0, len(issuers), 500)])
    return m[["entity_id", "ticker", "entity_name", "priced_on", "market_cap"]]

# Today's roster: security ids for prices, issuer ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
today_iss = mcap(RUN_DATE, tickers=today["ticker"].tolist())
sec_of = dict(zip(today_iss["entity_id"], today_iss["ticker"].map(dict(zip(today["ticker"], today["entity_id"])))))

rows = []
for d in forms:
    roster = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = mcap(d, tickers=roster["ticker"].tolist()).assign(pit=True)
    a["sec"] = a["ticker"].map(dict(zip(roster["ticker"], roster["entity_id"])))
    b = mcap(d, issuers=sorted(set(sec_of) - set(a["entity_id"]))).assign(pit=False)  # today's names outside it
    b["sec"] = b["entity_id"].map(sec_of)
    rows.append(pd.concat([a, b]).assign(form=d, roster=len(roster)))
S = pd.concat(rows, ignore_index=True).dropna(subset=["sec", "market_cap"])
S = S[(S["market_cap"] > 0) & ((S["form"] - pd.to_datetime(S["priced_on"])).dt.days.between(0, 7))]
S = S.drop_duplicates(["form", "entity_id"])
S["sec"] = S["sec"].astype(int)
S["today"] = S["entity_id"].isin(set(sec_of))

# Annual net income on file at each year-end snapshot (two years of fiscal years)
F = []
for k, d in enumerate(snaps):
    ids = sorted(set(S.loc[S["form"].isin(snaps[max(k - 1, 0):k + 1]), "entity_id"]))
    for i in range(0, len(ids), 500):
        f = xfl.fundamentals(entity_id=ids[i:i + 500], period_type="annual", as_of=ymd(d),
                             start=ymd(d - pd.DateOffset(years=2)), end=ymd(d),
                             fields=["net_income", "revenue"], max_rows=50_000)
        F.append(f.assign(snap=d))
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
F = F[F["net_income"].notna() & ~((F["off"] >= 2) & (F["near"] < 300))]
# One figure per fiscal year: the one in the first snapshot that holds it
FY = F.sort_values("snap").drop_duplicates(["entity_id", "period_end"])[
    ["entity_id", "period_end", "snap", "net_income", "filing_date"]]

def pick(rule):
    """Each company's fiscal year under a timing rule, ended within 18 months of the sort."""
    x = S[["form", "entity_id"]].merge(FY, on="entity_id")
    x = x[(x["form"] - x["period_end"]).dt.days <= 548]
    x = x[x["snap"] <= x["form"]] if rule == "pit" else \
        x[x["period_end"] <= pd.to_datetime(x["form"].dt.year.astype(str) + "-12-31")]
    return x.sort_values("period_end").drop_duplicates(["form", "entity_id"], keep="last").set_index(["form", "entity_id"])

for rule in ("pit", "la"):
    p = pick(rule)
    S = S.join(p[["period_end", "net_income", "filing_date"]].add_suffix("_" + rule), on=["form", "entity_id"])
same = (S["period_end_la"] - S["period_end_pit"]).dt.days.between(1, 299)  # two dates, one fiscal year
S.loc[same, ["period_end_la", "net_income_la", "filing_date_la"]] = \
    S.loc[same, ["period_end_pit", "net_income_pit", "filing_date_pit"]].values

# Monthly total returns, December 2010 to December 2024, by security
ids = sorted(S["sec"].unique())
px = pd.concat(xfl.prices(entity_id=ids[i:i + 200], start="2010-12-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily", "close"], max_rows=250_000) for i in range(0, len(ids), 200))
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2010-12", "2024-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(index=R.index, columns=ids).notna()

# Exits: recorded delisting return added to the last month (one exit year per call)
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

S["y"] = S["form"].dt.year
S = S[[traded.at[pd.Period(f"{y}-12", "M"), s] for y, s in zip(S["y"], S["sec"])]]
S = S.dropna(subset=["net_income_pit", "net_income_la"])  # one sample for both timings
# Companies sharing a company record with a merger partner before the merger, set aside for those
# years: MeadWestvaco (WestRock's record) to 2014; Tyco and Johnson Controls Inc to 2015
SET_ASIDE = {8097: 2014, 4074: 2015}
SET_ASIDE_YEARS = {670: (2012, 2013)}  # AGL Resources 2012-2013: an annual record dated ahead of its fiscal year
S = S[~(S["y"] <= S["entity_id"].map(SET_ASIDE))]
S = S[[not (e in SET_ASIDE_YEARS and SET_ASIDE_YEARS[e][0] <= y <= SET_ASIDE_YEARS[e][1])
       for e, y in zip(S["entity_id"], S["y"])]].copy()

def held(secs, y, m0=1, m1=12):
    """Equal weight at the start of month m0 of y+1, bought and held to m1; cash after an exit."""
    return (1 + R.loc[f"{y + 1}-{m0:02d}":f"{y + 1}-{m1:02d}", secs].fillna(0)).prod() - 1

for y, g in S.groupby("y"):
    secs = sorted(set(g["sec"]))
    for col, (m0, m1) in {"r": (1, 12), "r_jm": (1, 3), "r_ad": (4, 12)}.items():
        S.loc[g.index, col] = g["sec"].map(held(secs, y, m0, m1)).values

def quintiles(u, t):
    """0 = loss-makers (net income <= 0); 1 = lowest positive E/P ... 5 = highest (cheapest)."""
    ep = u["net_income_" + t] / u["market_cap"]
    q = pd.Series(0, index=u.index)
    pos = ep > 0
    q[pos] = ep[pos].groupby(u.loc[pos, "y"]).transform(lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)
    return q

ROSTERS = {"Roster in force": S["pit"], "Today's roster": S["today"]}
TIMING = {"point-in-time": "pit", "look-ahead": "la"}
cells, spread, Q = {}, {}, {}
for kr, mask in ROSTERS.items():
    for kt, t in TIMING.items():
        u = S[mask].assign(q=quintiles(S[mask], t))
        Q[kr, kt] = u
        tab = u.groupby(["y", "q"])["r"].mean().unstack()
        cells[kr, kt] = tab.assign(n=u.groupby("y").size())
        spread[kr, kt] = tab[5] - tab[1]
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
cagr = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
P, T = ROSTERS
N = len(forms)

# ---------------- report ----------------
print(f"{N} December sorts {forms[0].date()} to {forms[-1].date()}, holding years {forms[0].year + 1}-2024; "
      f"today's roster {len(today)} lines on {RUN_DATE.date()}; {len(ids):,} securities, {len(delist)} exit records")
n = S[S["pit"]].groupby("y").size()
print(f"Roster companies with both E/P figures: {n.min()}-{n.max()} a year; today's roster traced back: "
      f"{S[S['today']].groupby('y').size().min()}-{S[S['today']].groupby('y').size().max()}")
p = S[S["pit"]]
newer = p["period_end_la"] > p["period_end_pit"]  # always a full year later (rule above)
dec = p["period_end_la"].dt.month == 12
lag = (p.loc[newer, "filing_date_la"] - p.loc[newer, "form"]).dt.days
print(f"Look-ahead reads a newer fiscal year for {newer.mean():.1%} of roster company-years (December year-ends "
      f"{newer[dec].mean():.1%}, other year-ends {newer[~dec].mean():.1%}); that report was filed a median "
      f"{lag.median():.0f} days after the sort ({(lag <= 90).mean():.1%} within 90 days)")

print(f"\n{'roster, timing':32}{'names':>6}{'loss':>7}{'Q1 dear':>9}{'Q5 cheap':>10}{'Q5-Q1':>8}{'t':>6}{'Q5>Q1':>7}"
      f"{'Q1 cagr':>9}{'Q5 cagr':>9}")
for (kr, kt), tab in cells.items():
    s = spread[kr, kt]
    print(f"{kr + ', ' + kt:32}{tab['n'].mean():6.0f}{tab[0].mean():7.1%}{tab[1].mean():9.1%}{tab[5].mean():10.1%}"
          f"{s.mean() * 100:+8.2f}{tstat(s):6.2f}{(s > 0).sum():>4}/{N}{cagr(tab[1]):9.1%}{cagr(tab[5]):9.1%}")

print(f"\nBias in the Q5-Q1 spread, points a year, paired over {N} years")
for lab, x in [("Look-ahead, roster in force", spread[P, "look-ahead"] - spread[P, "point-in-time"]),
               ("Look-ahead, today's roster", spread[T, "look-ahead"] - spread[T, "point-in-time"]),
               ("Survivorship, point-in-time", spread[T, "point-in-time"] - spread[P, "point-in-time"]),
               ("Survivorship, look-ahead", spread[T, "look-ahead"] - spread[P, "look-ahead"]),
               ("Both (naive minus honest)", spread[T, "look-ahead"] - spread[P, "point-in-time"])]:
    print(f"  {lab:30}{x.mean() * 100:+6.2f}  paired t {tstat(x):5.2f}  larger in {(x > 0).sum():>2}/{N} years")
print("Bias in the average quintile's return, points a year")
for lab, a, b in [("Look-ahead, roster in force", (P, "look-ahead"), (P, "point-in-time")),
                  ("Survivorship, point-in-time", (T, "point-in-time"), (P, "point-in-time")),
                  ("Both (naive minus honest)", (T, "look-ahead"), (P, "point-in-time"))]:
    x = cells[a][[1, 2, 3, 4, 5]].mean(axis=1) - cells[b][[1, 2, 3, 4, 5]].mean(axis=1)
    print(f"  {lab:30}{x.mean() * 100:+6.2f}  paired t {tstat(x):5.2f}  larger in {(x > 0).sum():>2}/{N} years")

# Mechanism: who changes quintile when the newly ended fiscal year is read early (roster in force)
a, b = Q[P, "point-in-time"], Q[P, "look-ahead"]
m = a[["y", "q", "r_jm", "r_ad", "net_income_pit", "net_income_la"]].join(b["q"].rename("q_la"))
m["move"] = np.select([m["q_la"] > m["q"], m["q_la"] < m["q"]], ["cheaper under look-ahead", "dearer under look-ahead"],
                      "same quintile")
print(f"\nRoster in force: {(m['move'] != 'same quintile').mean():.1%} of company-years change quintile under look-ahead")
for k, g in m.groupby("move"):
    chg = (g["net_income_la"] - g["net_income_pit"]) / g["net_income_pit"].abs()
    print(f"  {k:26}{len(g):6} company-years  median change in net income {chg.median():+7.1%}  "
          f"Jan-Mar return {g['r_jm'].mean():6.2%}  Apr-Dec {g['r_ad'].mean():6.2%}")
for kt in TIMING:
    u = Q[P, kt]
    for col, lab in [("r_jm", "January-March"), ("r_ad", "April-December")]:
        t5 = u.groupby(["y", "q"])[col].mean().unstack()
        x = t5[5] - t5[1]
        print(f"  {kt:14} Q5-Q1, {lab:14}{x.mean() * 100:+6.2f} points (t {tstat(x):5.2f})")

# Chart: Q1 and Q5 compound returns and the Q5-Q1 spread in the four cells
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4)
labs = [f"{kr}\n{kt}" for kr, kt in cells]
for off, (lab, vals, color) in zip([-0.27, 0, 0.27], [
        ("Q1, lowest E/P (compound a year)", [cagr(t[1]) * 100 for t in cells.values()], "#c2780a"),
        ("Q5, highest E/P (compound a year)", [cagr(t[5]) * 100 for t in cells.values()], "#3b82f6"),
        ("Q5 minus Q1 (mean, points a year)", [s.mean() * 100 for s in spread.values()], "#0d9488")]):
    bars = ax.bar(x + off, vals, width=0.25, color=color, label=lab)
    ax.bar_label(bars, fmt="%.1f", padding=2, color="#e0e0e0", fontsize=8.5)
ax.axhline(0, color="#3f3f3f", linewidth=1)
ax.axvline(1.5, color="#3f3f3f", linewidth=1, linestyle=":")
ax.set_xticks(x, labs)
ax.set_ylabel(f"Return, {forms[0].year + 1}-2024 (%)")
ax.set_title("Look-Ahead Bias vs Survivorship Bias in an E/P Backtest (S&P 500)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("lookahead-bias-vs-survivorship-bias-earnings-yield-python.png", dpi=150, facecolor="#0a0a0a")
