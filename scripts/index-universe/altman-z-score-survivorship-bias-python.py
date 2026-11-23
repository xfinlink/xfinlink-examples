# Full write-up: https://xfinlink.com/blog/altman-z-score-survivorship-bias-python
#
# Does survivorship bias flatter distressed stocks in an Altman Z-score backtest? At each
# December close from 1999 to 2023, S&P 500 members get the Altman Z-score (original 1968
# public-manufacturer weights) of the latest annual report filed by that date. Financial
# companies (SIC 6000-6999) are left out. Members are put in Altman's zones (distress below
# 1.81, safe above 2.99) and held for the next calendar year, equal weight, bought and held.
# The sort runs on the roster in force each December, on that roster's members still in the
# index today, and on today's roster projected back. A holding that stops trading takes its
# recorded delisting return, then sits in cash. Returns end on 2024-12-31. The whole test is
# run twice: on all non-financial members, and with utilities, pipelines, power producers and waste services
# (SIC 4900-4999) set aside.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = "2026-11-23"  # the day today's roster was read
forms = pd.date_range("1999-12-01", "2023-12-31", freq="BYE")  # last business day of December
ymd = lambda d: d.strftime("%Y-%m-%d")
chunks = lambda xs: [xs[i:i + 50] for i in range(0, len(xs), 50)]

def z_rows(d, tickers=None, issuers=None):
    """Annual Altman Z rows from statements filed on or before d (two years of fiscal years)."""
    kw = dict(period_type="annual", fields=["altman_z_score"], as_of=ymd(d),
              start=ymd(d - pd.DateOffset(years=2)), end=ymd(d))
    out = [xfl.metrics(c, **kw) for c in chunks(tickers)] if tickers is not None else \
          [xfl.metrics(entity_id=c, **kw) for c in chunks(issuers)]
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# Today's roster: security ids for prices, issuer ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
tm = pd.concat([xfl.metrics(c, period_type="annual", fields=["altman_z_score"], start="2024-06-01")
                for c in chunks(today["ticker"].tolist())], ignore_index=True)
tsec = dict(zip(today["ticker"], today["entity_id"].astype(int)))
today_sec = tm.drop_duplicates("entity_id").set_index("entity_id")["ticker"].map(tsec).dropna().astype(int).to_dict()

# Each December: Z for the roster in force, and for today's companies outside it
rows = []
for d in forms:
    ro = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = z_rows(d, tickers=ro["ticker"].tolist())
    a["sec"] = a["ticker"].map(dict(zip(ro["ticker"], ro["entity_id"].astype(int))))
    a["pit"] = True
    extra = sorted(set(today_sec) - set(a["entity_id"]))
    b = z_rows(d, issuers=extra)
    if len(b):
        b["sec"] = b["entity_id"].map(today_sec)
        b["pit"] = False
    rows.append(pd.concat([a, b]).assign(form=d, roster=len(ro)))
Z = pd.concat(rows, ignore_index=True)

# Latest annual report filed by the sort date with all five Z inputs, fiscal year ended within 18 months
Z["altman_z_score"] = pd.to_numeric(Z["altman_z_score"])
Z = Z.dropna(subset=["altman_z_score", "sec"])
Z = Z[(Z["form"] - Z["period_end"]).dt.days <= 548]
Z = Z.sort_values("period_end").drop_duplicates(["form", "entity_id"], keep="last")
Z["sec"] = Z["sec"].astype(int)
Z["today"] = Z["entity_id"].isin(set(today_sec))

# Industry code (current SIC), read once per company on its first sort date; financials leave the sample
sic = {}
for d, g in Z.drop_duplicates("entity_id").groupby("form"):
    for c in chunks(sorted(g["entity_id"])):
        f = xfl.fundamentals(entity_id=c, period_type="annual", fields=["sic"], as_of=ymd(d),
                             start=ymd(d - pd.DateOffset(years=2)), end=ymd(d))
        sic.update(f.dropna(subset=["sic"]).drop_duplicates("entity_id").set_index("entity_id")["sic"].to_dict())
Z["sic"] = pd.to_numeric(Z["entity_id"].map(sic), errors="coerce")
# Classified by their business on the sort date where it differs from today's code (REIT conversions, successor
# records): Kmart/Sears Holdings, Wyndham Worldwide, American Tower, Crown Castle, Iron Mountain, Equinix, SBA, Potlatch
NOT_FINANCIAL_THROUGH = {83480: 2011, 32228: 2017, 28515: 2011, 28947: 2013, 26005: 2013, 30242: 2014,
                         29347: 2015, 5950: 2005}
fin = Z["sic"].between(6000, 6999) & ~(Z["form"].dt.year <= Z["entity_id"].map(NOT_FINANCIAL_THROUGH))
sample = (f"{len(Z):,} company-years with a Z-score; financial (SIC 6000-6999) {fin.sum():,} set aside, "
          f"no SIC on record {Z['sic'].isna().sum()} kept")
Z = Z[~fin]

# Monthly total returns, December 1999 to December 2024, by security
ids = sorted(Z["sec"].unique())
px = pd.concat([xfl.prices(entity_id=c, start="1999-12-01", end="2024-12-31", interval="1mo",
                           fields=["return_daily", "close"], max_rows=100_000) for c in chunks(ids)])
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("1999-12", "2024-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(columns=ids).notna()

# Exits: recorded delisting return added to the last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    for c in chunks(g.index.tolist()):
        d = xfl.prices(entity_id=c, start=f"{y}-01-01", end="2024-12-31", interval="1mo", fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

Z["y"] = Z["form"].dt.year
Z = Z[[traded.at[pd.Period(f"{y}-12", "M"), s] if pd.Period(f"{y}-12", "M") in traded.index else False
       for y, s in zip(Z["y"], Z["sec"])]]
Z["zone"] = np.select([Z["altman_z_score"] < 1.81, Z["altman_z_score"] > 2.99], ["distress", "safe"], "grey")

# Three universes each December, held the next calendar year
def held(secs, y):
    """Equal weight in January of y+1, bought and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12", sorted(secs)]
    return (1 + r.fillna(0)).prod() - 1

ZONES = ["distress", "grey", "safe"]
years = range(1999, 2024)
util = Z["sic"].between(4900, 4999)
U = {"Point-in-time rosters": Z["pit"], "Members then, still in today's index": Z["pit"] & Z["today"],
     "Today's roster": Z["today"]}
held_r = {}
def backtest(keep):
    res = {}
    for k, mask in U.items():
        u, out = Z[mask & keep], []
        for y in years:
            g = u[u["y"] == y]
            r = g["sec"].map(held(set(g["sec"]), y))
            held_r[k, y] = r
            q = np.ceil(g["altman_z_score"].rank(pct=True) * 5)          # 1 = lowest Z fifth
            out.append({"year": y + 1, "n": len(g), "nd": (g["zone"] == "distress").sum(),
                        **{z: r[g["zone"] == z].mean() for z in ZONES}, "Q1": r[q == 1].mean(), "Q5": r[q == 5].mean()})
        res[k] = pd.DataFrame(out).set_index("year")
    return res
cagr = lambda s: (1 + s).prod() ** (1 / len(s)) - 1
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
P, M, T = U

print(f"{len(forms)} December rosters 1999-2023, today's roster {len(today)} lines ({RUN_DATE}), "
      f"{len(ids):,} securities, {len(delist)} exit records; holding years 2000-2024\n{sample}")
out = {}
for label, keep in [("All non-financial members", pd.Series(True, index=Z.index)),
                    ("Utilities, pipelines, power and waste (SIC 4900-4999) set aside", ~util)]:
    res = out[label] = backtest(keep)
    ds = {k: df["distress"] - df["safe"] for k, df in res.items()}
    print(f"\n{label}\n{'universe':38}{'names':>6}{'in D':>6}{'distress':>10}{'grey':>8}{'safe':>8}"
          f"{'D-S':>7}{'t':>6}{'D ahead':>9}")
    for k, df in res.items():
        print(f"{k:38}{df['n'].mean():6.0f}{df['nd'].mean():6.0f}"
              + "".join(f"{cagr(df[z]):{10 if z == 'distress' else 8}.2%}" for z in ZONES)
              + f"{ds[k].mean() * 100:+7.2f}{tstat(ds[k]):6.2f}{(ds[k] > 0).sum():>6}/25")
    for x, lab in [(res[T]["distress"] - res[P]["distress"], "today's roster minus point-in-time, distress zone"),
                   (res[M]["distress"] - res[P]["distress"], "  of which members since removed"),
                   (res[T]["distress"] - res[M]["distress"], "  of which names not yet in the index"),
                   (res[T]["safe"] - res[P]["safe"], "today's roster minus point-in-time, safe zone"),
                   (ds[T] - ds[P], "today's roster minus point-in-time, D-S")]:
        print(f"{lab:52}{x.mean() * 100:+6.2f} points a year, paired t {tstat(x):5.2f}, ahead {(x > 0).sum()}/25")
    print("Lowest-Z fifth minus highest-Z fifth: " + ", ".join(
        f"{k.split(',')[0]} {(res[k]['Q1'] - res[k]['Q5']).mean() * 100:+.2f} (t {tstat(res[k]['Q1'] - res[k]['Q5']):.2f})"
        for k in U))
backtest(pd.Series(True, index=Z.index))  # leaves held_r on the full sample

# Who sat in each zone of the real rosters, and what happened to them
p = Z[U[P]].copy()
p["r"] = [held_r[P, y][i] for i, y in zip(p.index, p["y"])]
p["stopped"] = [s in ended.index and ended[s] <= pd.Period(f"{y + 1}-12", "M") for s, y in zip(p["sec"], p["y"])]
p["failed"] = [st and delist.get(s, {}).get("event_type") == "dropped" for s, st in zip(p["sec"], p["stopped"])]
print(f"\nPoint-in-time zones, {len(p):,} company-years")
for z in ZONES:
    g = p[p["zone"] == z]
    print(f"  {z:9}{len(g):5} company-years, SIC 4900-4999 {g['sic'].between(4900, 4999).mean():5.1%}, not in today's index "
          f"{1 - g['today'].mean():5.1%}, stopped trading in the year held {g['stopped'].mean():5.2%} "
          f"(dropped from exchange {g['failed'].sum()})")
for z in ["distress", "safe"]:
    g = p[p["zone"] == z]
    fate = np.where(g["today"], "still in today's index", np.where(g["sec"].isin(ended.index), "stopped trading later",
                                                                    "left the index, still trading"))
    print(f"  {z} zone, mean return in the year held: " + ", ".join(
        f"{f} {g['r'][fate == f].mean():.1%} ({(fate == f).sum()})" for f in
        ["still in today's index", "left the index, still trading", "stopped trading later"]))
print("Dropped from their exchange during the year held: " + "; ".join(
    f"{r.ticker} Z {r.altman_z_score:.2f} ({r.zone}, Dec {r.y}) {r.r:+.0%}" for r in p[p["failed"]].itertuples()))
e = p[(p["ticker"] == "ENE") & (p["y"] == 2000)].iloc[0]
print(f"Enron, December 2000: Z {e.altman_z_score:.2f} from the {e.period_end.year} annual report ({e.zone} zone), "
      f"{e.r:+.0%} in 2001")
t = Z[U[T] & ~util & (Z["zone"] == "distress")].copy()
t["r"] = [held_r[T, y][i] for i, y in zip(t.index, t["y"])]
print("Today's roster, distress zone without SIC 4900-4999, mean return in the year held: " + ", ".join(
    f"{lab} {t.loc[t['pit'] == f, 'r'].mean():.1%} ({(t['pit'] == f).sum()} company-years)"
    for f, lab in [(True, "members at the sort"), (False, "not yet members")]))
print("  largest: " + ", ".join(f"{r.ticker} (Z {r.altman_z_score:.2f}, Dec {r.y}) {r.r:+.0%} in {r.y + 1}"
                                for r in t.nlargest(4, "r").itertuples()))

# Chart: annual return by zone, and distress minus safe, three universes
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=True)
x = np.array([0, 1, 2, 3.6])
for ax, (label, res) in zip(axes, out.items()):
    for off, (k, color) in zip([-0.27, 0, 0.27], [(P, "#3b82f6"), (M, "#0d9488"), (T, "#c2780a")]):
        vals = [cagr(res[k][z]) * 100 for z in ZONES] + [(res[k]["distress"] - res[k]["safe"]).mean() * 100]
        bars = ax.bar(x + off, vals, width=0.25, color=color, label=k)
        ax.bar_label(bars, fmt="%.1f", padding=2, color="#e0e0e0", fontsize=8)
    ax.axhline(0, color="#3f3f3f", linewidth=1)
    ax.axvline(2.8, color="#3f3f3f", linewidth=1, linestyle=":")
    ax.set_xticks(x, ["Distress zone\nZ below 1.81", "Grey zone", "Safe zone\nZ above 2.99", "Distress minus\nsafe (points)"])
    ax.set_ylabel("Annual return (%)")
    ax.set_title(label, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(-3, 25.5)
axes[0].legend(frameon=False, loc="upper right", fontsize=9, ncol=3)
fig.suptitle("Altman Z-Score Backtest: Survivorship Bias in the S&P 500, 2000-2024")
plt.tight_layout()
plt.savefig("altman-z-score-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
