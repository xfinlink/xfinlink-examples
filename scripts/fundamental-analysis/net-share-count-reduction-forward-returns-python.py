# Full write-up: https://xfinlink.com/blog/net-share-count-reduction-forward-returns-python
"""Do companies that actually shrink their share count outperform?

Builds a net share-count-reduction signal from annual shares outstanding,
sorts a point-in-time S&P 500 roster on it at seven annual rebalances, and
measures forward 12-month returns against the equal-weighted universe.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

CHART = "net-share-count-reduction-forward-returns-python.png"
YEARS = list(range(2019, 2026))          # rebalance years, end of June
CHUNK = 100                              # entity ids per request
SPLIT_TAIL = pd.Timedelta(days=183)      # split window runs 6 months past the fiscal year end
GUARD = np.log(1.20)                     # only de-split when it moves the change 20%+ toward zero
MAX_CHANGE = 0.60                        # beyond this a count change is a merger, not a repurchase


def chunked(seq, n=CHUNK):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def anchor(start, end, ids):
    """Daily closes for one narrow calendar window, for every id."""
    return pd.concat([xfl.prices(entity_id=c, start=start, end=end,
                                 fields=["close", "adj_close"], max_rows=200000)
                      for c in chunked(ids)], ignore_index=True)


# ---------------------------------------------------------------- 1. rosters
rosters = {}
for y in YEARS + [2026]:
    r = xfl.index("sp500", as_of=f"{y}-06-30").dropna(subset=["entity_id"])
    r["entity_id"] = r["entity_id"].astype(int)
    rosters[y] = r.set_index("entity_id")["ticker"]
ids = sorted(set().union(*[set(r.index) for r in rosters.values()]))
print(f"point-in-time S&P 500 rosters {YEARS[0]}-2026: {len(ids)} distinct companies\n")

# ---------------------------------------------------------------- 2. share counts
fu = pd.concat([xfl.fundamentals(entity_id=c, period_type="annual",
                                 start="2016-06-01", end="2025-06-30",
                                 fields=["shares_outstanding", "dividends_per_share"])
                for c in chunked(ids)], ignore_index=True)
fu = fu[fu["shares_outstanding"].notna() & (fu["shares_outstanding"] > 0)]
fu["entity_id"] = fu["entity_id"].astype(int)
pe = fu["period_end"]
fu["fy"] = np.where(pe.dt.month == 1, pe.dt.year - 1, pe.dt.year)
fu = fu[((pe.dt.month == 12) & (pe.dt.day >= 15)) | ((pe.dt.month == 1) & (pe.dt.day <= 15))]
fu = fu.drop_duplicates(subset=["entity_id", "fy"], keep="last")
print(f"annual filings with a December fiscal year end: {len(fu)} rows, "
      f"{fu['entity_id'].nunique()} companies")

# ---------------------------------------------------------------- 3. prices
ids = sorted(fu["entity_id"].unique().tolist())
px = {y: anchor(f"{y}-06-18", f"{y}-07-06", ids) for y in YEARS + [2026]}
px[2017] = anchor("2017-12-18", "2018-01-10", ids)

anch = []
for y in YEARS + [2026]:
    t = pd.Timestamp(f"{y}-06-30")
    d = px[y][px[y]["date"].between(t - pd.Timedelta(days=7), t)]
    d = d.sort_values("date").groupby("entity_id", as_index=False).last()
    d["year"] = y
    anch.append(d[["entity_id", "year", "ticker", "date", "close", "adj_close"]])
anch = pd.concat(anch, ignore_index=True)
anch["entity_id"] = anch["entity_id"].astype(int)

# ---------------------------------------------------------------- 4. split events
# adj_close is split-adjusted and close is not, so close/adj_close is the cumulative
# future split factor. Where it moves between two anchors, pull the exact events.
grid = pd.concat([px[2017].assign(g=2017)] + [px[y].assign(g=y) for y in YEARS + [2026]])
grid["entity_id"] = grid["entity_id"].astype(int)
grid["F"] = grid["close"] / grid["adj_close"]
grid = grid.sort_values("date").groupby(["entity_id", "g"], as_index=False).last()
grid = grid.sort_values(["entity_id", "date"])
grid["F_next"] = grid.groupby("entity_id")["F"].shift(-1)
grid["d_next"] = grid.groupby("entity_id")["date"].shift(-1)
flag = grid[grid["F_next"].notna() & (grid["F"] / grid["F_next"] - 1).abs().gt(0.002)]

ev = {}
for _, r in flag.iterrows():
    p = xfl.prices(entity_id=int(r["entity_id"]), start=str(r["date"].date()),
                   end=str(r["d_next"].date()), fields=["split_ratio"])
    if len(p) == 0:
        continue
    p = p[p["split_ratio"].notna() & (p["split_ratio"] != 1)]
    ev.setdefault(int(r["entity_id"]), []).extend(zip(p["date"], p["split_ratio"].astype(float)))
print(f"share-count-changing events found in the price series: "
      f"{sum(len(v) for v in ev.values())} across {len(ev)} companies")


def split_prod(eid, d1, d2):
    return float(np.prod([r for d, r in ev.get(eid, []) if d1 < d <= d2] or [1.0]))


# ---------------------------------------------------------------- 5. signal
sh = fu.set_index(["entity_id", "fy"])[["shares_outstanding", "period_end", "dividends_per_share"]]
rows, declined, applied = [], [], []
for y in YEARS:
    for eid, tk in rosters[y].items():
        if (eid, y - 2) not in sh.index or (eid, y - 1) not in sh.index:
            continue
        s1, pe1, _ = sh.loc[(eid, y - 2)]
        s2, pe2, dps = sh.loc[(eid, y - 1)]
        if min(s1, s2) < 1.0:                        # not a count on the stated scale
            continue
        raw = s2 / s1
        P = split_prod(eid, pe1, pe2 + SPLIT_TAIL)
        cand = raw / P
        use_split = abs(np.log(cand)) < abs(np.log(raw)) - GUARD
        if P != 1.0:
            (applied if use_split else declined).append((y, tk, s1, s2, raw, P, use_split))
        rows.append((y, eid, tk, s1, s2, pe2, raw, P, cand if use_split else raw, dps))

sig = pd.DataFrame(rows, columns=["year", "entity_id", "ticker", "sh_prev", "sh_curr",
                                  "pe", "raw", "split", "ratio", "dps"])
sig["reduction"] = 1.0 - sig["ratio"]

print(f"\nSPLIT AUDIT  ({len(applied) + len(declined)} company-years had a split near the fiscal "
      f"year: {len(applied)} de-splitted, {len(declined)} left alone because the two counts were "
      f"already on one scale)")
print(f"{'yr':>5} {'ticker':<7} {'prior (M)':>12} {'latest (M)':>12} {'raw':>8} "
      f"{'split':>8} {'signal used':>12}")
for y, tk, s1, s2, raw, P, used in sorted(applied)[:5] + sorted(declined)[:7]:
    print(f"{y:>5} {tk:<7} {s1:>12,.1f} {s2:>12,.1f} {raw:>8.3f} {P:>8.3f} "
          f"{(1 - raw / P) * 100 if used else (1 - raw) * 100:>11.2f}%")
print(f"entity-years moving more than 25% before the split adjustment: "
      f"{(sig['raw'] - 1).abs().gt(0.25).sum()}; after: {(sig['ratio'] - 1).abs().gt(0.25).sum()}")

sig = sig[sig["reduction"].abs() <= MAX_CHANGE]

# ---------------------------------------------------------------- 6. forward returns
start = anch.rename(columns={"adj_close": "px0", "close": "cl0", "date": "d0", "ticker": "tk0"})
end = anch.rename(columns={"adj_close": "px1", "ticker": "tk1"})[["entity_id", "year", "px1", "tk1"]]
end["year"] -= 1
sig = sig.merge(start[["entity_id", "year", "d0", "tk0", "px0", "cl0"]], on=["entity_id", "year"], how="left")
sig = sig.merge(end, on=["entity_id", "year"], how="left")
sig = sig[sig["px0"].notna()]

# a name that stops trading inside the holding year exits at its last clean daily bar
gone = sig[sig["px1"].isna()]
for _, m in gone.iterrows():
    p = xfl.prices(entity_id=int(m["entity_id"]), start=str(m["d0"].date()),
                   end=f"{m['year'] + 1}-06-30", fields=["adj_close"]).dropna(subset=["adj_close"])
    step = np.log(p["adj_close"]).diff().abs()
    p = p[(step < np.log(2)) | step.isna()]
    if len(p):
        sig.loc[m.name, ["px1", "tk1"]] = [p["adj_close"].iloc[-1], p["ticker"].iloc[-1]]
print(f"\nnames that stopped trading inside a holding year: {len(gone)} (exited at last traded price)")

# Symbol integrity. Both price legs must carry either the company's live symbol or one the
# resolution layer shows it has retired, and a retired symbol may not appear after the live one.
live = {e: t for y in YEARS + [2026] for e, t in rosters[y].items()}
retired = {}
for t in sorted({t for e, t in zip(anch["entity_id"], anch["ticker"]) if t != live.get(e)}):
    try:
        for e in xfl.resolve(t, include=[])["data"][t]["entities"]:
            if e["ticker_valid_to"]:
                retired.setdefault(e["entity_id"], set()).add(t)
    except Exception:
        continue


seen_old = anch[[t in retired.get(e, set()) for e, t in zip(anch["entity_id"], anch["ticker"])]]
seen_live = anch[anch["ticker"] == anch["entity_id"].map(live)]
last_old = seen_old.groupby("entity_id")["date"].max()
first_live = seen_live.groupby("entity_id")["date"].min()
cut = {e: d for e, d in last_old.items() if e in first_live.index and d > first_live[e]}


def usable(e, t0, t1, d0):
    old = retired.get(e, set())
    valid = {live.get(e)} | old
    return (t0 in valid and t1 in valid
            and not (t0 == live.get(e) and t1 in old)
            and not (e in cut and d0 < cut[e]))


ok = np.array([usable(e, t0, t1, d0) for e, t0, t1, d0
               in zip(sig["entity_id"], sig["tk0"], sig["tk1"], sig["d0"])])
print(f"observations dropped by the symbol-integrity check on the two price legs: "
      f"{len(sig) - int(ok.sum())}")
sig = sig[ok & sig["px1"].notna()].copy()
sig["fwd"] = sig["px1"] / sig["px0"] - 1
sig["divy"] = sig["dps"].fillna(0) / [split_prod(e, p, d) for e, p, d
                                      in zip(sig["entity_id"], sig["pe"], sig["d0"])] / sig["cl0"]
sig["q"] = sig.groupby("year")["reduction"].transform(lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5]))

print(f"\nfinal panel: {len(sig)} company-years, "
      f"{sig.groupby('year').size().min()}-{sig.groupby('year').size().max()} names per rebalance, "
      f"missing values {int(sig[['reduction', 'fwd', 'q', 'divy']].isna().sum().sum())}")

# ---------------------------------------------------------------- 7. results
tab = sig.groupby("q").agg(n=("fwd", "size"), sig=("reduction", "mean"), ret=("fwd", "mean"),
                           med=("fwd", "median"), dy=("divy", "mean"))
base = sig["fwd"].mean()
print("\nQUINTILES ON NET SHARE-COUNT REDUCTION, pooled over 7 annual rebalances")
print(f"{'bucket':<24} {'n':>5} {'mean signal':>12} {'mean fwd 1y':>12} {'median':>9} {'div yield':>10}")
names = {1: "Q1 most dilution", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 most reduction"}
for q, r in tab.iterrows():
    print(f"{names[q]:<24} {int(r['n']):>5} {r['sig'] * 100:>11.2f}% {r['ret'] * 100:>11.2f}% "
          f"{r['med'] * 100:>8.2f}% {r['dy'] * 100:>9.2f}%")
print(f"{'equal-weighted universe':<24} {len(sig):>5} {sig['reduction'].mean() * 100:>11.2f}% "
      f"{base * 100:>11.2f}% {sig['fwd'].median() * 100:>8.2f}% {sig['divy'].mean() * 100:>9.2f}%")

per = sig.pivot_table(index="year", columns="q", values="fwd", aggfunc="mean")
per["base"] = sig.groupby("year")["fwd"].mean()
print("\nEVERY REBALANCE SEPARATELY (forward 12-month price return, equal weighted)")
print(f"{'buy 30 June':>12} {'Q1':>9} {'Q5':>9} {'universe':>9} {'Q5-Q1':>9} {'Q5-universe':>12}")
for y, r in per.iterrows():
    print(f"{y:>12} {r[1] * 100:>8.2f}% {r[5] * 100:>8.2f}% {r['base'] * 100:>8.2f}% "
          f"{(r[5] - r[1]) * 100:>8.2f}% {(r[5] - r['base']) * 100:>11.2f}%")
sp = (per[5] - per[1]) * 100
print(f"Q5 beat Q1 in {int((sp > 0).sum())} of {len(sp)} rebalances; spread mean {sp.mean():.2f}%, "
      f"standard deviation {sp.std():.2f}%, range {sp.min():.2f}% to {sp.max():.2f}%")

print("\nLARGEST SIGNAL VALUES, checked against the underlying counts")
ext = pd.concat([sig.nlargest(4, "reduction"), sig.nsmallest(4, "reduction")])
print(f"{'yr':>5} {'ticker':<7} {'prior (M)':>12} {'latest (M)':>12} {'split':>7} "
      f"{'reduction':>10} {'fwd 1y':>9}")
for r in ext.itertuples():
    print(f"{r.year:>5} {r.ticker:<7} {r.sh_prev:>12,.1f} {r.sh_curr:>12,.1f} {r.split:>7.2f} "
          f"{r.reduction * 100:>9.2f}% {r.fwd * 100:>8.2f}%")

# ---------------------------------------------------------------- 8. chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.bar([names[q] for q in tab.index], tab["ret"] * 100, color="#3b82f6", width=0.6)
ax1.axhline(base * 100, color="#e0e0e0", lw=1, ls="--")
ax1.text(-0.42, base * 100 + 0.35, f"equal-weighted universe {base * 100:.1f}%",
         color="#e0e0e0", ha="left", fontsize=9)
ax1.set_ylabel("Average forward 12-month return (%)")
ax1.set_title("Sorting the S&P 500 on net share-count reduction, 2019-2025", color="#e0e0e0")
ax2.bar(sp.index.astype(str), sp.values, color=["#3b82f6" if v > 0 else "#64748b" for v in sp])
ax2.axhline(0, color="#e0e0e0", lw=1)
ax2.set_ylabel("Shrinkers minus diluters (%)")
ax2.set_xlabel("Portfolio bought 30 June of")
for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {CHART}")
