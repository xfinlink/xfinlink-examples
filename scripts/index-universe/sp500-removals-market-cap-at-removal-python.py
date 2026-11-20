# Full write-up: https://xfinlink.com/blog/sp500-removals-market-cap-at-removal-python
#
# How small are S&P 500 members when they are demoted? Every removal effective 1990-2025
# is valued at its last month-end close as a member and ranked against that day's
# point-in-time roster, rebuilt from the dated event log. Removals are classed from each
# security's own price series: still trading 63+ trading days later (demoted) or stopped
# trading within 5 (mostly acquired).
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

CHART = "sp500-removals-market-cap-at-removal-python.png"
# Second share classes -> the company's first index line; BRK-A trades but is not an index line
CLASS_OF = {10003590: 10030636, 10032892: 10031087, 10006698: 10006697, 10003102: 10003101,
            10030753: 10030752, 10004763: 10031256, 10029970: 10029932, 10006133: 10025221}
# Guideline for new S&P 500 additions (unadjusted company market cap, US$bn), S&P DJI
GUIDE = pd.Series({"2007-07-18": 5.0, "2008-09-25": 4.0, "2008-12-18": 3.0, "2009-12-09": 3.5,
                   "2011-02-16": 4.0, "2013-06-19": 4.6, "2014-07-16": 5.3, "2017-03-10": 6.1,
                   "2019-02-20": 8.2, "2020-12-08": 9.8, "2021-03-17": 11.8, "2021-06-03": 13.1,
                   "2022-03-04": 14.6, "2023-01-04": 12.7, "2023-07-05": 14.5, "2024-01-02": 15.8,
                   "2024-04-01": 18.0, "2025-01-02": 20.5, "2025-07-01": 22.7})
GUIDE.index = pd.to_datetime(GUIDE.index)

# Membership spells: the 1979-12-31 roster plus every later addition, each closed by its removal
ev = pd.concat([xfl.index_events("sp500", offset=o) for o in range(0, 4000, 1000)], ignore_index=True)
ev["eff"] = pd.to_datetime(ev["effective_date"])
ev = ev.dropna(subset=["entity_id"]).astype({"entity_id": int})
r0 = xfl.index("sp500", as_of="1979-12-31")
OPEN = pd.Timestamp("2099-01-01")
spells = [(e, pd.Timestamp(a), pd.Timestamp(r) if pd.notna(r) else OPEN)
          for e, a, r in zip(r0["entity_id"], r0["added_date"], r0["removed_date"])]
rems = ev[ev["event_type"] == "removed"]
for a in ev[(ev["event_type"] == "added") & (ev["eff"] > "1979-12-31")].itertuples():
    out = rems.loc[(rems["entity_id"] == a.entity_id) & (rems["eff"] >= a.eff), "eff"]
    spells.append((a.entity_id, a.eff, out.min() if len(out) else OPEN))
sp = pd.DataFrame(spells, columns=["eid", "add", "rem"]).astype({"eid": int})

def members(day):
    return sp[(sp["add"] <= day) & (sp["rem"] >= day)]

checks = []
for d in ["1990-06-29", "2002-07-19", "2008-12-31", "2016-06-30"]:
    api = set(xfl.index("sp500", as_of=d)["entity_id"].dropna().astype(int))
    checks.append(api == set(members(pd.Timestamp(d))["eid"]))

# Month-end market value of every security in the index at any time since 1980
ids = sorted(set(sp.loc[sp["rem"] >= "1980-01-01", "eid"]) | {10006133})
mo = pd.concat([xfl.prices(entity_id=ids[i:i + 50], start="1980-01-01", end="2025-12-31", interval="1mo",
                           fields=["close", "market_cap"], max_rows=250_000)
                for i in range(0, len(ids), 50)], ignore_index=True)
mo["m"] = mo["date"].dt.to_period("M")
cap = mo.drop_duplicates(["entity_id", "m"]).pivot(index="m", columns="entity_id", values="market_cap")
cal = xfl.prices(entity_id=10001013, start="1979-12-01", end="2026-06-30", fields=["close"], max_rows=50_000)
days = pd.DatetimeIndex(sorted(cal["date"].unique()))          # Coca-Cola, a member throughout
mends = pd.Series(days).groupby(days.to_period("M")).max()      # last trading day of each month

def company_caps(day, extra=None):
    """Market value of each member company at a month-end: share classes summed."""
    ids = set(members(day)["eid"]) | ({10006133} if 10025221 in set(members(day)["eid"]) else set())
    v = cap.loc[day.to_period("M")].reindex(sorted(ids))
    return v.groupby(lambda e: CLASS_OF.get(e, e)).sum(min_count=1).dropna()

def rank(day, e):
    co = company_caps(day)
    others = co.drop(e, errors="ignore")
    v = co.get(e, np.nan)
    return v, 100 * (others < v).mean(), others.median(), len(others)

# Each removal's own daily bars around the removal: how long did the stock keep trading?
rem = rems[rems["eff"].between("1990-01-01", "2025-12-31")]
daily = {}
for y, g in rem.groupby(rem["eff"].dt.year):
    eids = sorted(set(g["entity_id"]))
    end = min(pd.Timestamp(f"{y + 1}-06-15"), pd.Timestamp("2026-06-15")).date().isoformat()
    for i in range(0, len(eids), 50):
        p = xfl.prices(entity_id=eids[i:i + 50], start=f"{y - 1}-12-15", end=end,
                       fields=["close", "market_cap"], max_rows=250_000)
        for e, s in p.drop_duplicates(["entity_id", "date"]).groupby("entity_id"):
            daily[(y, e)] = s["date"]

rows = []
for r in rem[~rem["entity_id"].isin(CLASS_OF)].drop_duplicates(["entity_id", "eff"]).itertuples():
    after = int((daily[(r.eff.year, r.entity_id)] > r.eff).sum())   # trading days after removal
    group = "demoted" if after >= 63 else "stopped trading" if after <= 5 else "stopped within 3 months"
    T = mends[mends <= r.eff].iloc[-1]                                 # last month-end as a member
    v, pct, med, n_val = rank(T, r.entity_id)
    staying = company_caps(T)
    staying = staying[staying.index.isin({CLASS_OF.get(e, e) for e in members(T).query("rem > @r.eff")["eid"]})]
    added = sp.loc[(sp["eid"] == r.entity_id) & (sp["rem"] == r.eff), "add"].min()
    row = {"eid": r.entity_id, "ticker": r.ticker, "name": r.entity_name, "eff": r.eff, "T": T,
           "group": group, "added": added, "cap": v, "pct": pct, "med": med, "n_val": n_val,
           "min_staying": staying.min(), "guide": GUIDE[GUIDE.index <= T].iloc[-1] if T >= GUIDE.index[0] else np.nan}
    Te = mends[mends >= added].iloc[0] if added >= pd.Timestamp("1980-01-01") else None
    if Te is not None and Te < T:                                      # first month-end as a member
        row["cap_in"], row["pct_in"], row["med_in"], _ = rank(Te, r.entity_id)
    rows.append(row)
res = pd.DataFrame(rows)
res["era"] = pd.cut(res["eff"].dt.year, [1989, 1999, 2009, 2019, 2025], labels=["1990s", "2000s", "2010s", "2020-25"])
short = (res["eff"] - res["added"]).dt.days < 30                       # spin-offs held for days
rule = res["eff"] == "2002-07-19"                                      # non-US members removed together
dem = res[(res["group"] == "demoted") & ~short & ~rule & res["cap"].notna()].copy()

now_ids = sorted(set(xfl.index("sp500")["entity_id"].dropna().astype(int)) | {10006133})
now = pd.concat([xfl.prices(entity_id=now_ids[i:i + 50], start="2026-11-12", end="2026-11-19", fields=["market_cap"])
                 for i in range(0, len(now_ids), 50)])
now = now[now["date"] == now["date"].max()].set_index("entity_id")["market_cap"]
now = now.groupby(lambda e: CLASS_OF.get(e, e)).sum(min_count=1).dropna()

# ---- output ------------------------------------------------------------------
bn = lambda x: x / 1e9
g = res["group"].value_counts()
print(f"S&P 500 removals effective 1990-2025: {len(res)} (second share classes folded into the company)")
print(f"  rebuilt rosters equal index(as_of) on {sum(checks)} of {len(checks)} test dates")
print(f"  stopped trading within 5 trading days: {g['stopped trading']}; still trading 63+ days later: "
      f"{g['demoted']}; in between: {g['stopped within 3 months']}")
d0 = res["group"] == "demoted"
print(f"  still trading, set aside: member under 30 days {(d0 & short).sum()}, "
      f"non-US removals of 2002-07-19 {(d0 & rule).sum()}, no market value at the last month-end "
      f"{(d0 & ~short & ~rule & res['cap'].isna()).sum()}")
print(f"  demoted members valued: {len(dem)}; other members valued at each of their month-ends: "
      f"{dem['n_val'].min()}-{dem['n_val'].max()}")

print("\nDemoted members at their last month-end in the index (company market value)")
print("era      n   median exit  median member  exit pctile  in bottom 2%  bottom 10%  a stayer was smaller")
for era, t in dem.groupby("era", observed=True):
    print(f"{era:<7} {len(t):>3}   ${bn(t['cap'].median()):>6.2f}bn   ${bn(t['med'].median()):>8.1f}bn"
          f"   {t['pct'].median():>10.1f}   {(t['pct'] < 2).mean():>11.0%}  {(t['pct'] < 10).mean():>10.0%}"
          f"  {(t['cap'] > t['min_staying']).mean():>19.0%}")
acq = res[(res["group"] == "stopped trading") & res["cap"].notna() & ~short]
print("\nFor contrast, stopped trading at removal (mostly acquired): median exit pctile by era")
print("  " + "  ".join(f"{era} {t['pct'].median():.1f} (n={len(t)})" for era, t in acq.groupby("era", observed=True)))

e = dem.dropna(subset=["cap_in"]).copy()
e["shrink"] = e["cap"] / e["cap_in"]
e["rel"] = (e["cap"] / e["med"]) / (e["cap_in"] / e["med_in"])
print(f"\nDemoted members that joined in 1980 or later: {len(e)}")
print(f"  median pctile at entry {e['pct_in'].median():.1f}, at exit {e['pct'].median():.1f}; "
      f"entered in the bottom 10% {(e['pct_in'] < 10).mean():.0%}")
print(f"  exit value / entry value: median {e['shrink'].median():.2f}; below entry value {(e['shrink'] < 1).mean():.0%}")
print(f"  size relative to the median member, exit / entry: median {e['rel'].median():.2f}")
print(f"  median years in the index {((e['eff'] - e['added']).dt.days / 365.25).median():.1f}")

gd = dem.dropna(subset=["guide"]).copy()
gd["x"] = bn(gd["cap"]) / gd["guide"]
print(f"\nDemoted at a month-end on or after 2007-07-18 (guideline for new additions in force): {len(gd)}")
for lab, t in [("2007-14", gd[gd["T"].dt.year <= 2014]), ("2015-19", gd[gd["T"].dt.year.between(2015, 2019)]),
               ("2020-25", gd[gd["T"].dt.year >= 2020]), ("all", gd)]:
    print(f"  {lab:<8} n={len(t):>3}  below the guideline {(t['x'] < 1).mean():>4.0%}  "
          f"median exit / guideline {t['x'].median():.2f}")

late = dem[dem["era"] == "2020-25"]
print(f"\nToday's roster, {len(now)} companies valued on 2026-11-18: smallest ${bn(now.min()):.1f}bn, "
      f"10th smallest ${bn(now.nsmallest(10).iloc[-1]):.1f}bn, median ${bn(now.median()):.1f}bn")
print(f"  2020-25 demoted exits below today's smallest member: {(late['cap'] < now.min()).sum()} of {len(late)}")
back = dem["eid"].map(lambda x: CLASS_OF.get(x, x)).isin(now.index)
print(f"  demoted members 1990-2025 that are in today's index: {back.sum()} of {len(dem)}")

up = dem[dem["pct"] >= 50].sort_values("eff")
print(f"\nDemoted from the upper half of the index: {len(up)}")
for _, t in up.iterrows():
    print(f"  {t['eff']:%Y-%m-%d}  {t['ticker']:<5} {t['name'][:30]:<30} ${bn(t['cap']):>5.1f}bn  pctile {t['pct']:>4.1f}")
print("Smallest at exit, 2020-25:")
for _, t in late.nsmallest(5, "cap").iterrows():
    print(f"  {t['eff']:%Y-%m-%d}  {t['ticker']:<5} {t['name'][:30]:<30} ${bn(t['cap']):>5.2f}bn  pctile {t['pct']:>4.1f}")

# ---- chart: exit value against the guideline (top), rank inside the index (bottom) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
bg, ink, blue, amber, grey = "#0a0a0a", "#e0e0e0", "#3b82f6", "#d97706", "#a3a3a3"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
fig.patch.set_facecolor(bg)
for ax in (a1, a2):
    ax.set_facecolor(bg)
    ax.tick_params(colors=grey, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
yr = lambda d: d.dt.year + (d.dt.day_of_year - 1) / 365.25
med = dem.groupby(dem["eff"].dt.year)["med"].median()
a1.plot(med.index + 0.5, bn(med), color=grey, lw=1.5, label="Median member of the index")
steps = pd.concat([GUIDE, pd.Series({pd.Timestamp("2025-12-31"): GUIDE.iloc[-1]})])
a1.step(yr(pd.Series(steps.index)), steps.values, where="post", color=amber, lw=2,
        label="Guideline for new additions (S&P DJI, from July 2007)")
a1.scatter(yr(dem["eff"]), bn(dem["cap"]), s=12, color=blue, alpha=0.75,
           label=f"Demoted member at its last month-end in the index (n={len(dem)})")
a1.set_yscale("log")
a1.minorticks_off()
a1.set_yticks([0.01, 0.1, 1, 10, 100])
a1.set_yticklabels(["$0.01bn", "$0.1bn", "$1bn", "$10bn", "$100bn"])
a1.set_ylabel("Market value (log scale)", color=ink)
a1.legend(frameon=False, labelcolor=ink, fontsize=9, loc="lower right")
a1.set_title("S&P 500 removals: how small demoted members are at exit, 1990-2025", color=ink, fontsize=12)
for frame, colour, label in [(acq, amber, "Stopped trading at removal (mostly acquired)"),
                             (dem, blue, "Still trading after removal (demoted)")]:
    y = frame.groupby(frame["eff"].dt.year)["pct"].median()
    a2.plot(y.index + 0.5, y.values, color=colour, lw=2, marker="o", ms=3, label=label)
a2.set_ylim(0, 100)
a2.set_ylabel("Median rank in the index\n(percentile, 0 = smallest)", color=ink)
a2.set_xlabel("Year of removal", color=ink)
a2.legend(frameon=False, labelcolor=ink, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor=bg)
