# Full write-up: https://xfinlink.com/blog/dow-jones-additions-deletions-performance-python
#
# Dow Jones additions and deletions since 1999: did the stocks added to the Dow do better than
# the stocks they replaced over the next one, three and five years?

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LAST_DAY = "2024-12-31"                                   # every window must end by this date
HORIZONS = {"1y": 252, "3y": 756, "5y": 1260}             # trading days
# Removal dates where the same company carried on under a new line, so nothing was swapped:
# J.P. Morgan into JPMorgan Chase (2000), DuPont into DowDuPont (2017), DowDuPont into Dow Inc (2019)
SUCCESSIONS = {"2000-12-29", "2017-08-31", "2019-04-01"}

ev = xfl.index_events("djia", start="1999-01-01")         # dated additions and removals, 1999-10-29 on
ev["entity_id"] = ev["entity_id"].astype(int)
rem, add = ev[ev["event_type"] == "removed"], ev[ev["event_type"] == "added"]

changes = []
for d, out in rem.groupby("effective_date"):
    if d in SUCCESSIONS or d > LAST_DAY:
        continue
    first_in = add.loc[add["effective_date"] > d, "effective_date"].min()  # additions take effect the next day
    inn = add[add["effective_date"] == first_in]
    changes.append({"date": d, "out": list(zip(out["entity_id"], out["ticker"])),
                    "in": list(zip(inn["entity_id"], inn["ticker"]))})

ids = sorted(set(ev["entity_id"]))
px = xfl.prices(entity_id=ids, start="1999-10-01", end=LAST_DAY, fields=["close", "return_daily"], max_rows=400_000)
R = px.pivot(index="date", columns="entity_id", values="return_daily")
last_bar = px.groupby("entity_id")["date"].max()
dia = xfl.prices("DIA", start="1999-10-01", end=LAST_DAY, fields=["return_daily"]).set_index("date")["return_daily"]
cal = dia.index

# Securities whose shares stopped trading before the end: the exit record gives the delisting return
delist = {}
for e in last_bar[last_bar < pd.Timestamp(LAST_DAY)].index:
    p = xfl.prices(entity_id=int(e), start=f"{last_bar[e].year}-01-01", end=LAST_DAY, fields=["return_daily"])
    for k, v in (p.attrs["meta"].get("delisting") or {}).items():
        delist[int(k)] = v


GM_OLD = 10001629  # listing ended 2009-06-01; common stock cancelled 2011-03-31 (Motors Liquidation Co.)


def leg(e, days):
    """Total return over days. After the last close the delisting return applies, then the proceeds follow DIA."""
    if e == GM_OLD:                                       # old GM: a total loss from the change date
        return -1.0
    r = R[e].reindex(days).fillna(0.0)
    if last_bar[e] < days[-1]:
        after = days > last_bar[e]
        r[after] = dia.reindex(days)[after]
        dlr = (delist.get(e) or {}).get("delisting_return")
        if days[0] <= last_bar[e]:                         # exit inside the window
            r[days[after][0]] = (1 + r[days[after][0]]) * (1 + (dlr or 0.0)) - 1
    return np.prod(1 + r) - 1


def run():
  rows, stock_rows = [], []
  for c in changes:
      k0 = cal.get_loc(pd.Timestamp(c["date"]))             # base: the close on the leaver's last day
      for h, n in HORIZONS.items():
          if k0 + n >= len(cal):                            # window would run past LAST_DAY: dropped
              continue
          days = cal[k0 + 1:k0 + n + 1]
          out_r = {t: leg(e, days) for e, t in c["out"]}
          in_r = {t: leg(e, days) for e, t in c["in"]}
          rows.append({"date": c["date"], "h": h, "removed": np.mean(list(out_r.values())),
                       "added": np.mean(list(in_r.values())), "dia": np.prod(1 + dia.reindex(days)) - 1,
                       "out": "/".join(out_r), "in": "/".join(in_r),
                       "survivors": np.mean([v for (e, t), v in zip(c["out"], out_r.values())
                                             if last_bar[e] >= pd.Timestamp(LAST_DAY)] or [np.nan])})
          stock_rows += [{"h": h, "date": c["date"], "ticker": t, "ret": v, "vs_added": v - rows[-1]["added"],
                          "stopped": last_bar[e] < days[-1]} for (e, t), v in zip(c["out"], out_r.values())]
  res, stk = pd.DataFrame(rows), pd.DataFrame(stock_rows)
  res["gap"] = res["added"] - res["removed"]                # positive: the additions did better
  return res, stk


res, stk = run()

print(f"Dow changes since 1999-10-29: {len(changes)} swaps ({len(SUCCESSIONS)} corporate successions set aside)\n")
print("Added minus removed total return, per change (percentage points)")
print(res.pivot(index=["date", "out", "in"], columns="h", values="gap")[list(HORIZONS)].map(lambda x: "" if pd.isna(x) else f"{x:+.0%}").to_string())
print("\nTotal return per change (equal weight within each side): mean / median, and how often each side won")
for h in HORIZONS:
    x, s = res[res["h"] == h], stk[stk["h"] == h]
    w = stats.wilcoxon(x["added"], x["removed"])
    print(f"{h}: n={len(x):>2}  removed {x['removed'].mean():+6.1%} / {x['removed'].median():+6.1%}  "
          f"added {x['added'].mean():+6.1%} / {x['added'].median():+6.1%}  DIA {x['dia'].mean():+6.1%}  "
          f"median gap {x['gap'].median():+6.1%}  removed side won {(x['gap'] < 0).sum()}/{len(x)}  "
          f"removed stocks ahead {(s['vs_added'] > 0).sum()}/{len(s)}  Wilcoxon p={w.pvalue:.2f}")
print("\nRemoved side counting only names still trading at end-2024")
for h in HORIZONS:
    x = res[(res["h"] == h)].dropna(subset=["survivors"])
    print(f"{h}: n={len(x):>2}  removed (survivors) {x['survivors'].mean():+6.1%}  added {x['added'].mean():+6.1%}  "
          f"removed side won {(x['survivors'] > x['added']).sum()}/{len(x)}")
print("\nRemoved stocks whose shares stopped trading inside a window (5-year return; DIA after the exit)")
ids_of = {(c["date"], t): e for c in changes for e, t in c["out"]}
for r in stk[(stk["h"] == "5y") & stk["stopped"] & (stk["ticker"] != "GM")].itertuples():
    e = ids_of[(r.date, r.ticker)]
    d = delist.get(e, {})
    print(f"  {r.date} {r.ticker:<3} last close {last_bar[e].date()}  exit {d.get('event_type')}, "
          f"delisting return {d.get('delisting_return', float('nan')):+.1%}  5y {r.ret:+.1%}")

# ---- chart: added minus removed, per change and horizon ----
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444"})
p = res.pivot(index="date", columns="h", values="gap")
names = res.drop_duplicates("date").set_index("date")
labels = [f"{d[:7]}\nin: {names.loc[d, 'in']}\nout: {names.loc[d, 'out']}" for d in p.index]
x = np.arange(len(p))
fig, ax = plt.subplots(figsize=(10, 5))
for i, (h, col) in enumerate(zip(HORIZONS, ["#bfdbfe", "#60a5fa", "#2563eb"])):
    ax.bar(x + (i - 1) * 0.27, p[h] * 100, 0.25, color=col, label=f"{h[0]} year{'s' if h != '1y' else ''} after")
ax.axhline(0, color="#888888", lw=0.8)
ax.set_xticks(x, labels, fontsize=6)
ax.set_ylabel("Added minus removed, total return (points)")
ax.set_title("Dow Jones Additions and Deletions: Added Minus Removed, by Change")
ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("dow-jones-additions-deletions-performance-python.png", dpi=150)
