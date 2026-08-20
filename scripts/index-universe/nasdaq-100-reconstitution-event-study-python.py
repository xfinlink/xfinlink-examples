# Full write-up: https://xfinlink.com/blog/nasdaq-100-reconstitution-event-study-python
"""What happens to a stock around the day it joins or leaves the Nasdaq-100?

Event study on the membership change log, abnormal returns measured against QQQ.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

PRE, POST = 20, 60

# 1. Every Nasdaq-100 membership change with a date attached.
events = xfl.index_events("ndx100", start="2014-01-01", end="2026-08-19")
events["effective_date"] = pd.to_datetime(events["effective_date"])
events["year"] = events["effective_date"].dt.year

# 2. Daily prices around each change, addressed by entity id rather than by
#    ticker so a symbol later reassigned to another company cannot be picked up.
frames = []
for year, group in events.groupby("year"):
    ids = sorted(group["entity_id"].dropna().unique().tolist())
    for i in range(0, len(ids), 60):
        frames.append(xfl.prices(entity_id=ids[i : i + 60],
                                 start=f"{year - 1}-09-01", end=f"{year + 1}-06-30",
                                 fields=["adj_close"], max_rows=300000))
px = pd.concat(frames, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])

bench = xfl.prices(["QQQ"], start="2014-01-01", end="2026-08-19",
                   fields=["adj_close"], max_rows=300000)
bench["date"] = pd.to_datetime(bench["date"])
bench = bench.set_index("date")["adj_close"].sort_index()
gaps = bench.index.to_series().diff().dt.days.max()
print(f"benchmark QQQ: {len(bench)} sessions, {bench.index.min().date()} to "
      f"{bench.index.max().date()}, largest gap {int(gaps)} days")

bench_ret = bench.pct_change()
sessions = bench.index

# 3. Line each event up on a common event-time axis and subtract the benchmark.
paths = {"added": [], "removed": []}
dropped = {"off session": 0, "window off the edge": 0, "series incomplete": 0}
for row in events.itertuples():
    stock = px[px["entity_id"] == row.entity_id].set_index("date")["adj_close"]
    stock = stock[~stock.index.duplicated(keep="last")].sort_index()
    if row.effective_date not in sessions:
        dropped["off session"] += 1
        continue
    day0 = sessions.get_loc(row.effective_date)
    if day0 - PRE < 0 or day0 + POST >= len(sessions):
        dropped["window off the edge"] += 1
        continue
    window = sessions[day0 - PRE : day0 + POST + 1]
    r = stock.reindex(window).pct_change()
    if r.iloc[1:].isna().sum() > 5:
        dropped["series incomplete"] += 1
        continue
    abnormal = (r - bench_ret.reindex(window)).fillna(0).values[1:]
    paths[row.event_type].append(np.cumsum(abnormal) * 100)

added = np.vstack(paths["added"])
removed = np.vstack(paths["removed"])
axis = np.arange(-PRE + 1, POST + 1)


def at(matrix, day):
    return matrix[:, list(axis).index(day)]


print(f"\nNasdaq-100 changes 2014 to 2026: {len(events)} logged, "
      f"{len(added)} additions and {len(removed)} removals with a usable window")
for reason, n in dropped.items():
    print(f"  dropped, {reason}: {n}")
print("\nCumulative abnormal return against QQQ, per cent")
print(f"{'window':>15} {'added mean':>11} {'median':>8} "
      f"{'removed mean':>13} {'median':>8}")
for day in (-1, 1, 5, 20, 60):
    a, b = at(added, day), at(removed, day)
    label = f"day -19 to {day:+d}"
    print(f"{label:>15} {a.mean():11.2f} {np.median(a):8.2f} "
          f"{b.mean():13.2f} {np.median(b):8.2f}")

print("\nPost-event drift, day +1 to day +60, per cent")
for name, mat in [("added", added), ("removed", removed)]:
    drift = at(mat, 60) - at(mat, 1)
    print(f"  {name:<8} mean {drift.mean():6.2f}   median {np.median(drift):6.2f}"
          f"   positive on {(drift > 0).mean() * 100:.0f}% of events")

# 4. Chart: the average path through the event.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(axis, added.mean(axis=0), color="#3b82f6", linewidth=2,
        label=f"added to the index (n={len(added)})")
ax.plot(axis, removed.mean(axis=0), color="#ef4444", linewidth=2,
        label=f"removed from the index (n={len(removed)})")
ax.axvline(0, color="#6b7280", linewidth=1, linestyle="--")
ax.axhline(0, color="#3a3a3a", linewidth=1)
ax.set_xlabel("Trading days relative to the effective date")
ax.set_ylabel("Mean cumulative abnormal return, per cent")
ax.set_title("Nasdaq-100 membership changes, 2014 to 2026")
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("nasdaq-100-reconstitution-event-study-python.png", dpi=150)
