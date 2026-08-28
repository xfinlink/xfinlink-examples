# Full write-up: https://xfinlink.com/blog/turn-of-the-month-effect-python
"""
Does the turn-of-the-month effect still work?

The turn-of-the-month effect is the claim that equity returns concentrate in a
short window spanning the last trading day of one month and the first three of
the next. This script tests it on point-in-time S&P 500 membership, 2015-2025,
using an equal-weighted cross-section of daily returns.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

SLUG = "turn-of-the-month-effect-python"


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch(fn, **kw):
    for attempt in range(3):
        try:
            return fn(**kw)
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(5)
    return None


# 1. The index as it stood at each year end, so the panel is not built from
#    today's survivors.
rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
           for y in range(2014, 2025)}
universe = sorted(set().union(*rosters.values()))
print(f"point-in-time universe: {len(universe)} entities")

parts = []
for chunk in chunked(universe, 50):
    got = fetch(xfl.prices, entity_id=chunk, start="2015-01-01",
                end="2025-12-31", fields=["return_daily"], max_rows=500000)
    if got is not None:
        parts.append(got)
px = pd.concat(parts, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
px = px.dropna(subset=["return_daily"])

# A company counts on a given date only if it was in the index at the previous
# year end.
px["prior_year"] = px["date"].dt.year - 1
px = px[[eid in rosters.get(y, set())
         for eid, y in zip(px["entity_id"], px["prior_year"])]]
print(f"panel rows {len(px):,}  entities {px['entity_id'].nunique()}")

# 2. Equal-weighted cross-sectional mean return for each trading day.
daily = px.groupby("date")["return_daily"].agg(["mean", "size"])
daily = daily[daily["size"] >= 100].rename(columns={"mean": "ret"}).reset_index()
print(f"trading days {len(daily)}  "
      f"{daily['date'].min().date()} to {daily['date'].max().date()}")

# 3. Label each day by its position in the month.
daily["ym"] = daily["date"].dt.to_period("M")
daily["from_start"] = daily.groupby("ym").cumcount() + 1
daily["from_end"] = daily.groupby("ym").cumcount(ascending=False)

daily["position"] = np.nan
daily.loc[daily["from_end"] <= 4, "position"] = -(daily["from_end"] + 1)
daily.loc[daily["from_start"] <= 6, "position"] = daily["from_start"]

# The classic window: last trading day of the month plus the first three.
daily["tom"] = (daily["from_end"] == 0) | (daily["from_start"] <= 3)

inside = daily[daily["tom"]]["ret"]
outside = daily[~daily["tom"]]["ret"]
t_stat, p_val = stats.ttest_ind(inside, outside, equal_var=False)

print(f"\nturn-of-month days: n={len(inside)}  mean={inside.mean() * 100:+.4f}%")
print(f"all other days:     n={len(outside)}  mean={outside.mean() * 100:+.4f}%")
print(f"difference {(inside.mean() - outside.mean()) * 100:+.4f}pp  "
      f"t={t_stat:.2f}  p={p_val:.3f}")

print("\nmean return by position in month, tested against every other day:")
profile = []
for pos in sorted(daily["position"].dropna().unique()):
    grp = daily[daily["position"] == pos]["ret"]
    oth = daily[daily["position"] != pos]["ret"]
    tt, pp = stats.ttest_ind(grp, oth, equal_var=False)
    profile.append((int(pos), grp.mean(), tt, pp, len(grp)))
    print(f"  {int(pos):+d}: n={len(grp)}  mean={grp.mean() * 100:+.4f}%  "
          f"t={tt:+.2f}  p={pp:.3f}")

daily["year"] = daily["date"].dt.year
per_year = daily.groupby(["year", "tom"])["ret"].mean().unstack()
per_year.columns = ["other", "tom"]
diff = (per_year["tom"] - per_year["other"]) * 100
print(f"\nturn-of-month beat the rest of the month in "
      f"{int((diff > 0).sum())} of {len(diff)} years")

# 4. Chart.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"{p:+d}" for p, *_ in profile]
values = [m * 100 for _, m, *_ in profile]
colours = ["#3b82f6" if p in (-1, 1, 2, 3) else "#4b5563" for p, *_ in profile]
ax.bar(labels, values, color=colours, width=0.62)
ax.axhline(0, color="#6b7280", linewidth=0.8)
ax.set_xlabel("Trading day position (negative = counting back from month end, "
              "positive = counting from month start)")
ax.set_ylabel("Mean equal-weighted daily return (%)")
ax.set_title("Daily returns around the turn of the month, S&P 500, 2015-2025")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart written to {SLUG}.png")
