# Full write-up: https://xfinlink.com/blog/piotroski-f-score-forward-returns-python
"""Does the Piotroski F-Score separate the next year's winners from its losers?

Point-in-time S&P 500 rosters, a six-month reporting lag, June-to-June returns.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORMATION_YEARS = list(range(2018, 2026))
SKIP_SECTORS = {"Financials", "Real Estate", "Utilities"}

# 1. Point-in-time membership on each 30 June formation date.
rosters = {y: xfl.index("sp500", as_of=f"{y}-06-30").dropna(subset=["ticker"])
           for y in FORMATION_YEARS}
universe = sorted({t for r in rosters.values() for t in r["ticker"]})

# 2. Annual F-Score and monthly prices for every name that was ever a member.
mframes, pframes = [], []
for i in range(0, len(universe), 100):
    batch = universe[i : i + 100]
    mframes.append(xfl.metrics(batch, period_type="annual", start="2016-01-01",
                               end="2026-08-19", fields=["piotroski_f_score"],
                               max_rows=300000))
    pframes.append(xfl.prices(batch, start="2017-12-01", end="2026-08-19",
                              interval="1mo", fields=["adj_close"],
                              max_rows=500000))
scores = pd.concat(mframes, ignore_index=True)
px = pd.concat(pframes, ignore_index=True)

# 3. One price observation per entity per calendar month, last bar in the month.
px["date"] = pd.to_datetime(px["date"])
px["month"] = px["date"].dt.to_period("M")
last = px.sort_values("date").groupby(["entity_id", "month"], as_index=False).last()
panel = last.pivot(index="month", columns="entity_id", values="adj_close")
sector = last.groupby("entity_id")["gics_sector"].last()

scores["period_end"] = pd.to_datetime(scores["period_end"])
scores = scores.dropna(subset=["piotroski_f_score"])

# 4. For each formation year, take the most recent fiscal year that ended at
#    least six months earlier, so only already-reported figures are used.
rows = []
for year in FORMATION_YEARS:
    cutoff = pd.Timestamp(f"{year}-01-01")
    known = scores[scores["period_end"] <= cutoff]
    latest = known.sort_values("period_end").groupby("entity_id").last()

    ids = [i for i in rosters[year]["entity_id"]
           if i in panel.columns and i in latest.index
           and sector.get(i) not in SKIP_SECTORS]
    m0, m1 = pd.Period(f"{year}-06", "M"), pd.Period(f"{year + 1}-06", "M")
    ret = (panel.loc[m1, ids] / panel.loc[m0, ids] - 1).dropna()

    frame = pd.DataFrame({"f": latest.loc[ret.index, "piotroski_f_score"],
                          "ret": ret})
    frame["year"] = year
    rows.append(frame)

BUCKETS = ["low 0-3", "middle 4-6", "high 7-9"]
pool = pd.concat(rows)
pool["bucket"] = pd.cut(pool["f"], [-0.1, 3, 6, 9], labels=BUCKETS)

summary = pool.groupby("bucket", observed=True).agg(
    names=("ret", "size"),
    median_pct=("ret", lambda s: s.median() * 100),
    mean_pct=("ret", lambda s: s.mean() * 100),
    positive_pct=("ret", lambda s: (s > 0).mean() * 100),
)

by_year = (pool.pivot_table(index="year", columns="bucket", values="ret",
                            aggfunc="median", observed=True)
           .reindex(columns=BUCKETS) * 100)
by_year["high - low"] = by_year["high 7-9"] - by_year["low 0-3"]

print(f"Pooled 12-month price returns, {len(pool)} company-years, "
      f"{pool['year'].nunique()} formation dates")
print(summary.to_string(float_format=lambda v: f"{v:8.1f}"))
print()
print("Median 12-month price return by formation year, per cent")
print(by_year.to_string(float_format=lambda v: f"{v:7.1f}"))
print()
spread = by_year["high - low"]
print(f"High minus low spread positive in {int((spread > 0).sum())} of "
      f"{len(spread)} years, median {spread.median():.1f} points")

# 5. Chart: the bucket spread, year by year.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"{y}-{str(y + 1)[2:]}" for y in by_year.index]
x = range(len(labels))
width = 0.27
for offset, col, colour in [(-width, "low 0-3", "#ef4444"),
                            (0.0, "middle 4-6", "#6b7280"),
                            (width, "high 7-9", "#3b82f6")]:
    ax.bar([i + offset for i in x], by_year[col], width, label=col, color=colour)
ax.axhline(0, color="#3a3a3a", linewidth=1)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Median 12-month price return, per cent")
ax.set_title("Piotroski F-Score buckets and the next year's return")
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("piotroski-f-score-forward-returns-python.png", dpi=150)
