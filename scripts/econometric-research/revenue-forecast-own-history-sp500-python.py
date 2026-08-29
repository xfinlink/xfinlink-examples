# Full write-up: https://xfinlink.com/blog/revenue-forecast-own-history-sp500-python
"""
Can a company's revenue be forecast from its own history?

Five forecasting rules, each estimable from annual revenue alone, scored out of
sample against actual revenue one year ahead. Ten forecast origins (2015-2024)
on point-in-time S&P 500 membership, carried by entity_id.
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

SLUG = "revenue-forecast-own-history-sp500-python"
ORIGINS = range(2015, 2025)


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


# 1. The index as it stood at each year end, so no company enters the sample on
#    the strength of a membership it did not yet have.
rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
           for y in ORIGINS}
universe = sorted(set().union(*rosters.values()))
print(f"point-in-time universe: {len(universe)} entities")

parts = []
for chunk in chunked(universe, 50):
    got = fetch(xfl.fundamentals, entity_id=chunk, period_type="annual",
                start="2008-01-01", end="2026-08-28", fields=["revenue"],
                max_rows=50000)
    if got is not None and len(got):
        parts.append(got)
fu = pd.concat(parts, ignore_index=True)
fu = fu[fu["revenue"] > 0].copy()

# 2. Anchor each statement to the calendar year it covers, taking that from the
#    period end rather than the fiscal year label. A year end falling in January
#    to May belongs to the previous year, which is the retail convention.
fu["period_end"] = pd.to_datetime(fu["period_end"])
fu["month"] = fu["period_end"].dt.month
fu["y"] = np.where(fu["month"] <= 5, fu["period_end"].dt.year - 1,
                   fu["period_end"].dt.year)

# Where a company reports two annual periods landing in the same year, keep the
# one matching its usual fiscal year end.
modal = fu.groupby("entity_id")["month"].agg(lambda s: s.mode().iloc[0])
fu["off"] = (fu["month"] - fu["entity_id"].map(modal)).abs()
dup_rows = int(fu.duplicated(["entity_id", "y"], keep=False).sum())
fu = fu.sort_values(["entity_id", "y", "off", "period_end"])
fu = fu.drop_duplicates(["entity_id", "y"], keep="first")
print(f"annual observations {len(fu):,}  resolved to one per company-year "
      f"from {dup_rows} overlapping rows")

rev = fu.pivot_table(index="entity_id", columns="y", values="revenue")
sector = fu.sort_values("y").groupby("entity_id")["gics_sector"].last()

# 3. Walk forward. At each origin t the forecast sees revenue through year t and
#    nothing else; the target is year t+1.
rows = []
for t in ORIGINS:
    for eid in rev.index:
        if eid not in rosters[t]:
            continue
        hist = [rev.at[eid, y] if y in rev.columns else np.nan
                for y in range(t - 4, t + 2)]
        if any(pd.isna(hist)):
            continue
        r = np.asarray(hist, dtype=float)
        g = r[1:5] / r[0:4] - 1               # growth in years t-3 .. t
        rows.append(dict(origin=t, entity_id=eid, R=r[4], actual=r[5],
                         g_last=g[-1], g_avg=g.mean(),
                         g_actual=r[5] / r[4] - 1))
p = pd.DataFrame(rows)
print(f"forecasts {len(p):,}  companies {p['entity_id'].nunique()}  "
      f"origins {p['origin'].min()}-{p['origin'].max()}")

# 4. Five rules. The peer median is the cross-sectional median growth among the
#    companies in that origin year, which is known at the time of the forecast.
p["peer"] = p.groupby("origin")["g_last"].transform("median")
MODELS = {
    "no growth (random walk)": p["R"],
    "last year's growth": p["R"] * (1 + p["g_last"]),
    "4-year average growth": p["R"] * (1 + p["g_avg"]),
    "own growth shrunk halfway": p["R"] * (1 + 0.5 * p["g_last"] + 0.5 * p["peer"]),
    "peer median growth only": p["R"] * (1 + p["peer"]),
}
ape = pd.DataFrame({k: (v - p["actual"]).abs() / p["actual"] * 100
                    for k, v in MODELS.items()})
base = ape["no growth (random walk)"]

print("\nout-of-sample absolute percentage error, one year ahead")
print(f"{'rule':28s}{'median':>8s}{'mean':>8s}{'p75':>8s}{'p90':>8s}{'beats RW':>10s}")
for k in MODELS:
    a = ape[k]
    beat = "" if k.startswith("no growth") else f"{(a < base).mean() * 100:9.1f}%"
    print(f"{k:28s}{a.median():8.2f}{a.mean():8.2f}{a.quantile(.75):8.2f}"
          f"{a.quantile(.90):8.2f}{beat:>10s}")

print("\ngain over the random walk, tested across the ten forecast origins")
for k in list(MODELS)[1:]:
    per = (ape.assign(o=p["origin"]).groupby("o")
           .apply(lambda d: d["no growth (random walk)"].median() - d[k].median()))
    t_stat, p_val = stats.ttest_1samp(per, 0)
    print(f"{k:28s}{per.mean():+6.2f}pp  wins {int((per > 0).sum())}/10  "
          f"t={t_stat:+.2f}  p={p_val:.3f}")

fit = stats.linregress(p["g_last"], p["g_actual"])
rho = stats.spearmanr(p["g_last"], p["g_actual"])
print(f"\nnext year's growth on this year's growth: slope {fit.slope:+.3f}  "
      f"r2={fit.rvalue ** 2:.3f}  rank correlation {rho.statistic:+.3f}")

p["gdec"] = p.groupby("origin")["g_last"].transform(
    lambda s: pd.qcut(s, 10, labels=False) + 1)
band = p.groupby("gdec")[["g_last", "g_actual"]].median() * 100
print("\ngrowth this year vs growth next year, by decile of this year's growth")
for d, r in band.iterrows():
    print(f"  decile {int(d):2d}: {r['g_last']:+7.2f}%  ->  {r['g_actual']:+6.2f}%")

print("\nmedian error by sector, random walk vs the best rule")
sec = (ape.assign(s=p["entity_id"].map(sector)).dropna(subset=["s"])
       .groupby("s")[["no growth (random walk)", "own growth shrunk halfway"]]
       .agg(["median", "size"]))
for s, r in sec.sort_values(("no growth (random walk)", "median")).iterrows():
    print(f"  {s:24s} n={int(r[('no growth (random walk)', 'size')]):4d}  "
          f"RW {r[('no growth (random walk)', 'median')]:6.2f}%   "
          f"shrunk {r[('own growth shrunk halfway', 'median')]:6.2f}%")

print("\nmedian error by revenue size decile at the forecast origin")
p["sdec"] = p.groupby("origin")["R"].transform(
    lambda s: pd.qcut(s, 10, labels=False) + 1)
size = ape.assign(d=p["sdec"], R=p["R"]).groupby("d").agg(
    rev=("R", "median"), rw=("no growth (random walk)", "median"),
    shrunk=("own growth shrunk halfway", "median"))
for d, r in size.iterrows():
    print(f"  decile {int(d):2d}: median revenue ${r['rev'] / 1000:7.1f}bn   "
          f"RW {r['rw']:5.2f}%   shrunk {r['shrunk']:5.2f}%")

# 5. Chart.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

meds = ape.median()
cols = ["#3b82f6" if k == meds.idxmin() else "#4b5563" for k in meds.index]
ax1.barh(list(meds.index), meds.values, color=cols, height=0.6)
for i, v in enumerate(meds.values):
    ax1.text(v + 0.08, i, f"{v:.2f}%", va="center", color="#e0e0e0", fontsize=10)
ax1.set_xlim(0, meds.max() * 1.18)
ax1.invert_yaxis()
ax1.set_xlabel("Median error one year ahead (% of actual revenue)")
ax1.set_title("One-year-ahead revenue forecast error, S&P 500, 2016-2025")
ax1.spines[["top", "right"]].set_visible(False)

lim = [band["g_last"].min() - 4, band["g_last"].max() + 4]
ax2.plot(lim, lim, color="#6b7280", linestyle="--", linewidth=1,
         label="if growth repeated exactly")
ax2.plot(band["g_last"], band["g_actual"], "o-", color="#3b82f6",
         markersize=6, linewidth=1.8, label="what actually happened")
ax2.set_xlabel("Revenue growth this year (%), companies grouped into ten bands")
ax2.set_ylabel("Growth next year (%)")
ax2.set_title("Why the rules barely differ: fast growth does not repeat")
ax2.legend(frameon=False, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart written to {SLUG}.png")
