# Full write-up: https://xfinlink.com/blog/sp500-return-dispersion-python
"""How far apart do S&P 500 members move from each other? Cross-sectional return
dispersion by calendar year, 2006-2025.

Dispersion is the standard deviation of member total returns measured across the
index at one point in time, not through time. Index membership is rebuilt at the
start of every year from the roster as it stood on that date and carried by
company identifier, so a company removed later still counts for the years it was
a member. Daily total returns are compounded into calendar-year returns inside
the script rather than requested as long bars.
"""
import time

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "sp500-return-dispersion-python"
YEARS = list(range(2006, 2026))
CHUNK = 50


def fetch(**kwargs):
    """One price call, retried, so a transient failure never leaves a hole."""
    for attempt in range(5):
        try:
            return xfl.prices(**kwargs)
        except xfl.XfinlinkError as exc:
            last = exc
            time.sleep(4 * (attempt + 1))
    raise last


# ── Calendar-year total return for every index member ─────────────────
rows, annual = [], {}
for year in YEARS:
    roster = xfl.index("sp500", as_of=f"{year}-01-01")
    ids = sorted({int(e) for e in roster["entity_id"].dropna()})
    px = pd.concat([fetch(entity_id=ids[i:i + CHUNK], start=f"{year}-01-01",
                          end=f"{year}-12-31", fields=["adj_close", "return_daily"],
                          max_rows=200_000)
                    for i in range(0, len(ids), CHUNK)], ignore_index=True)
    px = (px.drop_duplicates(["entity_id", "date"]).dropna(subset=["return_daily"])
            .sort_values(["entity_id", "date"]))

    days = px["date"].nunique()
    counts = px.groupby("entity_id")["return_daily"].size()
    # two screens: a member needs a price series covering the window, and its daily
    # returns have to agree with its own price path day by day. Names that fail
    # either one drop from the sample.
    step = (np.log1p(px["return_daily"]) - np.log(px["adj_close"]).groupby(px["entity_id"]).diff())
    agrees = step.abs().groupby(px["entity_id"]).max()
    kept = counts[counts >= 0.95 * days].index.intersection(agrees[agrees <= 0.5].index)
    ann = px[px["entity_id"].isin(kept)].groupby("entity_id")["return_daily"].apply(
        lambda s: float(np.prod(1.0 + s.values) - 1.0))
    annual[year] = ann

    n10 = int(round(len(ann) * 0.1))
    rows.append(dict(year=year, days=days, members=len(ann), dispersion=ann.std(ddof=1),
                     average=ann.mean(), top=ann.nlargest(n10).mean(),
                     bottom=ann.nsmallest(n10).mean()))

t = pd.DataFrame(rows).set_index("year")
t["spread"] = t["top"] - t["bottom"]

# ── The index itself, for the volatility comparison ───────────────────
spy = fetch(ticker="SPY", start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31",
            fields=["return_daily"], max_rows=200_000).dropna(subset=["return_daily"])
spy["year"] = spy["date"].dt.year
t["index_vol"] = spy.groupby("year")["return_daily"].std(ddof=1) * np.sqrt(252)
t["index_ret"] = spy.groupby("year")["return_daily"].apply(lambda s: np.prod(1.0 + s.values) - 1.0)

# ── Output ────────────────────────────────────────────────────────────
print(f"S&P 500 cross-sectional return dispersion, {YEARS[0]}-{YEARS[-1]}")
print(f"point-in-time membership, {int(t['members'].sum()):,} member-years, "
      f"{t['days'].min()}-{t['days'].max()} trading days per year\n")
head = ("year  members  dispersion  index vol  index return   top decile  "
        "bottom decile  decile spread")
print(head)
print("-" * len(head))
for y, r in t.iterrows():
    print(f"{y}      {r['members']:3.0f}      {r['dispersion']:6.1%}     {r['index_vol']:6.1%}"
          f"      {r['index_ret']:+7.1%}      {r['top']:+7.1%}       {r['bottom']:+7.1%}"
          f"       {100 * r['spread']:6.1f}pp")

wide, narrow = t["dispersion"].idxmax(), t["dispersion"].idxmin()
loud = t["index_vol"].idxmax()
print(f"\nwidest dispersion    {wide}  {t.loc[wide, 'dispersion']:.1%}"
      f"   (index volatility {t.loc[wide, 'index_vol']:.1%})")
print(f"narrowest dispersion {narrow}  {t.loc[narrow, 'dispersion']:.1%}"
      f"   (index volatility {t.loc[narrow, 'index_vol']:.1%})")
print(f"loudest index        {loud}  index volatility {t.loc[loud, 'index_vol']:.1%}"
      f", dispersion {t.loc[loud, 'dispersion']:.1%}")

print(f"\naverage dispersion {t['dispersion'].mean():.1%}, median {t['dispersion'].median():.1%}")

later, earlier = t["dispersion"].values[1:], t["index_vol"].values[:-1]
r_same, p_same = stats.pearsonr(t["dispersion"], t["index_vol"])
r_lag, p_lag = stats.pearsonr(later, earlier)
r_own, p_own = stats.pearsonr(t["dispersion"].values[1:], t["dispersion"].values[:-1])
print(f"\ndispersion against index volatility, same year   r={r_same:+.2f}  p={p_same:.3f}  "
      f"rank r={stats.spearmanr(t['dispersion'], t['index_vol'])[0]:+.2f}")
print(f"dispersion against index volatility, prior year  r={r_lag:+.2f}  p={p_lag:.3f}  "
      f"rank r={stats.spearmanr(later, earlier)[0]:+.2f}")
# one pair can carry a correlation this size, so refit the lagged test 19 times,
# leaving a different year out each time
loo = {y: stats.pearsonr(np.delete(later, i), np.delete(earlier, i))[0]
       for i, y in enumerate(t.index[1:])}
weakest = min(loo, key=loo.get)
print(f"   leaving one year out, that r runs {min(loo.values()):+.2f} to {max(loo.values()):+.2f}; "
      f"without {weakest} alone it is {loo[weakest]:+.2f}")
print(f"dispersion against its own prior year            r={r_own:+.2f}  p={p_own:.3f}")

normal = 2 * stats.norm.pdf(stats.norm.ppf(0.9)) / 0.1
print(f"\ndecile spread divided by dispersion: mean {(t['spread'] / t['dispersion']).mean():.2f}, "
      f"range {(t['spread'] / t['dispersion']).min():.2f}-{(t['spread'] / t['dispersion']).max():.2f}"
      f"  ({normal:.2f} if annual returns were normal)")

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1]})

ax1.plot(t.index, 100 * t["dispersion"], color="#3b82f6", marker="o", markersize=4,
         linewidth=2, label="Spread between member returns (dispersion)")
ax1.plot(t.index, 100 * t["index_vol"], color="#e0e0e0", marker="o", markersize=4,
         linewidth=1.4, linestyle="--", label="Volatility of the index itself")
ax1.set_ylabel("Percent per year")
ax1.set_title("How far apart S&P 500 members move, and how much the index itself moves")
ax1.legend(frameon=False, loc="upper right")

ax2.bar(t.index, 100 * t["spread"], color="#3b82f6", width=0.62)
ax2.set_ylabel("Percentage points")
ax2.set_xlabel("Calendar year")
ax2.set_title("Gap between the best and worst tenth of members, same year")
ax2.set_xticks(t.index)
ax2.tick_params(axis="x", rotation=45)
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
