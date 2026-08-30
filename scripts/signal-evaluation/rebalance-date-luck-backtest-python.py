# Full write-up: https://xfinlink.com/blog/rebalance-date-luck-backtest-python
#
# Rebalance-date luck: one 12-1 momentum decile sort, run 21 times, changing
# nothing except the day of the month the portfolio is re-formed.

import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SNAPSHOTS = pd.date_range("2014-11-30", "2026-07-31", freq="ME").strftime("%Y-%m-%d").tolist()
ANCHORS = [s for s in SNAPSHOTS if s >= "2016-01-01"]
START, END = "2014-11-01", "2026-07-31"
FORM, SKIP, OFFSETS = 252, 21, 21
PNG = "rebalance-date-luck-backtest-python.png"

# ---------------------------------------------------------------- data ------


def roster(as_of):
    return sorted(int(e) for e in xfl.index("sp500", as_of=as_of)["entity_id"].dropna())


def series(entity):
    return xfl.prices(entity_id=entity, start=START, end=END,
                      fields=["close", "return_daily"], max_rows=200000)


with ThreadPoolExecutor(8) as pool:
    rosters = dict(zip(SNAPSHOTS, pool.map(roster, SNAPSHOTS)))
universe = sorted({e for ids in rosters.values() for e in ids})

with ThreadPoolExecutor(8) as pool:
    px = pd.concat([d for d in pool.map(series, universe) if len(d)], ignore_index=True)

px = px.drop_duplicates(["entity_id", "date"])
px = px[px["close"] > 0]
R = px.pivot(index="date", columns="entity_id", values="return_daily").sort_index()
priced = R.shape[1]
R = R[[c for c in R.columns if not ((R[c] > 1.0).any() or (R[c] < -0.5).any())]]

cal = R.index
logret = np.log1p(R.fillna(0.0)).cumsum().values   # cumulative log total return
live = R.notna().cumsum().values                   # cumulative count of traded days
pos = {e: i for i, e in enumerate(R.columns)}
snaps = pd.to_datetime(SNAPSHOTS)
month_end = pd.Series(np.arange(len(cal))).groupby(np.asarray(cal.year * 100 + cal.month)).max()

# ------------------------------------------------------------ backtest ------


def backtest(offset):
    """Re-form the deciles 'offset' trading days before each calendar month end."""
    days = [int(month_end[int(a[:4]) * 100 + int(a[5:7])]) - offset for a in ANCHORS]
    rows, tops, counts = [], [], []
    for t, nxt in zip(days, days[1:]):
        members = rosters[SNAPSHOTS[snaps.searchsorted(cal[t], "left") - 1]]
        ids = np.array([pos[e] for e in members if e in pos])
        usable = ((live[t - SKIP, ids] - live[t - FORM, ids] == FORM - SKIP)
                  & (live[nxt, ids] - live[t, ids] == nxt - t))
        ids = ids[usable]
        signal = logret[t - SKIP, ids] - logret[t - FORM, ids]      # months t-12 to t-1
        held = np.expm1(logret[nxt, ids] - logret[t, ids])          # the month held
        rank = np.argsort(np.argsort(signal, kind="stable"), kind="stable")
        decile = rank * 10 // len(ids)
        rows.append((cal[nxt], held[decile == 9].mean(), held[decile == 0].mean()))
        tops.append(frozenset(ids[decile == 9]))
        counts.append(len(ids))
    out = pd.DataFrame(rows, columns=["date", "winners", "losers"]).set_index("date")
    out["spread"] = out["winners"] - out["losers"]
    return out, tops, counts, pd.DatetimeIndex(cal[days])


runs = {k: backtest(k) for k in range(OFFSETS)}
n = len(runs[0][0])


def annualised(x):
    return np.prod(1 + x) ** (12 / len(x)) - 1


summary = pd.DataFrame({
    k: {"day": int(np.median(runs[k][3].day)),
        "names": int(np.median(runs[k][2])),
        "winners": annualised(runs[k][0]["winners"]),
        "losers": annualised(runs[k][0]["losers"]),
        "spread": annualised(runs[k][0]["spread"]),
        "vol": runs[k][0]["spread"].std() * np.sqrt(12),
        "sharpe": runs[k][0]["spread"].mean() / runs[k][0]["spread"].std() * np.sqrt(12),
        "t": runs[k][0]["spread"].mean() / (runs[k][0]["spread"].std() / np.sqrt(n))}
    for k in runs}).T

best, worst = summary["spread"].idxmax(), summary["spread"].idxmin()
spreads = pd.DataFrame({k: runs[k][0]["spread"].to_numpy() for k in runs})
pairwise = spreads.corr().to_numpy()[np.triu_indices(OFFSETS, 1)].mean()
recent = spreads[np.asarray(runs[0][0].index.year) >= 2021].apply(annualised)
month_gap = (spreads.max(axis=1) - spreads.min(axis=1)).median()
overlap = np.mean([len(a & b) / len(a) for a, b in zip(runs[0][1], runs[10][1])])
stderr = runs[0][0]["spread"].std() / np.sqrt(n) * 12

# --------------------------------------------------------------- print ------

print("Rebalance-date sensitivity of a 12-1 momentum decile sort, point-in-time S&P 500")
print(f"Panel: {len(SNAPSHOTS)} month-end rosters covering {len(universe)} companies; "
      f"{priced} carry a daily series for the window and {priced - R.shape[1]} are set aside")
print(f"       by the return screen, leaving {R.shape[1]} names and {int(R.notna().sum().sum()):,} daily observations")
print(f"Runs:  {OFFSETS} rebalance offsets, each with {len(ANCHORS)} rebalance dates and "
      f"{n} holding periods, {ANCHORS[0][:7]} to {ANCHORS[-1][:7]}")
print("Offset k re-forms the deciles k trading days before the calendar month end (k=0 is month end)")
print()
print("Offset  Typical  Names   Winners   Losers   Long-short   Ann vol  Sharpe      t")
print(" (days)   day    ranked   a year   a year       a year")
for k in runs:
    r = summary.loc[k]
    print(f"{k:5.0f}   {r['day']:6.0f}   {r['names']:5.0f}  {r['winners']:7.2%}  "
          f"{r['losers']:7.2%}     {r['spread']:8.2%}   {r['vol']:7.2%}  {r['sharpe']:6.2f}  {r['t']:5.2f}")
print()
print("Spread of outcomes across the 21 rebalance offsets")
print(f"  Long-short a year        best  {summary.loc[best, 'spread']:7.2%}  "
      f"(offset {best:.0f}, around the {summary.loc[best, 'day']:.0f}th)")
print(f"                           worst {summary.loc[worst, 'spread']:7.2%}  "
      f"(offset {worst:.0f}, around the {summary.loc[worst, 'day']:.0f}th)")
print(f"                           gap   {summary.loc[best, 'spread'] - summary.loc[worst, 'spread']:7.2%} a year, "
      f"standard deviation {summary['spread'].std():.2%}")
print(f"  Winners leg a year       {summary['winners'].min():7.2%} to {summary['winners'].max():.2%}, "
      f"gap {summary['winners'].max() - summary['winners'].min():.2%}")
print(f"  Losers leg a year        {summary['losers'].min():7.2%} to {summary['losers'].max():.2%}, "
      f"gap {summary['losers'].max() - summary['losers'].min():.2%}")
print(f"  Sharpe                   {summary['sharpe'].min():7.2f} to {summary['sharpe'].max():.2f}")
print(f"  t-statistic              {summary['t'].min():7.2f} to {summary['t'].max():.2f}")
print(f"  One dollar becomes       {np.prod(1 + runs[best][0]['spread']):7.2f} at the best offset "
      f"against {np.prod(1 + runs[worst][0]['spread']):.2f} at the worst")
print(f"  The 2021-2026 half alone {recent.min():7.2%} to {recent.max():.2%}, "
      f"gap {recent.max() - recent.min():.2%}")
print()
print(f"  Typical gap between the best and worst offset in one month  {month_gap:.2%}")
print(f"  Sampling error of the month-end run alone, annualised   {stderr:.2%}")
print(f"  Mean pairwise correlation of the 21 monthly spreads     {pairwise:.3f}")
print(f"  Winner decile shared, month-end sort against mid-month  {overlap:.1%}")

# --------------------------------------------------------------- chart ------

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

colours = ["#f59e0b" if k == 0 else "#3b82f6" for k in runs]
ax1.bar(list(runs), summary["spread"] * 100, color=colours)
ax1.axhline(0, color="#e0e0e0", linewidth=0.8)
ax1.set_xticks(list(runs))
ax1.set_xticklabels([f"{summary.loc[k, 'day']:.0f}" for k in runs], fontsize=8)
ax1.set_xlabel("Day of the month the portfolio is re-formed (typical calendar day)")
ax1.set_ylabel("Winners minus losers, % a year")
ax1.set_title("Rebalance-date luck: one momentum strategy, 21 rebalance days, S&P 500 2016-2026")
ax1.annotate("the usual choice", xy=(0.45, -2.3), xytext=(2.0, -1.3),
             color="#f59e0b", fontsize=9, arrowprops={"color": "#f59e0b", "arrowstyle": "-"})

for k in runs:
    ax2.plot(runs[k][0].index, np.cumprod(1 + runs[k][0]["spread"]),
             color="#3b82f6", alpha=0.25, linewidth=1)
ax2.plot(runs[best][0].index, np.cumprod(1 + runs[best][0]["spread"]),
         color="#e0e0e0", linewidth=1.8, label=f"luckiest day (the {summary.loc[best, 'day']:.0f}th)")
ax2.plot(runs[worst][0].index, np.cumprod(1 + runs[worst][0]["spread"]),
         color="#f59e0b", linewidth=1.8, label=f"unluckiest day (the {summary.loc[worst, 'day']:.0f}th)")
ax2.set_ylabel("Value of one dollar")
ax2.set_title("Growth of one dollar in the same spread, all 21 rebalance days", fontsize=10)
ax2.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper right")
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
