# Full write-up: https://xfinlink.com/blog/vol-of-vol-forward-drawdown-python
"""Does an unstable volatility level warn of a deeper drawdown?

Point-in-time S&P 500 rosters, 2016-2026. For every stock-month, measure how
much the stock's one-month realized volatility moved around over the prior six
months, then measure the worst peak-to-trough fall over the next three months.
Sort on the wobble, then sort again holding the volatility level fixed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LOOK, FWD, BLOCK = 126, 63, 21          # 6 months back, 3 months forward, 1 month
START, END = "2015-01-01", "2026-06-30"
CHART = "vol-of-vol-forward-drawdown-python.png"

# ---------------------------------------------------------------- universe
rosters = pd.concat(
    [xfl.index("sp500", as_of=f"{y}-01-02").assign(roster_year=y) for y in range(2016, 2027)]
)
member = {y: set(g["entity_id"].astype(int)) for y, g in rosters.groupby("roster_year")}
ids = sorted({int(i) for i in rosters["entity_id"].dropna()})
print(f"point-in-time rosters 2016-2026: {len(ids)} distinct companies")

# ---------------------------------------------------------------- prices
chunks = [ids[i:i + 4] for i in range(0, len(ids), 4)]


def grab(chunk):
    return xfl.prices(entity_id=chunk, start=START, end=END,
                      fields=["close", "return_daily"], max_rows=200000)


with ThreadPoolExecutor(8) as pool:
    px = pd.concat([d for d in pool.map(grab, chunks) if len(d)], ignore_index=True)

px["date"] = pd.to_datetime(px["date"])
px = px.drop_duplicates(["entity_id", "date"], keep="last")
ret = px.pivot(index="date", columns="entity_id", values="return_daily").sort_index()
print(f"daily total returns: {len(px):,} rows, {ret.shape[1]} companies, "
      f"{ret.index.min().date()} to {ret.index.max().date()}")

# ---------------------------------------------------------------- panel
dates = ret.index
month_ends = pd.DatetimeIndex(pd.Series(dates).groupby([dates.year, dates.month]).max().values)
pos = {d: i for i, d in enumerate(dates)}

rows, dropped_short, dropped_extreme = [], 0, 0
for t in month_ends:
    i = pos[t]
    if i - LOOK + 1 < 0 or i + FWD >= len(dates) or t.year not in member:
        continue
    cols = [c for c in ret.columns if int(c) in member[t.year]]
    past, fwd = ret.iloc[i - LOOK + 1:i + 1][cols], ret.iloc[i + 1:i + 1 + FWD][cols]
    complete = past.notna().all() & fwd.notna().all()
    sane = (past.abs().max() <= 0.75) & (fwd.abs().max() <= 0.75)
    dropped_short += int((~complete).sum())
    dropped_extreme += int((complete & ~sane).sum())
    cols = [c for c in cols if complete.get(c, False) and sane.get(c, False)]

    p = past[cols].to_numpy()
    rv = np.stack([p[k * BLOCK:(k + 1) * BLOCK].std(axis=0, ddof=1)
                   for k in range(LOOK // BLOCK)]) * np.sqrt(252)
    level = rv.mean(axis=0)                      # average one-month vol over six months
    vov = rv.std(axis=0, ddof=1) / level         # how much that vol moved around
    roll = pd.DataFrame(p).rolling(BLOCK).std(ddof=1)
    vov_alt = np.log(roll).diff().std(ddof=1).to_numpy() * np.sqrt(252)

    f = fwd[cols].to_numpy()
    cum = np.cumprod(1.0 + f, axis=0)
    mdd = (cum / np.maximum.accumulate(cum, axis=0) - 1.0).min(axis=0)
    rows.append(pd.DataFrame({"date": t, "entity_id": cols, "level": level, "vov": vov,
                              "vov_alt": vov_alt, "mdd": mdd, "fret": cum[-1] - 1.0}))

panel = pd.concat(rows, ignore_index=True)
print(f"panel: {len(panel):,} stock-months over {panel['date'].nunique()} formation dates, "
      f"{panel['date'].min().date()} to {panel['date'].max().date()}, "
      f"{panel['entity_id'].nunique()} companies")
print(f"  set aside: {dropped_short:,} without a full six-month history and forward quarter, "
      f"{dropped_extreme} with a one-day move beyond 75%")

cut = lambda s, n: pd.qcut(s, n, labels=False, duplicates="drop")
for col in ("vov", "vov_alt"):
    panel[col + "_q"] = panel.groupby("date")[col].transform(lambda s: cut(s, 5))
panel["lt"] = panel.groupby("date")["level"].transform(lambda s: cut(s, 3))
panel["vt"] = panel.groupby("date").apply(
    lambda g: g.groupby("lt")["vov"].transform(lambda s: cut(s, 3))).reset_index(level=0, drop=True)


def nw_t(series):
    fit = sm.OLS(series.values * 100, np.ones(len(series))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3})
    return fit.params[0], fit.tvalues[0], fit.pvalues[0]


rank = panel.groupby("date").apply(lambda g: g["level"].corr(g["vov"], method="spearman")).mean()
print(f"\nrank correlation between volatility level and vol-of-vol: {rank:.2f}")

print("\nSorted on vol-of-vol alone (quintiles rebuilt every month)")
print("           vol-of-vol   annual vol   mean 3m drawdown   median   next 3m return")
grp = panel.groupby("vov_q").agg(v=("vov", "mean"), l=("level", "mean"),
                                 m=("mdd", "mean"), md=("mdd", "median"), r=("fret", "mean"))
for k, r in grp.iterrows():
    label = "Q1 steadiest" if k == 0 else ("Q5 wobbliest" if k == 4 else f"Q{int(k)+1}")
    print(f"  {label:<14}{r.v:8.2f}{r.l*100:12.1f}%{r.m*100:17.2f}%{r.md*100:9.2f}%"
          f"{r.r*100:16.2f}%")

for col, name in (("vov", "vol-of-vol"), ("vov_alt", "vol-of-vol, log-change measure")):
    s = panel.groupby("date").apply(
        lambda g: g[g[col + "_q"] == 4]["mdd"].mean() - g[g[col + "_q"] == 0]["mdd"].mean())
    b, t, p = nw_t(s)
    print(f"  Q5 minus Q1, {name}: {b:+.2f}pp   t = {t:+.2f}   p = {p:.3f}")
s = panel.groupby("date").apply(
    lambda g: g[g["lt"] == 2]["mdd"].mean() - g[g["lt"] == 0]["mdd"].mean())
b, t, p = nw_t(s)
print(f"  For comparison, wildest minus calmest volatility third: {b:+.2f}pp   "
      f"t = {t:+.2f}   p = {p:.4f}")

print("\nMean worst 3-month fall, sorted first on volatility, then on vol-of-vol inside it")
print("                      steadiest vol   middle   wobbliest vol")
names = ["calm (20% vol)", "middle (27% vol)", "wild (40% vol)"]
grid = panel.pivot_table(index="lt", columns="vt", values="mdd", aggfunc="mean") * 100
for k in grid.index:
    a, b_, c = grid.loc[k, 0], grid.loc[k, 1], grid.loc[k, 2]
    print(f"  {names[int(k)]:<20}{a:10.2f}%{b_:10.2f}%{c:10.2f}%     "
          f"wobbliest minus steadiest {c-a:+.2f}pp")

print("\nThe six formation dates whose next quarter was worst")
spread = panel.groupby("date").apply(
    lambda g: g[g["vov_q"] == 4]["mdd"].mean() - g[g["vov_q"] == 0]["mdd"].mean()) * 100
mkt = panel.groupby("date")["mdd"].mean() * 100
for d in mkt.nsmallest(6).index:
    print(f"  {d.date()}   all stocks {mkt[d]:7.2f}%   wobbliest minus steadiest {spread[d]:+6.2f}pp")

# ---------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a"})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
shades = ["#1e3a8a", "#3b82f6", "#93c5fd"]
labels = ["steadiest volatility", "middle", "wobbliest volatility"]
for j in range(3):
    vals = [-grid.loc[k, j] for k in range(3)]
    bars = ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=shades[j], label=labels[j])
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9, color="#e0e0e0")
ax.set_xticks(x, ["calm\n20% annual vol", "middle\n27% annual vol", "wild\n40% annual vol"])
ax.set_ylabel("Average worst fall over the next 3 months (%)")
ax.set_title("Volatility of volatility does not deepen the next drawdown", pad=12)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 22)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {CHART}")
