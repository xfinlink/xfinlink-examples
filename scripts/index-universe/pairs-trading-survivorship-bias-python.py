# Full write-up: https://xfinlink.com/blog/pairs-trading-survivorship-bias-python
#
# Pairs trading backtest, distance method (Gatev, Goetzmann and Rouwenhorst style), on
# point-in-time S&P 500 rosters and on the end-2024 list. Formation 12 months, trading
# 6 months, a new pair book every January and July, trading periods 2000-2024, weekly
# total-return bars. A leg that stops trading closes its pair at the last close, with
# the recorded delisting return applied.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

N_PAIRS, OPEN_SD = 20, 2.0
CHART = "pairs-trading-survivorship-bias-python.png"
monday = lambda d: d + pd.Timedelta(days=(7 - d.weekday()) % 7)        # first Monday on or after d
starts = [monday(pd.Timestamp(f"{y}-{m:02d}-01")) for y in range(2000, 2025) for m in (1, 7)]
periods = [(monday(t - pd.DateOffset(months=12)), t, monday(t + pd.DateOffset(months=6)))
           for t in starts]                                               # formation start, trading start, trading end

# Rosters: the S&P 500 on the Friday before each trading period, and the end-2024 list
rosters = {t: xfl.index("sp500", as_of=(t - pd.Timedelta(days=3)).date().isoformat())
           .dropna(subset=["entity_id"])["entity_id"].astype(int).tolist() for _, t, _ in periods}
survivors = xfl.index("sp500", as_of="2024-12-31").dropna(subset=["entity_id"])["entity_id"].astype(int).tolist()

# Weekly total returns by entity_id, 1999-2024, in chunks of 150 ids
ids = sorted(set().union(*rosters.values(), survivors))
chunks = [xfl.prices(entity_id=ids[i:i + 150], start="1999-01-01", end="2024-12-31", interval="1w",
                     fields=["return_daily"], max_rows=400_000) for i in range(0, len(ids), 150)]
px = pd.concat(chunks, ignore_index=True)
px["week"] = px["date"] - pd.to_timedelta(px["date"].dt.dayofweek, unit="D")   # Monday of the bar's week
px = px.drop_duplicates(["entity_id", "week"])
last_bar = px.dropna(subset=["return_daily"]).groupby("entity_id")["week"].max()

# Exit records (meta.delisting) for every series that ends before the last week of 2024:
# one short request per exit year, at most 80 ids, so each answer is a single page
ended = last_bar[last_bar < pd.Timestamp("2024-12-23")]
delist = {}
for y, grp in ended.groupby(ended.dt.year):
    g = grp.index.tolist()
    for i in range(0, len(g), 80):
        d = xfl.prices(entity_id=g[i:i + 80], start=f"{y}-01-01", end="2024-12-31", interval="1w",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
ret = px.pivot(index="week", columns="entity_id", values="return_daily").sort_index()
tick = px.pivot(index="week", columns="entity_id", values="ticker").sort_index()
company = px.groupby("entity_id")["entity_name"].last()

def trade_pair(a, b, sd, R, lag=0):
    """Trade one pair through the trading period on weekly closes.
    Returns (pair return, round trips opened, (a leg stopped trading, pair open when it did))."""
    n = len(R)
    stop = {}
    alive = np.ones(n, bool)
    for leg in (a, b):
        if last_bar[leg] < R.index[-1]:               # the leg stopped trading inside the window
            k = int(np.searchsorted(R.index, last_bar[leg], side="right"))
            alive[k:] = False
            stop[leg] = k
    stop_k = int(np.argmin(alive)) if not alive.all() else n
    pa = np.cumprod(1 + R[a].fillna(0).to_numpy()[:stop_k])  # normalised total-return prices
    pb = np.cumprod(1 + R[b].fillna(0).to_numpy()[:stop_k])
    s = pa - pb
    total, trades, pos, opened = 1.0, 0, 0, None
    def close(k, final=False):
        la, lb = pa[k], pb[k]
        if final:                                     # leg that stopped: value at exit
            for leg, side in ((a, 0), (b, 1)):
                if leg in stop and stop[leg] == stop_k:
                    d = delist.get(leg, {}).get("delisting_return")
                    if d is not None and not np.isnan(d):
                        if side == 0: la = la * (1 + d)
                        else: lb = lb * (1 + d)
        ra, rb = la / pa[opened] - 1, lb / pb[opened] - 1
        return pos * (rb - ra)                        # pos=+1: short a, long b
    signal = None
    for k in range(len(s)):
        if pos == 0:
            if signal is not None and k == signal + lag:
                pos, opened, signal = (1 if s[signal] > 0 else -1), k, None
                trades += 1
            elif signal is None and abs(s[k]) > OPEN_SD * sd:
                signal = k
                if lag == 0:
                    pos, opened, signal = (1 if s[k] > 0 else -1), k, None
                    trades += 1
        elif np.sign(s[k]) != pos or s[k] == 0:      # prices crossed: close
            total *= 1 + close(k)
            pos = 0
    open_at_stop = pos != 0 and stop_k < n
    if pos != 0:
        total *= 1 + close(len(s) - 1, final=stop_k < n)
    return total - 1, trades, (stop_k < n, open_at_stop)

def run(universe_of, lag=0):
    rows, pairs_out = [], []
    for f0, t0, t1 in periods:
        F = ret[(ret.index >= f0) & (ret.index < t0)]
        R = ret[(ret.index >= t0) & (ret.index < t1)]
        u = [e for e in universe_of(t0) if e in F.columns and F[e].notna().all()]
        P = np.cumprod(1 + F[u].to_numpy(), axis=0)            # normalised formation prices
        sq = (P ** 2).sum(0)
        ssd = sq[:, None] + sq[None, :] - 2 * P.T @ P
        names = company.reindex(u).to_numpy()
        ssd[np.triu_indices(len(u))] = np.inf
        ssd[names[:, None] == names[None, :]] = np.inf          # two share classes of one company
        flat = np.argsort(ssd, axis=None)[:N_PAIRS]
        out = []
        for i, j in zip(*np.unravel_index(flat, ssd.shape)):
            a, b = u[i], u[j]
            sd = (P[:, i] - P[:, j]).std(ddof=1)
            r, n, (stopped, open_stop) = trade_pair(a, b, sd, R, lag)
            out.append(r)
            pairs_out.append({"t0": t0, "t1": R.index[-1], "a": a, "b": b, "ta": tick[a].loc[:t0].dropna().iloc[-1],
                              "tb": tick[b].loc[:t0].dropna().iloc[-1], "ret": r, "trades": n,
                              "stopped": stopped, "open_at_stop": open_stop})
        traded = [x for x, p in zip(out, pairs_out[-N_PAIRS:]) if p["trades"]]
        rows.append({"t0": t0, "n": len(u), "cc": np.mean(out), "fi": np.mean(traded) if traded else 0.0})
    return pd.DataFrame(rows), pd.DataFrame(pairs_out)

members = lambda t: rosters[t]
books = {"Point-in-time rosters": run(members),
         "End-2024 list": run(lambda t: survivors),
         "End-2024 list, members then": run(lambda t: [e for e in survivors if e in set(rosters[t])])}
lagged = {k: run(f, lag=1)[0]["cc"].mean() for k, f in
          [("Point-in-time rosters", members), ("End-2024 list", lambda t: survivors)]}

# ---- output ------------------------------------------------------------------
tstat = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(len(x))
print(f"Securities priced: {len(ids)}; trading periods: {len(periods)}, "
      f"{periods[0][1].date()} to {min(periods[-1][2] - pd.Timedelta(days=1), pd.Timestamp('2024-12-31')).date()}; "
      f"top {N_PAIRS} pairs a period, open at {OPEN_SD:.0f} SD, weekly total-return closes")
print("Pair return = long leg return minus short leg return, $1 a side; committed capital = mean over "
      "all 20 pairs, fully invested = mean over pairs that opened\n")
print(f"{'':30}{'names':>6}{'6m cc':>8}{'t':>6}{'6m fi':>8}{'periods +':>10}{'opened':>8}{'stopped leg':>12}")
for name, (per, pr) in books.items():
    op = pr[pr["trades"] > 0]
    print(f"{name:<30}{per['n'].mean():>6.0f}{per['cc'].mean():>8.2%}{tstat(per['cc']):>6.2f}{per['fi'].mean():>8.2%}"
          f"{(per['cc'] > 0).mean():>10.0%}{len(op) / len(pr):>8.1%}{op['stopped'].mean():>12.1%}")
A, B, C = (books[k][0]["cc"] for k in books)
print(f"\nPoint-in-time minus end-2024 list: {(A - B).mean():+.2%} per 6 months (t = {tstat(A - B):.2f}); "
      f"minus members-then: {(A - C).mean():+.2%} (t = {tstat(A - C):.2f})")
print("Growth of $1 of committed capital over the 50 periods: " +
      ", ".join(f"{k} {np.prod(1 + v[0]['cc']):.2f}" for k, v in books.items()))
print("Entry one week after the signal: " + ", ".join(f"{k} {v:.2%}" for k, v in lagged.items()))

pr = books["Point-in-time rosters"][1]
op = pr[pr["trades"] > 0]
st = op[op["stopped"]]
print(f"\nPoint-in-time book: {len(op)} pairs opened, {len(st)} with a leg that stopped trading in the window "
      f"({st['open_at_stop'].sum()} open at the time)")
print(f"  mean 6m return: those {len(st)} pairs {st['ret'].mean():.2%}, the other {len(op) - len(st)} {op[~op['stopped']]['ret'].mean():.2%}")
for r in st.itertuples():
    for leg, tk in ((r.a, r.ta), (r.b, r.tb)):
        if last_bar[leg] < r.t1:
            x = delist.get(leg, {})
            dr = x.get("delisting_return")
            print(f"  {r.t0.date()}  {r.ta:>5}/{r.tb:<5} {tk:>5} last week {last_bar[leg].date()}  "
                  f"{x.get('event_type', 'no record'):<9} exit {'n/a' if dr is None else f'{dr:+.2%}':>7}  pair {r.ret:+.2%}")

print("\nMean 6-month committed-capital return by sub-period (share of periods positive)")
sub = pd.cut(pd.Series([p[1].year for p in periods]), [1999, 2007, 2015, 2024], labels=["2000-07", "2008-15", "2016-24"])
for name, (per, _) in books.items():
    g = per["cc"].groupby(sub.to_numpy(), observed=True)
    print(f"  {name:<30}" + "".join(f"{k} {v:6.2%} ({(per['cc'][sub == k] > 0).mean():.0%})  " for k, v in g.mean().items()))

bp = books["End-2024 list"][1]
late = bp.apply(lambda r: r.a not in set(rosters[r.t0]) or r.b not in set(rosters[r.t0]), axis=1)
print(f"\nEnd-2024 book: {late.sum()} of {len(bp)} pairs have a leg not yet in the index at selection; "
      f"mean 6m return {bp[late]['ret'].mean():.2%} against {bp[~late]['ret'].mean():.2%} for the rest")

# ---- chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#a3a3a3", "ytick.color": "#a3a3a3", "axes.edgecolor": "#404040"})
fig, ax = plt.subplots(figsize=(10, 5))
x = [periods[0][1]] + [p[2] for p in periods]
for name, colour, style in [("End-2024 list", "#d97706", "-"), ("End-2024 list, members then", "#a3a3a3", "--"),
                            ("Point-in-time rosters", "#3b82f6", "-")]:
    y = np.r_[1, np.cumprod(1 + books[name][0]["cc"].to_numpy())]
    ax.plot(x, y, color=colour, linestyle=style, linewidth=2, label=name)
    ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color="#e0e0e0", fontsize=10)
ax.axhline(1, color="#404040", linewidth=1)
ax.set_xlabel("End of trading period")
ax.set_ylabel("Value of $1 of committed capital")
ax.set_title("Pairs trading backtest, top 20 S&P 500 pairs, 2000-2024\n"
             "point-in-time rosters vs the end-2024 list (weekly total-return closes)", fontsize=12)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(CHART, dpi=150)
