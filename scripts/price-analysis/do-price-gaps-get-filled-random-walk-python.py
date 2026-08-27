# Full write-up: https://xfinlink.com/blog/do-price-gaps-get-filled-random-walk-python
"""Do opening price gaps get filled, and does that beat a random walk?

Universe: every member of the S&P 500 as of 2 January 2016, followed by entity id
through 2025, so names later removed from the index stay in the sample.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy.stats import norm

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2016-01-01", "2025-12-31"
MINGAP, MAXGAP, MINPX = 0.03, 0.25, 5.0
VOLWIN, MAXH = 60, 60
HOR = [5, 20, 60]
SLUG = "do-price-gaps-get-filled-random-walk-python"

# ── 1. point-in-time universe, addressed by entity id ────────────────────────
roster = xfl.index("sp500", as_of="2016-01-01")
ids = sorted(int(i) for i in roster["entity_id"])
print(f"S&P 500 roster as of 2016-01-01: {len(ids)} entity ids")


def pull(chunk):
    return xfl.prices(entity_id=chunk, start=START, end=END,
                      fields=["open", "high", "low", "close", "adj_close",
                              "dividend", "split_ratio"],
                      max_rows=300000)


t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    px = pd.concat(ex.map(pull, [ids[i:i + 50] for i in range(0, len(ids), 50)]),
                   ignore_index=True)
print(f"{len(px):,} daily bars in {time.time() - t0:.0f}s")

# ── 2. gap events ────────────────────────────────────────────────────────────
rows, series_used, bars = [], 0, 0
for (eid, tk), g in px.groupby(["entity_id", "ticker"], sort=False):
    g = g.sort_values("date")
    if len(g) < VOLWIN + MAXH + 5:
        continue
    f = (g["adj_close"] / g["close"]).to_numpy()      # split factor for that bar
    o, h, lo = (g[k].to_numpy() * f for k in ("open", "high", "low"))
    c = g["adj_close"].to_numpy()
    raw = g["close"].to_numpy()
    div = pd.notna(g["dividend"]).to_numpy()
    d = g["date"].to_numpy()
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & (c > 0) & (o > 0)
    if ok.sum() < VOLWIN + MAXH + 5:
        continue
    o, h, lo, c, div, d, raw = (a[ok] for a in (o, h, lo, c, div, d, raw))
    n = len(c)
    series_used += 1
    bars += n

    logret = np.diff(np.log(c), prepend=np.nan)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    prev_raw = np.concatenate([[np.nan], raw[:-1]])
    gap = o / prev_c - 1.0
    daygap = (d - np.concatenate([[d[0]], d[:-1]])).astype("timedelta64[D]").astype(int)

    a = np.abs(gap)
    idx = np.where(np.isfinite(gap) & (a >= MINGAP) & (a <= MAXGAP) & (~div)
                   & (prev_raw >= MINPX) & (daygap <= 7)
                   & (np.arange(n) >= VOLWIN) & (np.arange(n) <= n - MAXH))[0]
    for i in idx:
        pre = logret[i - VOLWIN:i]
        pre = pre[np.isfinite(pre)]
        if len(pre) < VOLWIN - 5:
            continue
        seg = np.concatenate([[np.log(c[i] / o[i])],
                              np.log(c[i + 1:i + MAXH] / c[i:i + MAXH - 1])])
        if not np.all(np.isfinite(seg)):
            continue
        r = {"ticker": tk, "date": d[i], "gap": gap[i], "up": bool(gap[i] > 0),
             "prev_c": prev_c[i], "open": o[i], "sig_pre": pre.std(ddof=1),
             "fill1": bool(lo[i] <= prev_c[i]) if gap[i] > 0 else bool(h[i] >= prev_c[i])}
        for N in HOR:
            r[f"fill{N}"] = bool(lo[i:i + N].min() <= prev_c[i]) if gap[i] > 0 \
                else bool(h[i:i + N].max() >= prev_c[i])
            r[f"ret{N}"] = c[i + N - 1] / o[i] - 1.0
            r[f"sig{N}"] = seg[:N].std(ddof=1)
        rows.append(r)

ev = pd.DataFrame(rows)
ev["date"] = pd.to_datetime(ev["date"])
ev["year"] = ev["date"].dt.year
ev["bucket"] = pd.cut(ev["gap"].abs(), [0.03, 0.05, 0.08, 0.12, 0.25],
                      labels=["3-5%", "5-8%", "8-12%", "12-25%"], include_lowest=True)


def rw(d, N, sig):
    """Reflection-principle probability of touching the prior close within N sessions."""
    return 2 * norm.cdf(-np.abs(np.log(d["open"] / d["prev_c"])) / (d[sig] * np.sqrt(N)))


# ── 3. date-matched drift benchmark ──────────────────────────────────────────
parts = []
for (eid, tk), g in px.groupby(["entity_id", "ticker"], sort=False):
    g = g.sort_values("date")
    c = g["adj_close"].to_numpy()
    dd = g["date"].to_numpy()
    ok = np.isfinite(c) & (c > 0)
    c, dd = c[ok], dd[ok]
    if len(c) < 80:
        continue
    p = {"date": dd}
    for N in (20, 60):
        f = np.full(len(c), np.nan)
        f[:len(c) - N] = c[N:] / c[:len(c) - N] - 1
        p[f"m{N}"] = f
    parts.append(pd.DataFrame(p))
mkt = pd.concat(parts).groupby("date").median()
ev = ev.merge(mkt.reset_index(), on="date", how="left")
for N in (20, 60):
    ev[f"x{N}"] = ev[f"ret{N}"] - ev[f"m{N}"]

# ── 4. output ────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Do opening price gaps get filled?  S&P 500 roster of 2 January 2016, "
      "followed to 2025")
print(f"{series_used} price series, {bars:,} daily bars, "
      f"{len(ev):,} gaps of {MINGAP:.0%} to {MAXGAP:.0%}")
print("=" * 78)

rng = np.random.default_rng(7)
print("\nEstimator check: simulated driftless random walk, 26 steps a session, "
      "40,000 paths")
print(f"{'barrier':>9}{'daily vol':>11}{'sessions':>10}{'formula':>10}{'simulated':>11}")
for a, sd, N in [(0.03, 0.02, 5), (0.05, 0.02, 20), (0.10, 0.03, 20), (0.05, 0.02, 60)]:
    path = np.cumsum(rng.normal(0, sd / np.sqrt(26), size=(40000, N * 26)), axis=1)
    hit = (path.reshape(40000, N, 26).min(axis=2) <= -a).any(axis=1).mean()
    print(f"{a:9.1%}{sd:11.1%}{N:10d}{2*norm.cdf(-a/(sd*np.sqrt(N))):10.3f}{hit:11.3f}")

for lab, sub in [("GAP UP", ev[ev["up"]]), ("GAP DOWN", ev[~ev["up"]])]:
    print(f"\n{lab}   n = {len(sub):,}   median gap {sub['gap'].abs().median():.2%}"
          f"   filled same session {sub['fill1'].mean():.3f}")
    print(f"{'sessions':>9}{'filled':>9}{'RW, trailing vol':>19}{'RW, realised vol':>19}")
    for N in HOR:
        print(f"{N:9d}{sub[f'fill{N}'].mean():9.3f}"
              f"{rw(sub, N, 'sig_pre').mean():19.3f}{rw(sub, N, f'sig{N}').mean():19.3f}")

print("\nFilled within 20 sessions, by gap size")
print(f"{'':8}{'gap up':^31}   {'gap down':^31}")
print(f"{'size':>8}{'n':>7}{'filled':>8}{'trail':>8}{'real':>8}"
      f"   {'n':>7}{'filled':>8}{'trail':>8}{'real':>8}")
for b in ev["bucket"].cat.categories:
    u, dn = ev[ev["up"] & (ev["bucket"] == b)], ev[~ev["up"] & (ev["bucket"] == b)]
    print(f"{b:>8}{len(u):7d}{u['fill20'].mean():8.3f}{rw(u, 20, 'sig_pre').mean():8.3f}"
          f"{rw(u, 20, 'sig20').mean():8.3f}"
          f"   {len(dn):7d}{dn['fill20'].mean():8.3f}{rw(dn, 20, 'sig_pre').mean():8.3f}"
          f"{rw(dn, 20, 'sig20').mean():8.3f}")

print("\nFilled within 20 sessions, by year")
print(f"{'year':>6}{'n':>7}{'up':>8}{'RW':>7}{'down':>8}{'RW':>7}")
for y, sub in ev.groupby("year"):
    u, dn = sub[sub["up"]], sub[~sub["up"]]
    print(f"{y:6d}{len(sub):7d}{u['fill20'].mean():8.3f}{rw(u, 20, 'sig20').mean():7.3f}"
          f"{dn['fill20'].mean():8.3f}{rw(dn, 20, 'sig20').mean():7.3f}")

print("\nReturn from the gap open, net of the universe median over the same dates")
print(f"{'':10}{'20d median':>12}{'share > 0':>11}{'60d median':>12}{'share > 0':>11}")
for lab, sub in [("gap up", ev[ev["up"]]), ("gap down", ev[~ev["up"]])]:
    print(f"{lab:10}{sub['x20'].median():12.2%}{(sub['x20'] > 0).mean():11.3f}"
          f"{sub['x60'].median():12.2%}{(sub['x60'] > 0).mean():11.3f}")

# ── 5. chart ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
cats = list(ev["bucket"].cat.categories)
x = np.arange(len(cats))
for ax, (lab, sub) in zip(axes, [("Gap up", ev[ev["up"]]), ("Gap down", ev[~ev["up"]])]):
    obs = [sub.loc[sub["bucket"] == b, "fill20"].mean() for b in cats]
    ben = [rw(sub[sub["bucket"] == b], 20, "sig20").mean() for b in cats]
    ax.bar(x, obs, width=0.62, color="#3b82f6", label="Actually filled")
    ax.plot(x, ben, "o--", color="#f59e0b", markersize=7, linewidth=1.6,
            label="Random walk of the same volatility")
    for xi, a in enumerate(obs):
        ax.text(xi, a - 0.055, f"{a:.2f}", ha="center", color="#0a0a0a", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("Size of the opening gap")
    ax.set_title(lab, color="#e0e0e0")
    ax.set_ylim(0, 0.88)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Share filled within 20 sessions")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Gaps fill at the rate their own volatility implies", color="#e0e0e0")
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {SLUG}.png")
