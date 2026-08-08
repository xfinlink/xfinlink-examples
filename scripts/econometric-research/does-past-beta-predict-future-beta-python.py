# Full write-up: https://xfinlink.com/blog/does-past-beta-predict-future-beta-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF = "2015-12-31"
START, END = "2016-01-01", "2025-12-31"
WINDOWS = [(2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)]

# Membership as it stood at the end of 2015, not today's roster: the sample is
# fixed before any of the return data it is tested on.
members = xfl.index("sp500", as_of=AS_OF)
ids = sorted(members["entity_id"].dropna().astype(int).unique())
sample = ids[::3]              # systematic 1-in-3 sample, unrelated to beta
print(f"{len(ids)} members at {AS_OF}, systematic sample of {len(sample)}")

# Keyed on entity id, not ticker. A company that changed symbol inside the
# window (Alcoa to Arconic, Facebook to Meta) keeps one continuous series
# instead of splitting into two partial ones.
px = xfl.prices(entity_id=sample, start=START, end=END, interval="1w",
                fields=["close", "return_daily"], max_rows=200000)
spy = xfl.prices("SPY", start=START, end=END, interval="1w",
                 fields=["close", "return_daily"])
px["date"] = pd.to_datetime(px["date"])
spy["date"] = pd.to_datetime(spy["date"])
wk = px.pivot_table(index="date", columns="entity_id", values="return_daily")
mkt = spy.set_index("date")["return_daily"].reindex(wk.index)
labels = px.drop_duplicates("entity_id").set_index("entity_id")["ticker"]
print(f"{len(wk)} weeks, {wk.shape[1]} companies with a weekly record")
print()


def betas(y0, y1, min_obs=80):
    """OLS slope of each stock's weekly return on the market's, one window."""
    m = mkt[str(y0):str(y1)]
    out = {}
    for t in wk.columns:
        r = wk[t].loc[m.index]
        ok = r.notna() & m.notna()
        if ok.sum() >= min_obs:
            out[t] = np.polyfit(m[ok], r[ok], 1)[0]
    return pd.Series(out)


est = pd.DataFrame({f"{a}-{b}": betas(a, b) for a, b in WINDOWS})
print(f"{'window':12s} {'names':>6s} {'mean':>7s} {'median':>7s} {'sd':>7s} "
      f"{'min':>7s} {'max':>7s}")
for c in est.columns:
    s = est[c].dropna()
    print(f"{c:12s} {len(s):6d} {s.mean():7.2f} {s.median():7.2f} {s.std():7.2f} "
          f"{s.min():7.2f} {s.max():7.2f}")
print()

# How much of a window's beta survives into the next window?
pairs = []
for i in range(len(est.columns) - 1):
    a, b = est.columns[i], est.columns[i + 1]
    d = est[[a, b]].dropna()
    slope, icpt = np.polyfit(d[a], d[b], 1)
    pairs.append((a, b, len(d), d[a].corr(d[b]), slope, icpt))

print(f"{'from':12s} {'to':12s} {'n':>5s} {'corr':>7s} {'slope':>7s} {'intercept':>10s}")
for a, b, n, r, s, c in pairs:
    print(f"{a:12s} {b:12s} {n:5d} {r:7.3f} {s:7.3f} {c:10.3f}")
print()

pool = pd.concat([est[[a, b]].dropna().set_axis(["prior", "next"], axis=1)
                  for a, b, *_ in pairs], ignore_index=True)
slope, icpt = np.polyfit(pool["prior"], pool["next"], 1)
print(f"pooled: n={len(pool)}  corr={pool['prior'].corr(pool['next']):.3f}  "
      f"next = {icpt:.3f} + {slope:.3f} x prior")
print("Bloomberg's standard adjustment is a fixed 0.67 x prior + 0.33")
print()

# Does shrinking the raw estimate toward 1 forecast better than using it raw?
methods = {
    "raw prior beta":            lambda x: x,
    "Bloomberg 0.67b + 0.33":    lambda x: 0.67 * x + 0.33,
    "fitted on this sample":     lambda x: slope * x + icpt,
    "always 1.00":               lambda x: pd.Series(1.0, index=x.index),
}
print(f"{'forecast':26s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s}")
for name, f in methods.items():
    err = f(pool["prior"]) - pool["next"]
    print(f"{name:26s} {err.abs().mean():7.3f} {np.sqrt((err ** 2).mean()):7.3f} "
          f"{err.mean():7.3f}")
print()

# Where the raw estimate goes wrong: by starting quintile.
pool["q"] = pd.qcut(pool["prior"], 5, labels=False) + 1
print(f"{'quintile':9s} {'n':>5s} {'mean prior':>11s} {'mean next':>10s} {'drift':>8s}")
for q, g in pool.groupby("q"):
    print(f"{q:<9d} {len(g):5d} {g['prior'].mean():11.2f} {g['next'].mean():10.2f} "
          f"{g['next'].mean() - g['prior'].mean():+8.2f}")

# Chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f"})
fig, ax = plt.subplots(figsize=(10, 7))
lo = min(0.0, pool[["prior", "next"]].min().min() * 1.1)
lim = [lo, pool[["prior", "next"]].max().max() * 1.05]
ax.scatter(pool["prior"], pool["next"], s=14, alpha=0.45, color="#3b82f6",
           edgecolors="none", zorder=2)
ax.plot(lim, lim, color="#6b7280", lw=1, ls="--", zorder=3,
        label="beta repeats exactly")
xs = np.linspace(*lim, 50)
ax.plot(xs, icpt + slope * xs, color="#f59e0b", lw=2, zorder=4,
        label=f"fitted: {icpt:.2f} + {slope:.2f} x prior")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Beta estimated over the prior two years")
ax.set_ylabel("Beta realised over the following two years")
ax.set_title("Beta pulls back toward 1 in the window after it is measured\n"
             f"{len(pool)} two-year pairs, S&P 500 members as of 2015, weekly returns",
             color="#e0e0e0", fontsize=12)
ax.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0",
          fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("does-past-beta-predict-future-beta-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart written")
