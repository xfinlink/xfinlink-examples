# Full write-up: https://xfinlink.com/blog/how-many-stocks-to-diversify-python
"""How many stocks does it take to diversify?

Random equal-weighted portfolios drawn from the S&P 500, sized 1 to 100,
measured on annualised volatility and maximum drawdown. Reports the average
outcome and the spread of outcomes, and checks the simulated variance against
the closed form for a randomly drawn equal-weighted portfolio.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2021-08-04", "2026-08-03"
SIZES = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
DRAWS = 2000
SEED = 20260804
CHART = "how-many-stocks-to-diversify-python.png"

# ── universe ──────────────────────────────────────────────────────────
members = xfl.index("sp500").dropna(subset=["entity_id"])
ids = sorted(set(members["entity_id"].astype(int)))

px = pd.concat(
    [xfl.prices(entity_id=ids[i:i + 100], start=START, end=END,
                fields=["adj_close"], max_rows=200000)
     for i in range(0, len(ids), 100)],
    ignore_index=True,
)

# One column per continuously listed company: entities that traded under more
# than one symbol during the window are set aside, then names without a
# complete daily history, then names whose largest single-day change exceeds
# 100% (a corporate action rather than a price change).
single = px.groupby("entity_id")["ticker"].nunique() == 1
wide = (px[px["entity_id"].isin(single[single].index)]
        .pivot_table(index="date", columns="ticker", values="adj_close")
        .sort_index())
full = wide.dropna(axis=1)
ret = full.pct_change().dropna()
screened = ret.columns[ret.abs().max() >= 1.0]
ret = ret.drop(columns=screened)

R = ret.to_numpy()
n_names = R.shape[1]
assert not np.isnan(R).any()


def max_dd(r):
    """Deepest peak-to-trough fall of the cumulative return path."""
    curve = (1.0 + r).cumprod()
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


# ── simulation ────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
rows = []
for n in SIZES:
    vol, dd, var = [], [], []
    for _ in range(DRAWS):
        p = R[:, rng.choice(n_names, size=n, replace=False)].mean(axis=1)
        vol.append(p.std(ddof=1) * np.sqrt(252))
        var.append(p.var(ddof=1))
        dd.append(max_dd(p))
    vol, dd = np.array(vol), np.array(dd)
    rows.append({"n": n, "vol_mean": vol.mean(),
                 "vol_lucky": np.percentile(vol, 10),
                 "vol_unlucky": np.percentile(vol, 90),
                 "dd_mean": dd.mean(), "dd_lucky": np.percentile(dd, 90),
                 "dd_unlucky": np.percentile(dd, 10), "var_mean": np.mean(var)})
res = pd.DataFrame(rows)

panel = R.mean(axis=1)
vol_all, dd_all = panel.std(ddof=1) * np.sqrt(252), max_dd(panel)

# Closed form for a randomly drawn equal-weighted portfolio of n names:
# E[variance] = average variance / n + (1 - 1/n) * average covariance.
S = np.cov(R, rowvar=False, ddof=1)
avg_var = float(np.mean(np.diag(S)))
avg_cov = float(S[~np.eye(n_names, dtype=bool)].mean())
floor = np.sqrt(avg_cov * 252)

spy = xfl.prices("SPY", start=START, end=END, fields=["adj_close"])
spy_r = spy.set_index("date")["adj_close"].pct_change().dropna()

# ── output ────────────────────────────────────────────────────────────
print("S&P 500 members, daily returns from split-adjusted closes")
print("%s to %s (%s sessions)"
      % (ret.index.min().date(), ret.index.max().date(), f"{len(ret):,}"))
print("%d roster entities, %d under a single symbol across the window,"
      % (len(ids), int(single.sum())))
print("%d with a complete daily history, %d in the panel after the "
      "corporate-action screen\n" % (full.shape[1], n_names))
print("Equal-weighted portfolios rebalanced daily, %s random draws per size\n"
      % f"{DRAWS:,}")
print("            annualised volatility %        maximum drawdown %")
print("  size     mean   lucky   unlucky        mean   lucky   unlucky")
for _, r in res.iterrows():
    print("  %4d    %5.2f   %5.2f    %5.2f      %6.2f  %6.2f    %6.2f"
          % (r["n"], 100 * r["vol_mean"], 100 * r["vol_lucky"], 100 * r["vol_unlucky"],
             100 * r["dd_mean"], 100 * r["dd_lucky"], 100 * r["dd_unlucky"]))
print("  %4s    %5.2f                      %6.2f" % ("all", 100 * vol_all, 100 * dd_all))
print("\n  lucky = 10th percentile of the %s draws, unlucky = 90th" % f"{DRAWS:,}")
print("\nSystematic floor from average pairwise covariance   %5.2f%%" % (100 * floor))
print("SPY over the same window: volatility %.2f%%, maximum drawdown %.2f%%"
      % (100 * spy_r.std(ddof=1) * np.sqrt(252), 100 * max_dd(spy_r.to_numpy())))

print("\nSimulated against closed form, annualised volatility")
print("  size   simulated   closed form   difference bp")
for _, r in res.iterrows():
    n = int(r["n"])
    cf = np.sqrt((avg_var / n + (1 - 1 / n) * avg_cov) * 252)
    sim = np.sqrt(r["var_mean"] * 252)
    print("  %4d      %6.2f        %6.2f          %+5.1f"
          % (n, 100 * sim, 100 * cf, 1e4 * (sim - cf)))

# ── chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
x = res["n"]

ax1.fill_between(x, 100 * res["vol_lucky"], 100 * res["vol_unlucky"],
                 color="#3b82f6", alpha=0.22, lw=0, label="middle 80% of draws")
ax1.plot(x, 100 * res["vol_mean"], color="#3b82f6", lw=1.8, label="average draw")
ax1.axhline(100 * floor, color="#e0e0e0", ls="--", lw=1, label="systematic floor")
ax1.set_ylabel("Annualised volatility (%)")
ax1.set_title("Volatility", color="#e0e0e0")

ax2.fill_between(x, 100 * res["dd_unlucky"], 100 * res["dd_lucky"],
                 color="#f59e0b", alpha=0.22, lw=0, label="middle 80% of draws")
ax2.plot(x, 100 * res["dd_mean"], color="#f59e0b", lw=1.8, label="average draw")
ax2.axhline(100 * dd_all, color="#e0e0e0", ls="--", lw=1, label="whole panel")
ax2.set_ylabel("Maximum drawdown (%)")
ax2.set_title("Maximum drawdown", color="#e0e0e0")

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of stocks held")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.suptitle("How many stocks does it take to diversify? S&P 500, 2021-2026",
             color="#e0e0e0", fontsize=12)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
