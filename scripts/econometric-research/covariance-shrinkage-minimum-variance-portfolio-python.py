# Full write-up: https://xfinlink.com/blog/covariance-shrinkage-minimum-variance-portfolio-python
#
# Does covariance shrinkage beat the sample covariance matrix out of sample?
# Rolling minimum-variance backtest on the current Dow 30, calendar years 2021-2025,
# at two estimation-window lengths.
#
# pip install -U xfinlink numpy pandas scipy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2021-01-01", "2025-12-31"
STEP = 21           # rebalance and holding period, in trading days
OOS_START = 252     # common first out-of-sample day for every window length
ANN = np.sqrt(252)
NAMES = ["Equal weight", "Min-var, sample cov", "Min-var, shrunk cov", "Min-var, long-only"]

# ── Data ──────────────────────────────────────────────────────────────
tickers = sorted(xfl.index("djia")["ticker"].tolist())
px = xfl.prices(tickers, start=START, end=END, fields=["return_daily"])
R = px.pivot(index="date", columns="ticker", values="return_daily").sort_index().dropna()
dates = R.index
X_all = R.to_numpy()
N = X_all.shape[1]

print("=== SANITY ===")
print("tickers:", N, "| trading days:", len(R))
print("range:", dates[0].date(), "->", dates[-1].date(),
      "| monotonic:", dates.is_monotonic_increasing)
print("NaNs in matrix:", int(np.isnan(X_all).sum()))
print("min daily return: %.4f  max daily return: %.4f" % (X_all.min(), X_all.max()))
big = px[px["return_daily"].abs() > 0.20]
print("obs with |return| > 20%%: %d" % len(big))
for _, r in big.iterrows():
    print("   ", r["date"].date(), r["ticker"], round(r["return_daily"], 4))


# ── Estimators ────────────────────────────────────────────────────────
def ledoit_wolf(X):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity."""
    T, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / n
    d2 = np.sum((S - mu * np.eye(n)) ** 2) / n
    b_bar2 = (np.sum(np.einsum("ij,ij->i", Xc, Xc) ** 2) - T * np.sum(S ** 2)) / (n * T ** 2)
    intensity = float(np.clip(b_bar2 / d2, 0.0, 1.0))
    return intensity * mu * np.eye(n) + (1 - intensity) * S, intensity


def min_var_unconstrained(cov):
    """Analytic minimum-variance weights: shorting allowed, weights sum to 1."""
    inv1 = np.linalg.solve(cov, np.ones(len(cov)))
    return inv1 / inv1.sum()


def min_var_long_only(cov):
    """Minimum-variance weights subject to 0 <= w <= 1 and sum(w) = 1."""
    n = len(cov)
    res = minimize(
        lambda w: w @ cov @ w,
        np.repeat(1 / n, n),
        jac=lambda w: 2 * cov @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    return res.x


# ── Rolling out-of-sample backtest ────────────────────────────────────
def backtest(window):
    oos = {k: [] for k in NAMES}
    lev = {k: [] for k in NAMES}
    maxw = {k: [] for k in NAMES}
    shorts = {k: [] for k in NAMES}
    intensities, rebal_dates, wsum_dev, lo_min, cond = [], [], [], [], []

    for t in range(OOS_START, len(X_all) - STEP + 1, STEP):
        train, test = X_all[t - window:t], X_all[t:t + STEP]
        S = np.cov(train, rowvar=False)
        LW, intensity = ledoit_wolf(train)
        intensities.append(intensity)
        rebal_dates.append(dates[t])
        cond.append(np.linalg.cond(S))

        w = {
            "Equal weight": np.repeat(1 / N, N),
            "Min-var, sample cov": min_var_unconstrained(S),
            "Min-var, shrunk cov": min_var_unconstrained(LW),
            "Min-var, long-only": min_var_long_only(S),
        }
        wsum_dev.append(max(abs(wk.sum() - 1.0) for wk in w.values()))
        lo_min.append(w["Min-var, long-only"].min())
        for k, wk in w.items():
            oos[k].append(test @ wk)
            lev[k].append(np.abs(wk).sum())
            maxw[k].append(wk.max())
            shorts[k].append(int((wk < -1e-6).sum()))

    oos = {k: np.concatenate(v) for k, v in oos.items()}
    stats = {}
    for k in NAMES:
        r = oos[k]
        vol = r.std(ddof=1) * ANN
        ret = r.mean() * 252
        stats[k] = dict(vol=vol, ret=ret, sharpe=ret / vol, gross=np.mean(lev[k]),
                        maxw=np.mean(maxw[k]), shorts=np.mean(shorts[k]))
    return dict(stats=stats, lev=lev, oos=oos, dates=rebal_dates, intensity=intensities,
                wsum_dev=wsum_dev, lo_min=lo_min, cond=cond, n_days=len(oos[NAMES[0]]))


results = {w: backtest(w) for w in (252, 60)}

for w, res in results.items():
    print()
    print("=== %d-DAY ESTIMATION WINDOW: %d rebalances, %d out-of-sample days, %d assets ==="
          % (w, len(res["dates"]), res["n_days"], N))
    print("%-22s %8s %8s %8s %8s %8s %7s"
          % ("Portfolio", "Vol", "Return", "Sharpe", "Gross", "MaxWgt", "Shorts"))
    for k in NAMES:
        s = res["stats"][k]
        print("%-22s %7.2f%% %7.2f%% %8.2f %8.2f %7.1f%% %7.1f"
              % (k, s["vol"] * 100, s["ret"] * 100, s["sharpe"], s["gross"],
                 s["maxw"] * 100, s["shorts"]))
    print("Mean shrinkage intensity: %.3f   Vol change, shrunk vs sample: %+.1f%%"
          % (np.mean(res["intensity"]),
             (res["stats"]["Min-var, shrunk cov"]["vol"]
              / res["stats"]["Min-var, sample cov"]["vol"] - 1) * 100))

print()
print("=== CHECKS ===")
for w, res in results.items():
    print("window %3d | max |sum(w)-1| = %.1e | min long-only weight = %.1e | "
          "shrinkage %.3f-%.3f | max cond(S) = %.0f | NaNs = %d"
          % (w, max(res["wsum_dev"]), min(res["lo_min"]), min(res["intensity"]),
             max(res["intensity"]), max(res["cond"]),
             sum(int(np.isnan(v).sum()) for v in res["oos"].values())))

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = np.arange(len(NAMES))
v252 = [results[252]["stats"][k]["vol"] * 100 for k in NAMES]
v60 = [results[60]["stats"][k]["vol"] * 100 for k in NAMES]
b1 = ax1.bar(x - 0.19, v252, 0.36, color="#3b82f6", label="252-day estimation window")
b2 = ax1.bar(x + 0.19, v60, 0.36, color="#ef4444", label="60-day estimation window")
for bars, vals in ((b1, v252), (b2, v60)):
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.2, "%.1f" % v, ha="center", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(NAMES)
ax1.set_ylabel("Out-of-sample annualised volatility (%)")
ax1.set_title("Minimum-variance portfolios, Dow 30, 2021-2025")
ax1.set_ylim(0, max(v60 + v252) * 1.2)
ax1.legend(frameon=False, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

d60 = results[60]["dates"]
ax2.plot(d60, results[60]["lev"]["Min-var, sample cov"], color="#ef4444", lw=1.6,
         label="Sample covariance")
ax2.plot(d60, results[60]["lev"]["Min-var, shrunk cov"], color="#3b82f6", lw=1.6,
         label="Shrunk covariance")
ax2.axhline(1.0, color="#a3a3a3", lw=1.2, ls="--", label="Long-only / equal weight")
ax2.set_ylabel("Gross exposure (sum of |weights|)")
ax2.set_xlabel("Rebalance date, 60-day estimation window")
ax2.legend(frameon=False, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("covariance-shrinkage-minimum-variance-portfolio-python.png", dpi=150,
            facecolor="#0a0a0a")
print("chart written")
