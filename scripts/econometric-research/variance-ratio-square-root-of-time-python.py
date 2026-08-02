# Full write-up: https://xfinlink.com/blog/variance-ratio-square-root-of-time-python
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ASSETS = {
    "SPY": "US large cap",
    "IWM": "US small cap",
    "EFA": "Developed ex-US",
    "TLT": "20y+ Treasuries",
    "XLE": "Energy sector",
    "XLU": "Utilities sector",
}
START, END = "2006-01-01", "2025-12-31"
HORIZONS = [2, 5, 10, 20, 60]

px = xfl.prices(list(ASSETS), start=START, end=END,
                fields=["adj_close", "return_daily"])

# ---------------------------------------------------------------- screening
px = px.drop_duplicates(["ticker", "date"]).dropna(subset=["return_daily"])
px = px[px["adj_close"] > 0]
dropped = int((px["return_daily"].abs() > 0.5).sum())
px = px[px["return_daily"].abs() <= 0.5]


def variance_ratio(x, q):
    """Lo-MacKinlay overlapping variance ratio with heteroskedasticity-robust z."""
    n = len(x)
    mu = x.mean()
    e = x - mu
    var_1 = (e ** 2).sum() / (n - 1)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    qsum = cum[q:] - cum[:-q]                       # overlapping q-day sums
    m = q * (n - q + 1) * (1 - q / n)
    var_q = ((qsum - q * mu) ** 2).sum() / m
    vr = var_q / var_1
    denom = ((e ** 2).sum()) ** 2
    theta = sum((((2 * (q - j)) / q) ** 2) * n * ((e[j:] ** 2) * (e[:-j] ** 2)).sum() / denom
                for j in range(1, q))
    z = np.sqrt(n) * (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))


rows, logret = [], {}
for t in ASSETS:
    r = px[px["ticker"] == t].sort_values("date")["return_daily"].to_numpy()
    x = np.log1p(r)
    logret[t] = x
    rec = {"ticker": t, "n": len(x), "ann_vol": x.std(ddof=1) * np.sqrt(252)}
    for q in HORIZONS:
        rec[f"VR{q}"], rec[f"z{q}"], rec[f"p{q}"] = variance_ratio(x, q)
    # non-overlapping 20-day check, independent of the estimator above
    k = len(x) // 20
    rec["sd20_actual"] = x[: k * 20].reshape(k, 20).sum(axis=1).std(ddof=1)
    rec["sd20_scaled"] = x.std(ddof=1) * np.sqrt(20)
    rec["blocks"] = k
    rows.append(rec)

res = pd.DataFrame(rows).set_index("ticker")

# ---------------------------------------------------------------- output
span = px["date"].agg(["min", "max"])
print(f"Variance ratio test on daily total returns, "
      f"{span['min']:%Y-%m-%d} to {span['max']:%Y-%m-%d}")
print("Overlapping estimator, heteroskedasticity-robust z (Lo and MacKinlay 1988)")
print("VR = 1 means variance grows linearly with time; * marks |z| > 1.96\n")
print(f"{'':6}{'obs':>6}{'ann vol':>9}", end="")
for q in HORIZONS:
    print(f"{('q=' + str(q)):>16}", end="")
print()
print(f"{'':21}", end="")
for _ in HORIZONS:
    print(f"{'VR':>8}{'z':>8}", end="")
print()
for t, r in res.iterrows():
    print(f"{t:<6}{int(r['n']):>6}{r['ann_vol']:>8.1%}", end="")
    for q in HORIZONS:
        star = "*" if abs(r[f"z{q}"]) > 1.96 else " "
        print(f"{r[f'VR{q}']:>8.3f}{r[f'z{q}']:>7.2f}{star}", end="")
    print()

print("\nWhat the square-root rule costs at a one-month horizon")
print(f"{'':6}{'':18}{'20-day sd':>11}{'sqrt(20) x daily':>18}{'error':>9}{'blocks':>8}")
for t, r in res.iterrows():
    err = r["sd20_actual"] / r["sd20_scaled"] - 1
    print(f"{t:<6}{ASSETS[t]:<18}{r['sd20_actual']:>10.2%}{r['sd20_scaled']:>17.2%}"
          f"{err:>+9.1%}{int(r['blocks']):>8}")

print(f"\nRows removed by the return-bound screen: {dropped}")
print("Implied one-month volatility from VR(20), annualised:")
for t, r in res.iterrows():
    naive = r["ann_vol"]
    print(f"  {t:<5}{naive:>7.1%} naive   {naive * np.sqrt(r['VR20']):>7.1%} VR-adjusted")

# ---------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
COLORS = {"SPY": "#3b82f6", "IWM": "#22d3ee", "EFA": "#a78bfa",
          "TLT": "#f59e0b", "XLE": "#ef4444", "XLU": "#10b981"}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.6, 1]})
ax1.axhline(1.0, color="#666666", lw=1.0, ls="--")
for t in ASSETS:
    vals = [res.loc[t, f"VR{q}"] for q in HORIZONS]
    ax1.plot(HORIZONS, vals, marker="o", ms=5, lw=1.8,
             color=COLORS[t], label=f"{t}  {ASSETS[t]}")
ax1.set_xscale("log")
ax1.set_xticks(HORIZONS)
ax1.set_xticklabels([str(q) for q in HORIZONS])
ax1.minorticks_off()
ax1.text(2.05, 1.008, "random walk", color="#9ca3af", fontsize=9)
ax1.set_xlabel("Holding period in trading days")
ax1.set_ylabel("Variance ratio")
ax1.set_title("Variance grows more slowly than time at every horizon tested")
ax1.legend(frameon=False, fontsize=9, ncol=2, loc="lower left")

err = (res["sd20_actual"] / res["sd20_scaled"] - 1) * 100
ax2.axhline(0, color="#666666", lw=1.0)
ax2.bar(list(res.index), err.values,
        color=[COLORS[t] for t in res.index], width=0.6)
for i, v in enumerate(err.values):
    ax2.text(i, v + (0.6 if v >= 0 else -1.4), f"{v:+.1f}%",
             ha="center", color="#e0e0e0", fontsize=9)
ax2.set_ylabel("Error in %")
ax2.set_xlabel("")
ax2.set_title("Square-root scaling versus measured one-month volatility",
              fontsize=10)
ax2.margins(y=0.25)
plt.tight_layout()
plt.savefig("variance-ratio-square-root-of-time-python.png", dpi=150)
