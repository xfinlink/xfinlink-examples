# Full write-up: https://xfinlink.com/blog/volatility-forecast-har-vs-ewma-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "AAPL", "MSFT", "JPM", "KO"]
START, END = "2005-07-01", "2026-06-30"   # 2005 is warm-up for the 63-day window
OOS_START = "2016-01-01"
H = 21                                    # forecast horizon, trading days
LAM = 0.94                                # RiskMetrics decay

# ── Data ─────────────────────────────────────────────────────────────
# One call per ticker: a single wide multi-decade request is a large
# paginated fan-out, and per-ticker calls keep each response small.
px = pd.concat([xfl.prices(t, start=START, end=END, fields=["adj_close"])
                for t in TICKERS], ignore_index=True)


def panel(ticker):
    """Realised-volatility features and the forward target for one name."""
    s = px[px["ticker"] == ticker].sort_values("date").set_index("date")
    sq = np.log(s["adj_close"]).diff().pow(2)          # squared log returns

    def av(window):                                    # annualised realised vol
        return np.sqrt(252.0 * sq.rolling(window).mean())

    # forward mean of squared returns over days t+1 .. t+H
    fwd = sq[::-1].rolling(H).mean()[::-1].shift(-1)

    return pd.DataFrame({
        "v5": av(5), "v21": av(21), "v63": av(63),
        "ewma": np.sqrt(252.0 * sq.ewm(alpha=1 - LAM, adjust=False).mean()),
        "y": np.sqrt(252.0 * fwd),
    }).dropna()


def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


COLS = ["v5", "v21", "v63"]
rows, paths = [], {}

for t in TICKERS:
    d = panel(t)
    # evaluate every H-th day so the test windows never overlap
    test = np.flatnonzero(d.index >= OOS_START)[::H]
    recs = []
    for i in test:
        train = d.iloc[:i - H]        # last training target ends before day i
        b = fit(train[COLS].values, train["y"].values)
        recs.append({"date": d.index[i], "actual": d.iloc[i]["y"],
                     "har": float(b[0] + b[1:] @ d.iloc[i][COLS].values),
                     "rw": d.iloc[i]["v21"], "ewma": d.iloc[i]["ewma"],
                     "mean": train["y"].mean()})
    e = pd.DataFrame(recs).set_index("date")
    paths[t] = e
    rmse = {m: np.sqrt(((e[m] - e["actual"]) ** 2).mean())
            for m in ["har", "rw", "ewma", "mean"]}
    rows.append({"ticker": t, "avg_vol": e["actual"].mean(), **rmse,
                 "r2_vs_rw": 1 - (rmse["har"] / rmse["rw"]) ** 2, "beta": b})

res = pd.DataFrame(rows)
allf = pd.concat(paths.values())

print(f"Out-of-sample monthly volatility forecasts, {allf.index.min().date()} "
      f"to {allf.index.max().date()}  ({len(allf)} non-overlapping windows)\n")
print("           mean vol   RMSE, annualised volatility points      HAR gain")
print("                        HAR    EWMA      RW    mean   vs RW")
for _, r in res.iterrows():
    print(f"{r.ticker:>6}  {r.avg_vol*100:9.1f}  {r.har*100:6.2f}  {r.ewma*100:6.2f}"
          f"  {r.rw*100:6.2f}  {r['mean']*100:6.2f}  {r.r2_vs_rw*100:6.1f}%")
pooled = {m: np.sqrt(((allf[m] - allf["actual"]) ** 2).mean())
          for m in ["har", "rw", "ewma", "mean"]}
print(f"{'pooled':>6}  {allf.actual.mean()*100:9.1f}  {pooled['har']*100:6.2f}"
      f"  {pooled['ewma']*100:6.2f}  {pooled['rw']*100:6.2f}  {pooled['mean']*100:6.2f}"
      f"  {(1-(pooled['har']/pooled['rw'])**2)*100:6.1f}%")

print("\nMean forecast error (forecast minus realised, volatility points)")
calm = allf["rw"] <= allf["rw"].median()
for label, g in [("calm months    ", allf[calm]), ("stressed months", allf[~calm])]:
    print(f"  {label}  n={len(g):3d}   HAR {((g.har-g.actual).mean())*100:+6.2f}"
          f"   EWMA {((g.ewma-g.actual).mean())*100:+6.2f}"
          f"   RW {((g.rw-g.actual).mean())*100:+6.2f}")

print("\nFinal HAR coefficients (intercept, week, month, quarter)")
for _, r in res.iterrows():
    print(f"  {r.ticker:>6}  " + "  ".join(f"{x:6.3f}" for x in r.beta)
          + f"   slope sum {r.beta[1:].sum():.3f}")

print("\nSPY through the 2020 shock")
print(f"  {'date':>10}  {'realised':>9}  {'HAR':>7}  {'RW':>7}")
for dt, r in paths["SPY"].loc["2020-01-01":"2020-06-30"].iterrows():
    print(f"  {dt.date()}  {r.actual*100:8.1f}%  {r.har*100:6.1f}%  {r.rw*100:6.1f}%")

# ── Chart ────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

spy = paths["SPY"]
ax1.plot(spy.index, spy["actual"] * 100, color="#e0e0e0", lw=1.6, label="Realised")
ax1.plot(spy.index, spy["har"] * 100, color="#3b82f6", lw=1.6, label="HAR forecast")
ax1.plot(spy.index, spy["rw"] * 100, color="#f59e0b", lw=1.2, alpha=0.85,
         label="Last month carried forward")
ax1.set_title("S&P 500 ETF: volatility over the following month, forecast and realised")
ax1.set_ylabel("Annualised volatility (%)")
ax1.legend(frameon=False, loc="upper left")
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

x = np.arange(len(res))
w = 0.27
ax2.bar(x - w, res["rw"] * 100, w, color="#f59e0b", label="Last month carried forward")
ax2.bar(x, res["ewma"] * 100, w, color="#6b7280", label="EWMA")
ax2.bar(x + w, res["har"] * 100, w, color="#3b82f6", label="HAR")
ax2.set_xticks(x, res["ticker"])
ax2.set_ylabel("Forecast error, RMSE\n(volatility points)")
ax2.set_title("Out-of-sample error by name, 2016 to 2026")
ax2.set_ylim(0, 18)
ax2.legend(frameon=False, ncol=3, loc="upper left")
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout(h_pad=2.5)
plt.savefig("volatility-forecast-har-vs-ewma-python.png", dpi=150,
            facecolor="#0a0a0a")
