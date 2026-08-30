# Full write-up: https://xfinlink.com/blog/does-volume-predict-volatility-python
"""Does today's trading volume help predict tomorrow's absolute return?

Per-name regressions of tomorrow's absolute return on trailing volatility and on
detrended log volume, across the S&P 500 roster of 2019-01-02, 2019-2025, with a
2019-2022 fit and a 2023-2025 out-of-sample test.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2019-01-01", "2025-12-31"
VOL_WIN, TREND_WIN, MIN_OBS = 21, 60, 1200
SPLIT = pd.Timestamp("2023-01-01")

# 1. Point-in-time roster, carried by entity id.
roster = xfl.index("sp500", as_of="2019-01-02")
ids = sorted(roster["entity_id"].dropna().astype(int).unique().tolist())

# 2. Daily prices, 50 entity ids per call.
px = pd.concat(
    [xfl.prices(entity_id=ids[i:i + 50], start=START, end=END,
                fields=["close", "volume", "return_daily"], max_rows=200000)
     for i in range(0, len(ids), 50)],
    ignore_index=True,
)
px["date"] = pd.to_datetime(px["date"])

# 3. Per name: screen, build the predictors, run the two regressions.
rows, panel = [], []
for eid, g in px.groupby("entity_id"):
    g = g.sort_values("date")
    if (g["volume"].fillna(0) > 0).mean() < 0.95 or g["close"].median() < 1:
        continue
    g = g[(g["volume"] > 0) & g["return_daily"].notna()
          & (g["return_daily"].abs() <= 0.5)]
    if len(g) < MIN_OBS:
        continue
    a, lv = g["return_daily"].abs(), np.log(g["volume"])
    d = pd.DataFrame({
        "date": g["date"].values,
        "entity_id": eid,
        "y": a.shift(-1).values,                            # tomorrow's absolute return
        "vol": a.rolling(VOL_WIN).mean().values,            # trailing 21-session mean |return|
        "dv": (lv - lv.rolling(TREND_WIN).mean()).values,   # log volume less its 60-session average
    }).dropna()
    if len(d) < MIN_OBS:
        continue
    panel.append(d)

    y = d["y"].to_numpy()
    X0 = sm.add_constant(d[["vol"]].to_numpy())         # volatility only
    X1 = sm.add_constant(d[["vol", "dv"]].to_numpy())   # volatility plus volume
    m0 = sm.OLS(y, X0).fit()
    m1 = sm.OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    tr, te = (d["date"] < SPLIT).to_numpy(), (d["date"] >= SPLIT).to_numpy()
    f0, f1 = sm.OLS(y[tr], X0[tr]).fit(), sm.OLS(y[tr], X1[tr]).fit()
    sst = ((y[te] - y[tr].mean()) ** 2).sum()
    rows.append({
        "entity_id": eid, "n": len(d), "mean_y": y.mean(),
        "c_dv": m1.params[2], "t_dv": m1.tvalues[2],
        "r2_vol": m0.rsquared, "r2_both": m1.rsquared,
        "oos_vol": 1 - ((y[te] - f0.predict(X0[te])) ** 2).sum() / sst,
        "oos_both": 1 - ((y[te] - f1.predict(X1[te])) ** 2).sum() / sst,
    })

R = pd.DataFrame(rows)
P = pd.concat(panel, ignore_index=True)
R["oos_gain"] = R["oos_both"] - R["oos_vol"]

# 4. Descriptive: next-day absolute return by quintile of today's detrended volume.
P["q"] = P.groupby("entity_id")["dv"].transform(
    lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
P["yrel"] = P["y"] / P.groupby("entity_id")["y"].transform("mean")
quint = P.groupby("q").agg(dv=("dv", "mean"), yrel=("yrel", "mean"), n=("y", "size"))

print(f"{len(R)} names, {len(P):,} name-days, "
      f"{P['date'].min():%Y-%m-%d} to {P['date'].max():%Y-%m-%d}\n")

print("Next-day absolute return by quintile of today's detrended volume")
print(f"{'quintile':>9}{'volume vs own 60-day avg':>26}{'next-day |return|':>19}{'name-days':>12}")
for q, r in quint.iterrows():
    print(f"{int(q) + 1:>9}{np.exp(r['dv']):>25.2f}x{r['yrel']:>18.2f}x{int(r['n']):>12,}")

print("\nPer-name regression of tomorrow's |return| on trailing volatility and volume")
print(f"{'median t-statistic on detrended volume':<46}{R['t_dv'].median():>8.2f}")
print(f"{'  range of that t-statistic':<46}"
      f"{R['t_dv'].min():>8.2f} to {R['t_dv'].max():.2f}")
print(f"{'names with t above +1.96':<46}{(R['t_dv'] > 1.96).sum():>8d} of {len(R)}")
print(f"{'names with a negative volume coefficient':<46}{(R['c_dv'] < 0).sum():>8d} of {len(R)}")
print(f"{'median R-squared, volatility only':<46}{R['r2_vol'].median():>8.4f}")
print(f"{'median R-squared, volatility plus volume':<46}{R['r2_both'].median():>8.4f}")
print(f"{'median effect of a doubling in volume':<46}"
      f"{100 * R['c_dv'].median() * np.log(2):>8.2f} pp")
print(f"{'  against a mean absolute return of':<46}"
      f"{100 * R['mean_y'].median():>8.2f} pp")

print("\nOut-of-sample: coefficients fitted 2019-2022, tested 2023-2025")
print(f"{'median R-squared, volatility only':<46}{R['oos_vol'].median():>8.4f}")
print(f"{'median R-squared, volatility plus volume':<46}{R['oos_both'].median():>8.4f}")
print(f"{'median change in R-squared':<46}{R['oos_gain'].median():>8.4f}")
print(f"{'mean change in R-squared':<46}{R['oos_gain'].mean():>8.4f}")
print(f"{'names improved by adding volume':<46}{(R['oos_gain'] > 0).sum():>8d} of {len(R)}")
print(f"{'names worse by more than 1 R-squared point':<46}"
      f"{(R['oos_gain'] < -0.01).sum():>8d} of {len(R)}")

# 5. Chart.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 9.5,
})
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

ax1.bar(range(1, len(quint) + 1), quint["yrel"], color="#3b82f6", width=0.55)
ax1.axhline(1.0, color="#7a7a7a", lw=1, ls="--")
ax1.set_xticks(range(1, len(quint) + 1))
ax1.set_xticklabels([f"{np.exp(v):.2f}x" for v in quint["dv"]])
ax1.set_xlabel("Today's volume as a multiple of its own 60-day average")
ax1.set_ylabel("Next-day absolute return\n(multiple of the name's own average)")
ax1.set_title("Heavy volume today, larger move tomorrow",
              color="#e0e0e0", fontsize=11.5)
ax1.set_ylim(0.7, 1.45)
for q, v in zip(range(1, len(quint) + 1), quint["yrel"]):
    ax1.text(q, v + 0.02, f"{v:.2f}x", ha="center", color="#e0e0e0", fontsize=9)

ax2.hist(R["t_dv"], bins=36, color="#3b82f6")
ax2.axvline(1.96, color="#e0e0e0", lw=1, ls="--")
ax2.text(2.15, ax2.get_ylim()[1] * 0.92, "1.96", color="#e0e0e0", fontsize=8.5)
ax2.set_xlabel("t-statistic on volume, one regression per company")
ax2.set_ylabel("Companies")
ax2.set_title("In sample: volume matters almost everywhere",
              color="#e0e0e0", fontsize=10.5)

ax3.hist(R["oos_gain"] * 100, bins=36, color="#3b82f6")
ax3.axvline(0, color="#e0e0e0", lw=1, ls="--")
ax3.set_xlabel("Change in out-of-sample R-squared, percentage points")
ax3.set_ylabel("Companies")
ax3.set_title("Out of sample: two in five get worse",
              color="#e0e0e0", fontsize=10.5)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("does-volume-predict-volatility-python.png", dpi=150, facecolor="#0a0a0a")
