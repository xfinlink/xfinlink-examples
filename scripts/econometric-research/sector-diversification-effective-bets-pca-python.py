# Full write-up: https://xfinlink.com/blog/sector-diversification-effective-bets-pca-python
"""How many independent bets does a nine-sector equity portfolio actually give you?

Eigenvalue entropy of the rolling correlation matrix of the nine original
Select Sector SPDR funds, 1999-2026.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
START, END, WINDOW = "1998-12-01", "2026-07-24", 252

# adj_close is split-adjusted and gap-free, so a share split cannot enter as a return
frames = [xfl.prices(t, start=START, end=END, fields=["adj_close"], max_rows=20000)
          for t in SECTORS]
px = pd.concat(frames).pivot(index="date", columns="ticker", values="adj_close").dropna()
rets = px.pct_change().dropna()

n = rets.shape[1]
R = rets.values
upper = np.triu_indices(n, 1)
rows = []
for i in range(WINDOW - 1, len(rets)):
    corr = np.corrcoef(R[i - WINDOW + 1:i + 1], rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    p = np.clip(eig / n, 1e-12, None)
    rows.append((rets.index[i],
                 float(np.exp(-(p * np.log(p)).sum())),   # effective number of bets
                 eig[0] / n,                              # share of variance in PC1
                 corr[upper].mean(),
                 eig.sum()))

res = pd.DataFrame(rows, columns=["date", "n_eff", "pc1", "avg_corr", "trace"]).set_index("date")

# realised diversification benefit: how much volatility the blend removes
sector_vol = rets.rolling(WINDOW).std().mean(axis=1)
port_vol = rets.mean(axis=1).rolling(WINDOW).std()
res["vol_cut"] = (1 - port_vol / sector_vol).reindex(res.index)
res["port_vol"] = (port_vol * np.sqrt(252)).reindex(res.index)

EPISODES = {
    "Dot-com unwind": "2002-09-30",
    "Global financial crisis": "2008-11-28",
    "Euro sovereign crisis": "2012-06-29",
    "COVID crash": "2020-03-31",
    "Inflation shock": "2022-09-30",
    "Latest window": str(res.index[-1].date()),
}

print(f"Nine Select Sector SPDR funds, daily price returns from split-adjusted closes")
print(f"Sample      : {rets.index[0].date()} to {rets.index[-1].date()} "
      f"({len(rets):,} trading days, {len(res):,} rolling windows)")
print(f"Window      : {WINDOW} trading days")
print(f"Check       : eigenvalues sum to {res['trace'].min():.6f} - {res['trace'].max():.6f} "
      f"(must equal {n})")

print("\nEFFECTIVE NUMBER OF INDEPENDENT BETS   (maximum possible = 9)")
print(f"{'':26s}{'window end':>12s}{'N':>7s}{'PC1 share':>12s}{'avg corr':>10s}"
      f"{'vol cut':>9s}{'port vol':>10s}")
for label, day in EPISODES.items():
    r = res.loc[:day].iloc[-1]
    print(f"{label:26s}{str(r.name.date()):>12s}{r['n_eff']:>7.2f}{r['pc1'] * 100:>11.1f}%"
          f"{r['avg_corr']:>10.2f}{r['vol_cut'] * 100:>8.1f}%{r['port_vol'] * 100:>9.1f}%")

print("\nWHAT A GIVEN N IS WORTH")
bins = pd.cut(res["n_eff"], [0, 2.5, 3, 3.5, 4, 4.5, 99],
              labels=["below 2.5", "2.5 - 3.0", "3.0 - 3.5", "3.5 - 4.0", "4.0 - 4.5", "above 4.5"])
grp = res.groupby(bins, observed=True).agg(windows=("n_eff", "size"),
                                           avg_corr=("avg_corr", "mean"),
                                           vol_cut=("vol_cut", "mean"))
print(f"{'N range':>12s}{'windows':>10s}{'avg corr':>11s}{'volatility removed':>21s}")
for label, r in grp.iterrows():
    print(f"{label:>12s}{int(r['windows']):>10,d}{r['avg_corr']:>11.2f}{r['vol_cut'] * 100:>20.1f}%")

def n_bets(frame):
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(frame.values, rowvar=False)))[::-1]
    q = np.clip(eig / n, 1e-12, None)
    return float(np.exp(-(q * np.log(q)).sum()))


# one shock month dominates a trailing window until it rolls out of it
shock = rets.loc["2025-04-01":"2025-04-30"]
w = rets.loc[:"2026-03-31"].tail(WINDOW)
print(f"\nWindow sensitivity: April 2025 on its own carries an average pairwise "
      f"correlation of {shock.corr().values[upper].mean():.2f} across {len(shock)} days.")
print(f"  252 days to 2026-03-31            N = {n_bets(w):.2f}")
print(f"  same window minus April 2025      N = {n_bets(w.drop(shock.index, errors='ignore')):.2f}")

lo, hi, last = res["n_eff"].idxmin(), res["n_eff"].idxmax(), res.iloc[-1]
print(f"\nFull-sample low  : {res['n_eff'].min():.2f} on {lo.date()}   "
      f"high: {res['n_eff'].max():.2f} on {hi.date()}   median: {res['n_eff'].median():.2f}")
print(f"Latest reading   : {last['n_eff']:.2f}, above "
      f"{(res['n_eff'] < last['n_eff']).mean() * 100:.1f}% of all windows since 1999")
print(f"N tracks the realised volatility reduction with a correlation of "
      f"{res['n_eff'].corr(res['vol_cut']):.3f}")

# ---- chart -------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(res.index, res["n_eff"], color="#3b82f6", lw=1.1)
ax1.axhline(res["n_eff"].median(), color="#6b7280", ls="--", lw=0.9)
ax1.text(res.index[40], res["n_eff"].median() + 0.12,
         f"median {res['n_eff'].median():.1f}", color="#9ca3af", fontsize=9)
ax1.set_ylim(1.4, 6.9)
ax1.annotate(f"{res['n_eff'].min():.2f}", xy=(lo, res["n_eff"].min()),
             xytext=(-38, -4), textcoords="offset points", color="#f87171", fontsize=9)
ax1.annotate(f"{last['n_eff']:.2f}", xy=(res.index[-1], last["n_eff"]),
             xytext=(-38, 6), textcoords="offset points", color="#e0e0e0", fontsize=9)
ax1.set_ylabel("Effective number of bets")
ax1.set_title("Sector diversification decay: nine sectors rarely behave like nine bets",
              color="#e0e0e0", fontsize=13, pad=10)

ax2.fill_between(res.index, res["vol_cut"] * 100, color="#3b82f6", alpha=0.28)
ax2.plot(res.index, res["vol_cut"] * 100, color="#3b82f6", lw=1.0)
ax2.set_ylabel("Volatility removed (%)")
ax2.set_xlabel("Window ending")
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("sector-diversification-effective-bets-pca-python.png", dpi=150,
            facecolor="#0a0a0a")
