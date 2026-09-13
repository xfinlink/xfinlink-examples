# Full write-up: https://xfinlink.com/blog/sector-beta-market-crash-python
"""Sector beta and correlation on the worst 10% of market days, 1999-2024.

Daily price returns for the nine Select Sector SPDRs and SPY. Each sector is
regressed on the market over the full sample, over the worst decile of market
days, and over the best decile, so that beta and correlation can be compared
across the same three buckets.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "1999-01-01", "2024-12-31"
SECTORS = {
    "XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary",
}

frames = [
    xfl.prices(t, start=START, end=END, fields=["adj_close"], max_rows=200000)
    for t in ["SPY"] + list(SECTORS)
]
px = (pd.concat(frames, ignore_index=True)
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index())
ret = px.pct_change().dropna()

mkt = ret["SPY"]
worst = mkt <= mkt.quantile(0.10)
best = mkt >= mkt.quantile(0.90)
allday = pd.Series(True, index=mkt.index)


def fit(mask):
    """OLS beta, its standard error, correlation and residual volatility."""
    out = {}
    for t in SECTORS:
        x, y = mkt[mask], ret[t][mask]
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = resid.std(ddof=2) / (np.sqrt(len(x)) * x.std(ddof=0))
        out[t] = (b, se, np.corrcoef(x, y)[0, 1], resid.std(ddof=2))
    return pd.DataFrame(out, index=["beta", "se", "corr", "resid"]).T


f_all, f_worst, f_best = fit(allday), fit(worst), fit(best)

W = 82
print("=" * W)
print("Sector beta and correlation on the worst 10% of market days")
print("=" * W)
print(f"Universe : SPY and the nine Select Sector SPDRs, daily price returns")
print(f"Window   : {ret.index.min():%Y-%m-%d} to {ret.index.max():%Y-%m-%d}"
      f"   ({len(ret):,} trading days)")
print(f"Buckets  : worst 10% = {int(worst.sum())} days with SPY at or below "
      f"{mkt.quantile(0.10) * 100:+.2f}%")
print(f"           best 10%  = {int(best.sum())} days with SPY at or above "
      f"{mkt.quantile(0.90) * 100:+.2f}%")
print(f"Market return: mean {mkt[worst].mean() * 100:+.2f}% and standard "
      f"deviation {mkt[worst].std() * 100:.2f}% inside the worst decile,")
print(f"               against {mkt.std() * 100:.2f}% across all days")
print()

print("Beta on the S&P 500 ETF")
print("-" * W)
print(f"{'Sector':<30}{'all days':>10}{'worst 10%':>11}{'s.e.':>8}"
      f"{'best 10%':>10}{'worst - all':>13}")
for t in sorted(SECTORS, key=lambda k: f_all.loc[k, "beta"] - f_worst.loc[k, "beta"]):
    print(f"{SECTORS[t] + ' (' + t + ')':<30}"
          f"{f_all.loc[t, 'beta']:>10.3f}{f_worst.loc[t, 'beta']:>11.3f}"
          f"{f_worst.loc[t, 'se']:>8.3f}{f_best.loc[t, 'beta']:>10.3f}"
          f"{f_worst.loc[t, 'beta'] - f_all.loc[t, 'beta']:>+13.3f}")
print(f"{'average':<30}{f_all['beta'].mean():>10.3f}{f_worst['beta'].mean():>11.3f}"
      f"{'':>8}{f_best['beta'].mean():>10.3f}"
      f"{f_worst['beta'].mean() - f_all['beta'].mean():>+13.3f}")
print()

print("Correlation with the S&P 500 ETF")
print("-" * W)
print(f"{'Sector':<30}{'all days':>10}{'worst 10%':>11}{'best 10%':>10}"
      f"{'worst - all':>13}")
for t in sorted(SECTORS, key=lambda k: f_all.loc[k, "beta"] - f_worst.loc[k, "beta"]):
    print(f"{SECTORS[t] + ' (' + t + ')':<30}"
          f"{f_all.loc[t, 'corr']:>10.3f}{f_worst.loc[t, 'corr']:>11.3f}"
          f"{f_best.loc[t, 'corr']:>10.3f}"
          f"{f_worst.loc[t, 'corr'] - f_all.loc[t, 'corr']:>+13.3f}")
print(f"{'average':<30}{f_all['corr'].mean():>10.3f}{f_worst['corr'].mean():>11.3f}"
      f"{f_best['corr'].mean():>10.3f}"
      f"{f_worst['corr'].mean() - f_all['corr'].mean():>+13.3f}")
print()

print("Why correlation falls while beta rises")
print("-" * W)
resid_ratio = (f_worst["resid"] / f_all["resid"]).mean()
pred = (f_all["beta"] * mkt[worst].std()) / np.sqrt(
    (f_all["beta"] * mkt[worst].std()) ** 2 + f_all["resid"] ** 2)
print(f"{'Sector-specific volatility in the worst decile, relative to all days':<74}"
      f"{resid_ratio:>7.2f}x")
print(f"{'Correlation implied by holding beta and sector-specific volatility fixed':<74}"
      f"{pred.mean():>8.3f}")
print(f"{'Correlation actually observed inside the worst decile':<74}"
      f"{f_worst['corr'].mean():>8.3f}")
print()

# Robustness: the same split with the two worst crisis stretches removed.
calm = ret.drop(ret.loc["2008-09-01":"2008-12-31"].index) \
          .drop(ret.loc["2020-02-01":"2020-04-30"].index)
cm = calm["SPY"]
cw = cm <= cm.quantile(0.10)
print("Beta with September-December 2008 and February-April 2020 removed")
print("-" * W)
print(f"{len(calm):,} trading days, {int(cw.sum())} of them in the worst decile")
print(f"{'Sector':<30}{'all days':>10}{'worst 10%':>11}{'worst - all':>13}")
for t in sorted(SECTORS, key=lambda k: f_all.loc[k, "beta"] - f_worst.loc[k, "beta"]):
    ba = np.polyfit(cm, calm[t], 1)[0]
    bw = np.polyfit(cm[cw], calm[t][cw], 1)[0]
    print(f"{SECTORS[t] + ' (' + t + ')':<30}{ba:>10.3f}{bw:>11.3f}{bw - ba:>+13.3f}")
print("=" * W)

# ── Chart ────────────────────────────────────────────────────────────
order = sorted(SECTORS, key=lambda k: f_worst.loc[k, "beta"] - f_all.loc[k, "beta"])
labels = [f"{SECTORS[t]} ({t})" for t in order]
y = np.arange(len(order))

plt.rcParams.update({"text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0"})
fig, axes = plt.subplots(1, 2, figsize=(10, 7), facecolor="#0a0a0a")
panels = [("beta", "Beta against the S&P 500 ETF"),
          ("corr", "Correlation with the S&P 500 ETF")]

for ax, (col, title) in zip(axes, panels):
    ax.set_facecolor("#0a0a0a")
    a = [f_all.loc[t, col] for t in order]
    w = [f_worst.loc[t, col] for t in order]
    ax.hlines(y, a, w, color="#404040", linewidth=2, zorder=1)
    ax.scatter(a, y, s=55, color="#8a8a8a", zorder=2, label="All days")
    ax.scatter(w, y, s=55, color="#3b82f6", zorder=3, label="Worst 10% of market days")
    ax.set_yticks(y)
    ax.set_yticklabels(labels if col == "beta" else [])
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylim(-0.6, len(order) - 0.2)
    if col == "beta":
        ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1, zorder=0)
    for s in ax.spines.values():
        s.set_color("#333333")

axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Sector beta rises in a falling market while sector correlation falls",
             fontsize=13, y=0.97)
plt.tight_layout()
plt.savefig("sector-beta-market-crash-python.png", dpi=150, facecolor="#0a0a0a")
