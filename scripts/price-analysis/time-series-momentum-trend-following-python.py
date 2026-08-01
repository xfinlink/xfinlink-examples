# Full write-up: https://xfinlink.com/blog/time-series-momentum-trend-following-python
"""Time-series momentum across asset classes.

Signal  : sign of an asset's own trailing k-month total return, k in {1,3,6,9,12}.
Rule    : long the asset next month when the trailing return is positive,
          hold T-bills (BIL) otherwise. Rebalanced monthly.
Timing  : the signal uses data through month t-1 only; the position is held
          during month t. No look-ahead.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START = "2007-06-01"
CASH = "BIL"
LOOKBACKS = [1, 3, 6, 9, 12]
ASSETS = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC"]
LABELS = {
    "SPY": "US large cap", "IWM": "US small cap", "EFA": "Developed ex-US",
    "EEM": "Emerging markets", "VNQ": "REITs", "TLT": "Long Treasuries",
    "IEF": "7-10yr Treasuries", "LQD": "IG credit", "HYG": "High yield",
    "GLD": "Gold", "DBC": "Commodities",
}

# ---------------------------------------------------------------- data
px = pd.concat(
    [xfl.prices(t, start=START, fields=["return_daily"]) for t in ASSETS + [CASH]]
)
daily = px.pivot_table(index="date", columns="ticker", values="return_daily")
assert daily.isna().sum().sum() == 0, "incomplete daily panel"

monthly = (1 + daily).resample("ME").prod() - 1
monthly = monthly[monthly.index >= "2007-06-30"]
rf, R = monthly[CASH], monthly[ASSETS]

win = R.index >= R.index[max(LOOKBACKS)]      # common window for every lookback
rf_w = rf[win]
bh = R[win].mean(axis=1)                      # equal-weight buy-and-hold

# signal at month-end t; .shift(1) puts it to work during month t+1
signal = {
    k: (((1 + R).rolling(k).apply(np.prod, raw=True) - 1) > 0).astype(float)
    for k in LOOKBACKS
}


def stats_of(r, rfs):
    """Annualised return, annualised volatility, Sharpe, max drawdown."""
    n = len(r)
    ann = (1 + r).prod() ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    ex = r - rfs
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    eq = (1 + r).cumprod()
    return ann, vol, sharpe, (eq / eq.cummax() - 1).min()


def trend_portfolio(k, mask):
    """Equal-weight long/flat composite for lookback k over `mask` months."""
    s = signal[k].shift(1)[mask]
    return (R[mask] * s + rf[mask].values[:, None] * (1 - s)).mean(axis=1), s


def jobson_korkie(a, b, rfs):
    """Memmel-corrected test of the difference between two Sharpe ratios."""
    x, y = (a - rfs).values, (b - rfs).values
    n = len(x)
    s1, s2 = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    v = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / n
    z = (s1 - s2) / np.sqrt(v)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


print(f"Sample        : {monthly.index.min().date()} to {monthly.index.max().date()} "
      f"monthly, {len(ASSETS)} assets, cash leg {CASH}")
print(f"Backtest      : {R.index[win][0].date()} to {R.index[win][-1].date()}  "
      f"({win.sum()} months)")

# ------------------------------------------- 1. is the signal predictive?
print("\nNext-month excess return, sorted on the sign of the trailing k-month return")
print(f"{'lookback':>9}{'obs +':>8}{'obs -':>8}{'mean +':>9}{'mean -':>9}"
      f"{'spread':>9}{'t':>7}{'p':>7}{'hit +':>8}{'hit -':>8}")
excess = R[win].sub(rf_w, axis=0)
for k in LOOKBACKS:
    s = signal[k].shift(1)[win]
    # monthly cross-sectional means; months where every asset shares one sign
    # carry no spread and drop out of both legs and the test alike
    up = excess.where(s == 1).mean(axis=1)
    dn = excess.where(s == 0).mean(axis=1)
    both = up.notna() & dn.notna()
    diff = (up - dn)[both]
    t, p = stats.ttest_1samp(diff, 0)
    n_up, n_dn = int((s == 1).sum().sum()), int((s == 0).sum().sum())
    print(f"{k:>8}m{n_up:>8}{n_dn:>8}{up[both].mean()*100:>+8.3f}%"
          f"{dn[both].mean()*100:>+8.3f}%{diff.mean()*100:>+8.3f}%"
          f"{t:>+7.2f}{p:>7.3f}"
          f"{100*((excess>0)&(s==1)).sum().sum()/n_up:>7.1f}%"
          f"{100*((excess>0)&(s==0)).sum().sum()/n_dn:>7.1f}%")
print(f"{'':>9}unconditional mean {excess.stack().mean()*100:+.3f}% per month, "
      f"hit rate {100*(excess.stack()>0).mean():.1f}%")

# ---------------------------------- 2. composite trend rule vs buy-and-hold
print("\nEqual-weight composite: long/flat trend rule vs buy-and-hold")
print(f"{'':<14}{'return':>8}{'vol':>8}{'Sharpe':>8}{'max DD':>9}"
      f"{'switches/yr':>13}{'JK z':>7}{'p':>7}")
a, v, sh, dd = stats_of(bh, rf_w)
print(f"{'buy-and-hold':<14}{a*100:>7.2f}%{v*100:>7.2f}%{sh:>8.2f}{dd*100:>8.2f}%"
      f"{0.0:>13.2f}")
comp = {}
for k in LOOKBACKS:
    port, s = trend_portfolio(k, win)
    comp[k] = port
    a, v, sh, dd = stats_of(port, rf_w)
    sw = s.diff().abs().sum(axis=1).mean() / len(ASSETS) * 12
    z, p = jobson_korkie(port, bh, rf_w)
    print(f"{'trend ' + str(k) + 'm':<14}{a*100:>7.2f}%{v*100:>7.2f}%{sh:>8.2f}"
          f"{dd*100:>8.2f}%{sw:>13.2f}{z:>+7.2f}{p:>7.3f}")

# ------------------------------------------------- 3. per asset, 12m rule
print("\nPer asset, 12-month rule")
print(f"{'':<22}{'B&H ret':>9}{'B&H Sh':>8}{'B&H DD':>9}"
      f"{'trend ret':>11}{'trend Sh':>10}{'trend DD':>10}{'invested':>10}")
s12 = signal[12].shift(1)[win]
for t in ASSETS:
    r_bh = R[win][t]
    r_tr = R[win][t] * s12[t] + rf_w * (1 - s12[t])
    a1, _, sh1, dd1 = stats_of(r_bh, rf_w)
    a2, _, sh2, dd2 = stats_of(r_tr, rf_w)
    print(f"{t + '  ' + LABELS[t]:<22}{a1*100:>8.2f}%{sh1:>8.2f}{dd1*100:>8.2f}%"
          f"{a2*100:>10.2f}%{sh2:>10.2f}{dd2*100:>9.2f}%{s12[t].mean()*100:>9.1f}%")

# --------------------------------------------------- 4. where it came from
crisis = win & (R.index <= "2009-12-31")
calm = win & (R.index >= "2010-01-01")
print(f"\nCrisis window {R.index[crisis][0].date()} to {R.index[crisis][-1].date()}: "
      f"buy-and-hold {((1+R[crisis].mean(axis=1)).prod()-1)*100:+.2f}%, "
      + ", ".join(f"trend {k}m {((1+trend_portfolio(k, crisis)[0]).prod()-1)*100:+.2f}%"
                  for k in LOOKBACKS))
a, v, sh, dd = stats_of(R[calm].mean(axis=1), rf[calm])
print(f"From 2010-01 onward ({calm.sum()} months): buy-and-hold Sharpe {sh:.2f}, "
      f"max DD {dd*100:.2f}%; "
      + ", ".join(f"trend {k}m Sharpe {stats_of(trend_portfolio(k, calm)[0], rf[calm])[2]:.2f}"
                  for k in LOOKBACKS))

# ------------------------------------------------------------------ chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.5, 1]})

for series, label, colour in [(bh, "Buy and hold, equal weight", "#9ca3af"),
                              (comp[12], "12-month trend rule", "#3b82f6")]:
    curve = (1 + series).cumprod()
    curve.loc[R.index[win][0] - pd.offsets.MonthEnd(1)] = 1.0
    ax1.plot(curve.sort_index().index, curve.sort_index().values,
             color=colour, linewidth=1.6, label=label)
ax1.set_yscale("log")
ax1.set_yticks([0.7, 1, 1.5, 2, 3])
ax1.set_yticklabels(["$0.70", "$1.00", "$1.50", "$2.00", "$3.00"])
ax1.set_ylabel("Value of $1 invested")
ax1.set_title("Time-series momentum across 11 asset classes, 2008-2026", pad=12)
ax1.legend(frameon=False, loc="upper left")
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

sharpes = [stats_of(comp[k], rf_w)[2] for k in LOOKBACKS]
ax2.bar([str(k) + "m" for k in LOOKBACKS], sharpes, color="#3b82f6", width=0.55)
ax2.axhline(stats_of(bh, rf_w)[2], color="#9ca3af", linestyle="--", linewidth=1.2,
            label="Buy and hold, equal weight")
for i, s in enumerate(sharpes):
    ax2.text(i, s + 0.02, f"{s:.2f}", ha="center", color="#e0e0e0", fontsize=9)
ax2.set_ylim(0, max(sharpes) * 1.32)
ax2.legend(frameon=False, loc="upper left", fontsize=9)
ax2.set_ylabel("Sharpe ratio")
ax2.set_xlabel("Length of the trailing return used as the signal")
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("time-series-momentum-trend-following-python.png", dpi=150)
