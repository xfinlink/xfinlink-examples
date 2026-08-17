# Full write-up: https://xfinlink.com/blog/amihud-illiquidity-sp500-python
"""Inside the S&P 500, does illiquidity pay, or does it only add risk?

Amihud (2002) illiquidity = average of |daily return| / daily dollar volume. Stocks
are sorted into quintiles each month on the trailing 12-month average and held for
one month, equal weighted.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LOOKBACK = 12

members = xfl.index("sp500")
ids = members["entity_id"].dropna().astype(int).tolist()
parts = []
for i in range(0, len(ids), 50):
    parts.append(xfl.prices(entity_id=ids[i:i + 50], start="2022-06-01", end="2025-12-31",
                            fields=["close", "volume", "return_daily"], max_rows=400000))
px = pd.concat(parts, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])

d = px.dropna(subset=["close", "volume", "return_daily"])
d = d[(d["volume"] > 0) & (d["close"] > 0)].copy()
d["dollar_vol"] = d["close"] * d["volume"]
d["illiq"] = d["return_daily"].abs() / d["dollar_vol"]
d["month"] = d["date"].dt.to_period("M")

monthly = d.groupby(["entity_id", "month"]).agg(
    illiq=("illiq", "mean"),
    dollar_vol=("dollar_vol", "median"),
    ret=("return_daily", lambda r: (1 + r).prod() - 1),
    n=("return_daily", "size")).reset_index()
monthly = monthly[monthly["n"] >= 15]

ill = monthly.pivot(index="month", columns="entity_id", values="illiq")
dv = monthly.pivot(index="month", columns="entity_id", values="dollar_vol")
ret = monthly.pivot(index="month", columns="entity_id", values="ret")

months, rows = list(ill.index), []
for k in range(LOOKBACK, len(months) - 1):
    form, hold = months[k], months[k + 1]
    frame = pd.DataFrame({
        "signal": ill.loc[months[k - LOOKBACK + 1]:form].mean(),
        "dv": dv.loc[months[k - LOOKBACK + 1]:form].median(),
        "fwd": ret.loc[hold]}).dropna()
    frame["q"] = pd.qcut(frame["signal"], 5, labels=False) + 1
    g = frame.groupby("q").agg(fwd=("fwd", "mean"), dv=("dv", "median"))
    rows.append(pd.DataFrame({"month": hold, "q": g.index, "fwd": g["fwd"].values, "dv": g["dv"].values}))
r = pd.concat(rows, ignore_index=True)

piv = r.pivot(index="month", columns="q", values="fwd")
ann_ret = (1 + piv.mean()) ** 12 - 1
ann_vol = piv.std(ddof=1) * np.sqrt(12)
dvm = r.groupby("q")["dv"].median() / 1e6

print(f"holding months: {len(piv)}  ({piv.index.min()} to {piv.index.max()})")
print("\nQuintile  median daily $ volume  annualised return  annualised volatility")
for q in piv.columns:
    print(f"   Q{q}         ${dvm[q]:8.1f}m            {ann_ret[q] * 100:6.2f}%             {ann_vol[q] * 100:6.2f}%")

spread = (piv[5] - piv[1]).dropna()
t, p = stats.ttest_1samp(spread, 0)
se = spread.std(ddof=1) / np.sqrt(len(spread))
lo, hi = spread.mean() - 1.96 * se, spread.mean() + 1.96 * se
print(f"\nQ5 minus Q1: {spread.mean() * 100:.3f}% a month, {((1 + spread.mean()) ** 12 - 1) * 100:.2f}% annualised")
print(f"  t = {t:.2f}, p = {p:.3f}, {len(spread)} months, {int((spread > 0).sum())} of {len(spread)} positive")
print(f"  95% confidence interval, monthly: {lo * 100:+.2f}% to {hi * 100:+.2f}%")

# how long a sample would be needed to call this effect real at the observed noise
need = (2 * spread.std(ddof=1) / spread.mean()) ** 2
print(f"  months needed for t=2 at this effect size and volatility: {need:.0f} ({need / 12:.1f} years)")

# With only five points a p-value here would be meaningless, so report the ordering itself.
steps_up = int((np.diff(ann_vol.values) > 0).sum())
print(f"\nvolatility rises at {steps_up} of 4 steps from Q1 to Q5"
      f"  (rank correlation with quintile = {stats.spearmanr(piv.columns, ann_vol.values)[0]:.2f})")

# ---- chart ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
qs = list(piv.columns)
err = [1.96 * piv[q].std(ddof=1) / np.sqrt(len(piv)) * np.sqrt(12) * 100 for q in qs]
axes[0].bar(qs, ann_ret * 100, width=0.62, color="#3b82f6", zorder=2)
axes[0].errorbar(qs, ann_ret * 100, yerr=err, fmt="none", ecolor="#c8c8c8", capsize=4, lw=1, zorder=3)
axes[0].set_title("Return: wide error bars, no clean order")
axes[0].set_ylabel("Annualised return (percent)")
axes[1].bar(qs, ann_vol * 100, width=0.62, color="#3b82f6", zorder=2)
axes[1].set_title("Risk: rises with illiquidity, every step")
axes[1].set_ylabel("Annualised volatility (percent)")
for ax in axes:
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{q}\n${dvm[q]:.0f}m" for q in qs])
    ax.set_xlabel("Illiquidity quintile, median daily dollar volume")
    ax.grid(axis="y", color="#1f1f1f", lw=0.7, zorder=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("Amihud illiquidity inside the S&P 500, 30 monthly holding periods to December 2025")
plt.tight_layout()
plt.savefig("amihud-illiquidity-sp500-python.png", dpi=150)
