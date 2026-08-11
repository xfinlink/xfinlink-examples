# Full write-up: https://xfinlink.com/blog/how-many-years-of-data-to-estimate-expected-return-python
"""How long does a price history have to be before its average return means anything?

The standard error of a mean return falls with the square root of calendar time, and
under independent returns sampling frequency does not enter it at all. This measures
that error on twelve exchange-traded funds, converts it into the history each one
would need for a two-standard-error result, and contrasts it with the error on the
volatility estimate, which does fall with the observation count.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "XLK", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ", "XLU", "XLP"]
NAMES = {
    "SPY": "US large cap", "XLK": "Technology", "IWM": "US small cap",
    "EFA": "Developed ex-US", "EEM": "Emerging markets", "TLT": "Long Treasuries",
    "IEF": "Mid Treasuries", "LQD": "IG corporates", "GLD": "Gold",
    "VNQ": "US REITs", "XLU": "Utilities", "XLP": "Consumer staples",
}
START, END = "1990-01-01", "2026-08-07"
TD = 252                      # trading days per year
CHART = "how-many-years-of-data-to-estimate-expected-return-python.png"

# ── 1. Daily total returns, longest history each fund has ─────────────

px = xfl.prices(TICKERS, start=START, end=END,
                fields=["return_daily"], max_rows=200000)
px = (px.dropna(subset=["return_daily"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True))

# Each fund contributes its longest unbroken run: any break of more than a month
# starts a new block, and only the block running to the present is kept.
block = (px.groupby("ticker")["date"].diff().dt.days > 31).cumsum()
px = px[block == block.groupby(px["ticker"]).transform("last")]

# ── 2. Mean, volatility, and the error attached to each ───────────────

rows = []
for tk in TICKERS:
    r = px.loc[px["ticker"] == tk, "return_daily"].to_numpy()
    n = len(r)
    years = n / TD
    mean = r.mean() * TD                       # annualised arithmetic mean
    vol = r.std(ddof=1) * np.sqrt(TD)          # annualised volatility
    se_mean = vol / np.sqrt(years)             # sigma / sqrt(calendar time)
    se_vol = vol / np.sqrt(2 * n)              # falls with the observation count
    sharpe = mean / vol
    rows.append({
        "ticker": tk, "label": NAMES[tk], "years": years,
        "mean": mean * 100, "vol": vol * 100,
        "se_mean": se_mean * 100, "se_vol": se_vol * 100,
        "lo": (mean - 1.96 * se_mean) * 100, "hi": (mean + 1.96 * se_mean) * 100,
        "t": mean / se_mean, "need": 4.0 / sharpe ** 2,
    })

tab = pd.DataFrame(rows).sort_values("need").reset_index(drop=True)

# ── 3. Does sampling more often help? Same history, three frequencies ──

spy = px[px["ticker"] == "SPY"].set_index("date")["return_daily"]
freq = []
for label, rule, per in [("daily", None, TD), ("weekly", "W-FRI", 52), ("monthly", "ME", 12)]:
    s = spy if rule is None else (1.0 + spy).resample(rule).prod().dropna() - 1.0
    m, v = s.mean() * per, s.std(ddof=1) * np.sqrt(per)
    freq.append({
        "freq": label, "obs": len(s), "mean": m * 100, "vol": v * 100,
        "se_mean": v / np.sqrt(len(s) / per) * 100,
        "se_vol": v / np.sqrt(2 * len(s)) * 100,
    })
freq = pd.DataFrame(freq)

# ── 4. What the estimate looked like as SPY's history accumulated ─────

r_spy = spy.to_numpy()
step = np.arange(2 * TD, len(r_spy) + 1, 21)             # monthly snapshots from year 2
yrs_path = step / TD
mean_path = np.array([r_spy[:i].mean() * TD for i in step]) * 100
band = np.array([r_spy[:i].std(ddof=1) * np.sqrt(TD) / np.sqrt(i / TD)
                 for i in step]) * 100 * 1.96

# ── 5. Report ─────────────────────────────────────────────────────────

print(f"Daily total returns, {px['date'].min().date()} to {px['date'].max().date()}\n")
print(f"{'':22}{'yrs':>5}{'mean':>8}{'vol':>7}{'se':>7}{'95% interval':>18}{'t':>6}{'yrs for t=2':>13}")
for _, r in tab.iterrows():
    print(f"{r['ticker'] + ' ' + r['label']:22}{r['years']:>5.1f}{r['mean']:>7.2f}%"
          f"{r['vol']:>6.1f}%{r['se_mean']:>6.2f}%"
          f"{f'{r.lo:.1f}% to {r.hi:.1f}%':>18}{r['t']:>6.2f}{r['need']:>12.0f}")

print(f"\nFunds whose mean return clears two standard errors: "
      f"{int((tab['t'].abs() >= 2).sum())} of {len(tab)}")

print("\nSPY, one history sampled three ways")
print(f"{'':10}{'obs':>7}{'mean':>8}{'vol':>7}{'se(mean)':>10}{'se(vol)':>9}{'t':>7}")
for _, f in freq.iterrows():
    print(f"{f['freq']:10}{f['obs']:>7.0f}{f['mean']:>7.2f}%{f['vol']:>6.1f}%"
          f"{f['se_mean']:>9.2f}%{f['se_vol']:>8.2f}%{f['mean'] / f['se_mean']:>7.2f}")

spy_row = tab[tab["ticker"] == "SPY"].iloc[0]
print(f"\nSPY after {spy_row['years']:.1f} years: mean {spy_row['mean']:.2f}% "
      f"+/- {1.96 * spy_row['se_mean']:.2f}% at 95% confidence")
print(f"Volatility estimate: {spy_row['vol']:.1f}% "
      f"+/- {1.96 * spy_row['se_vol']:.2f}% on the same data")
print(f"A three-year record at that volatility carries a standard error of "
      f"{spy_row['vol'] / np.sqrt(3):.1f}% a year")

# ── 6. Chart ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.fill_between(yrs_path, mean_path - band, mean_path + band,
                 color="#3b82f6", alpha=0.22, label="95% confidence interval")
ax1.plot(yrs_path, mean_path, color="#3b82f6", lw=1.6, label="Estimate from data so far")
ax1.axhline(0, color="#888888", lw=0.9)
ax1.set_xlim(2, yrs_path[-1])
ax1.set_xlabel("Years of history used")
ax1.set_ylabel("Average annual return (%)")
ax1.set_title("The S&P 500 average return, and how sure the data lets anyone be about it")
ax1.legend(frameon=False, loc="upper right")

pos = np.arange(len(tab))
ax2.hlines(pos, tab["years"], tab["need"], color="#3f3f46", lw=2.0, zorder=1)
ax2.scatter(tab["years"], pos, s=46, color="#3b82f6", zorder=2, label="History available")
ax2.scatter(tab["need"], pos, s=46, color="#9ca3af", zorder=2, label="History needed")
ax2.set_yticks(pos)
ax2.set_yticklabels(tab["ticker"] + "  " + tab["label"], fontsize=9)
ax2.invert_yaxis()
ax2.set_xscale("log")
ax2.set_xlim(min(tab["need"].min(), tab["years"].min()) * 0.6,
             max(tab["need"].max(), tab["years"].max()) * 5.0)
ax2.set_xlabel("Years of daily data (log scale)")
ax2.set_title("History needed before an average return separates from zero")
ax2.legend(frameon=False, loc="upper right", ncol=2)

for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

plt.tight_layout(h_pad=2.0)
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nchart saved to {CHART}")
