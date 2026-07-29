# Full write-up: https://xfinlink.com/blog/turn-of-the-month-first-trading-day-python
"""Where inside the turn-of-the-month window does the return actually sit?

Builds a trading-day-of-month return profile for SPY (1996-2024) and tests the
classic -1 to +3 window against a first-trading-day-only rule across 7 ETFs.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "MDY", "IWM", "EFA", "EEM", "TLT", "LQD"]
START, END = "1996-01-01", "2024-12-31"


def load(ticker):
    """Daily total returns, indexed by trading-day position within the month."""
    d = xfl.prices(
        ticker, start=START, end=END, fields=["close", "adj_close", "dividend"]
    ).sort_values("date")
    factor = d["adj_close"] / d["close"]
    d["ret"] = (
        d["adj_close"] + d["dividend"].fillna(0.0) * factor
    ) / d["adj_close"].shift(1) - 1

    month = d["date"].dt.to_period("M")
    d["fwd"] = d.groupby(month).cumcount() + 1          # 1 = first session of month
    d["bwd"] = d.groupby(month).cumcount(ascending=False) + 1  # 1 = last session
    d["pos"] = np.where(d["bwd"] <= 5, -d["bwd"], d["fwd"])
    d["classic"] = (d["bwd"] == 1) | (d["fwd"] <= 3)    # Lakonishok-Smidt window

    first_full_month = (month.min() + 1).to_timestamp()
    return d[d["date"] >= first_full_month].dropna(subset=["ret"]).reset_index(drop=True)


panel = {t: load(t) for t in TICKERS}
spy = panel["SPY"]

# 1. SPY return profile by trading-day position within the month
prof = spy[spy["pos"].between(-5, 10)].groupby("pos")["ret"].agg(["mean", "std", "count"])
prof["bps"] = prof["mean"] * 1e4
prof["t"] = prof["mean"] / (prof["std"] / np.sqrt(prof["count"]))
print("SPY mean total return by trading day of month, 1996-2024")
print("  pos  months   bps   t-stat")
for pos, r in prof.iterrows():
    print(f"  {pos:+3d}   {int(r['count']):4d}  {r['bps']:6.2f}   {r['t']:5.2f}")

# 2. Classic -1..+3 window vs the rest of the month, by sub-period
print("\nSPY classic turn-of-month window (-1 to +3) vs rest of month")
periods = [("1996-02-01", "2005-12-31"), ("2006-01-01", "2015-12-31"),
           ("2016-01-01", "2024-12-31")]
for lo, hi in periods:
    s = spy[(spy["date"] >= lo) & (spy["date"] <= hi)]
    win, rest = s.loc[s["classic"], "ret"], s.loc[~s["classic"], "ret"]
    tt = stats.ttest_ind(win, rest, equal_var=False)
    print(f"  {lo[:4]}-{hi[:4]}  window {win.mean()*1e4:5.2f} bps   "
          f"rest {rest.mean()*1e4:4.2f} bps   diff {(win.mean()-rest.mean())*1e4:5.2f}   "
          f"t {tt.statistic:5.2f}   p {tt.pvalue:.3f}")

# 3. First trading day of the month vs every other day, across asset classes
print("\nFirst trading day of month vs all other days")
print("  ticker  from        months   day+1     other      diff   t-stat   p      hit")
first_day = {}
for t in TICKERS:
    d = panel[t]
    a, b = d.loc[d["pos"] == 1, "ret"], d.loc[d["pos"] != 1, "ret"]
    tt = stats.ttest_ind(a, b, equal_var=False)
    first_day[t] = (a.mean() * 1e4, b.mean() * 1e4)
    print(f"  {t:6s}  {d['date'].min().date()}  {len(a):5d}  "
          f"{a.mean()*1e4:7.2f}  {b.mean()*1e4:7.2f}  {(a.mean()-b.mean())*1e4:8.2f}  "
          f"{tt.statistic:6.2f}  {tt.pvalue:.4f}  {(a>0).mean():.3f}")

d1 = spy.loc[spy["pos"] == 1, "ret"]
years = (spy["date"].max() - spy["date"].min()).days / 365.25
print(f"\nSPY 1996-2024: $1 held only on the first trading day of each month -> "
      f"${(1+d1).prod():.3f} across {len(d1)} sessions ({len(d1)/len(spy)*100:.1f}% of "
      f"trading days); buy and hold -> ${(1+spy['ret']).prod():.2f} "
      f"({(1+spy['ret']).prod()**(1/years)-1:.2%} a year)")
print(f"SPY first-day median {d1.median()*1e4:.2f} bps, 5% trimmed mean "
      f"{stats.trim_mean(d1, 0.05)*1e4:.2f} bps, ex-January mean "
      f"{spy[(spy['pos']==1) & (spy['date'].dt.month!=1)]['ret'].mean()*1e4:.2f} bps")

# ---- chart -------------------------------------------------------------
BG, FG, ACCENT = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG)

colors = ["#f59e0b" if p == 1 else ACCENT for p in prof.index]
ax1.bar([str(p) for p in prof.index], prof["bps"], color=colors)
ax1.axhline(0, color=FG, lw=0.8)
ax1.set_title("SPY average daily total return by trading day of the month, 1996-2024",
              color=FG, fontsize=12)
ax1.set_ylabel("Average return (basis points)", color=FG)
ax1.set_xlabel("Trading day of month (negative counts back from month end)", color=FG)

y = np.arange(len(TICKERS))
ax2.barh(y + 0.2, [first_day[t][0] for t in TICKERS], height=0.4,
         color="#f59e0b", label="First trading day")
ax2.barh(y - 0.2, [first_day[t][1] for t in TICKERS], height=0.4,
         color=ACCENT, label="All other days")
ax2.set_yticks(y, TICKERS)
ax2.axvline(0, color=FG, lw=0.8)
ax2.invert_yaxis()
ax2.set_xlabel("Average daily total return (basis points)", color=FG)
ax2.set_title("Equity ETFs pay on the first session of the month; bond ETFs do not",
              color=FG, fontsize=12)
leg = ax2.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG)

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#333333")

plt.tight_layout()
plt.savefig("turn-of-the-month-first-trading-day-python.png", dpi=150, facecolor=BG)
