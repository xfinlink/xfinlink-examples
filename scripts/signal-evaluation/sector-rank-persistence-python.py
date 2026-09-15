# Full write-up: https://xfinlink.com/blog/sector-rank-persistence-python
#
# Sector rotation rests on an unstated assumption: that the ordering of sector
# returns carries from one period into the next. This measures that ordering
# directly. Nine Select Sector SPDRs, quarterly total returns from 1999 to 2024,
# ranked 1 to 9 each quarter. The Spearman rank correlation between quarter t
# and quarter t+h says how much of the ordering survives h quarters. The same
# measurement is repeated on realised volatility, so the return result has a
# control that is known to persist rather than only a null to report.

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Cons Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons Discretionary",
}
TICKERS = list(SECTORS)
START, END = "1998-12-01", "2024-12-31"
FIRST_Q, LAST_Q = "1999Q1", "2024Q4"
HORIZONS = [1, 2, 4, 8]
PNG = "sector-rank-persistence-python.png"

BG, FG, ACCENT, MUTED = "#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b"

# ---------------------------------------------------------------- data
raw = xfl.prices(TICKERS, start=START, end=END,
                 fields=["close", "adj_close", "return_daily"], max_rows=200000)

px = raw.sort_values(["ticker", "date"]).copy()
px["price_return"] = px.groupby("ticker")["adj_close"].pct_change()
px["year"] = px["date"].dt.year

# distribution check: total return less price return, per fund-year, in points
gap = (px[(px["date"] >= "1999-01-01") & px["return_daily"].notna()
          & px["price_return"].notna()]
         .assign(d=lambda x: x["return_daily"] - x["price_return"])
         .groupby(["ticker", "year"])["d"].sum() * 100)

daily = px.dropna(subset=["return_daily"])
daily = daily[daily["date"] >= "1999-01-01"]
piv = daily.pivot(index="date", columns="ticker", values="return_daily")[TICKERS]
qtr = piv.index.to_period("Q")

ret = piv.groupby(qtr).apply(lambda x: (1 + x).prod() - 1)
vol = piv.groupby(qtr).std() * np.sqrt(252)
sessions = piv.groupby(qtr).size()
ret = ret.loc[FIRST_Q:LAST_Q]
vol = vol.loc[FIRST_Q:LAST_Q]
sessions = sessions.loc[FIRST_Q:LAST_Q]

ret_rank = ret.rank(axis=1, ascending=False)   # 1 = best return
vol_rank = vol.rank(axis=1, ascending=False)   # 1 = most volatile

# ---------------------------------------------------------------- statistics
def persistence(rank_table, h):
    """Mean Spearman rank correlation between quarter t and quarter t+h."""
    rhos = np.array([
        stats.spearmanr(rank_table.iloc[i], rank_table.iloc[i + h]).statistic
        for i in range(len(rank_table) - h)
    ])
    t = stats.ttest_1samp(rhos, 0.0)
    ci = 1.96 * rhos.std(ddof=1) / np.sqrt(len(rhos))
    return rhos.mean(), ci, t.statistic, t.pvalue, len(rhos)

per = {
    "Quarterly return": [persistence(ret_rank, h) for h in HORIZONS],
    "Realised volatility": [persistence(vol_rank, h) for h in HORIZONS],
}

# the same question at other frequencies, in case the quarter is the wrong unit
mth = piv.groupby(piv.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
mth = mth.loc["1999-01":"2024-12"].rank(axis=1, ascending=False)
yr = piv.groupby(piv.index.year).apply(lambda x: (1 + x).prod() - 1)
yr = yr.loc[1999:2024].rank(axis=1, ascending=False)
other = [("Monthly rank, 1 month on", persistence(mth, 1)),
         ("Monthly rank, 12 months on", persistence(mth, 12)),
         ("Annual rank, 1 year on", persistence(yr, 1))]

# repeat rates for the extreme ranks
def repeat(rank_table, target):
    who = rank_table.idxmin(axis=1) if target == "top" else rank_table.idxmax(axis=1)
    hits = int((who.values[:-1] == who.values[1:]).sum())
    n = len(who) - 1
    return hits, n, stats.binomtest(hits, n, 1 / len(TICKERS)).pvalue

top_ret = repeat(ret_rank, "top")
bot_ret = repeat(ret_rank, "bottom")
top_vol = repeat(vol_rank, "top")
bot_vol = repeat(vol_rank, "bottom")

# next-quarter return conditioned on this quarter's rank
nxt = {}
for r in range(1, len(TICKERS) + 1):
    vals = [ret.iloc[i + 1][ret_rank.iloc[i][ret_rank.iloc[i] == r].index[0]]
            for i in range(len(ret) - 1)]
    nxt[r] = np.array(vals)
all_next = ret.iloc[1:].values.mean()

spread = (ret.max(axis=1) - ret.min(axis=1)) * 100

# ---------------------------------------------------------------- output
print(f"Select Sector SPDRs, quarterly total returns, {FIRST_Q} to {LAST_Q}")
print(f"Funds: {len(TICKERS)}   quarters: {len(ret)}   daily rows: {len(daily):,}")
print(f"Sessions per quarter: min {sessions.min()}, median {int(sessions.median())}, max {sessions.max()}")
print(f"Every fund priced on every session: {bool(piv.notna().all().all())}   "
      f"dates ordered: {bool(piv.index.is_monotonic_increasing)}")
print(f"Implied distribution per fund-year: median {gap.median():.2f} points, "
      f"max {gap.max():.2f}, none below -0.05 across {len(gap)} fund-years: "
      f"{bool((gap > -0.05).all())}")
print(f"Largest quarterly move: {ret.max().max() * 100:+.1f}% / {ret.min().min() * 100:+.1f}%")
print()

print("How much of the sector ordering survives? Mean Spearman rank correlation")
print("between quarter t and quarter t+h, one correlation per pair of quarters.")
print()
print(f"{'Ranked on':<22}{'Horizon':>9}{'Pairs':>8}{'Mean rho':>11}{'95% CI':>18}"
      f"{'t':>8}{'p':>12}")
print("-" * 88)
for label, rows in per.items():
    for h, (m, ci, t, p, n) in zip(HORIZONS, rows):
        hz = f"{h}Q"
        pv = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        print(f"{label:<22}{hz:>9}{n:>8}{m:>+11.3f}"
              f"{f'{m - ci:+.3f} to {m + ci:+.3f}':>18}{t:>+8.2f}{pv:>12}")
    print()

print("The same measurement on returns ranked at other frequencies")
print(f"{'Ranked on':<30}{'Pairs':>8}{'Mean rho':>11}{'t':>8}{'p':>10}")
print("-" * 67)
for label, (m, ci, t, p, n) in other:
    print(f"{label:<30}{n:>8}{m:>+11.3f}{t:>+8.2f}{p:>10.3f}")
print()

print("Does the extreme sector repeat? Chance alone gives 11.1% for 9 funds.")
print(f"{'Event':<40}{'Repeats':>10}{'Rate':>9}{'p':>10}")
print("-" * 69)
for name, (hits, n, p) in [
    ("Best-returning sector repeats next quarter", top_ret),
    ("Worst-returning sector repeats", bot_ret),
    ("Most volatile sector repeats", top_vol),
    ("Least volatile sector repeats", bot_vol),
]:
    print(f"{name:<40}{f'{hits} of {n}':>10}{hits / n * 100:>8.1f}%{p:>10.4f}")
print()

print("Next quarter's return, by this quarter's rank")
print(f"{'Rank this quarter':<20}{'Mean next Q':>13}{'Median':>10}{'Positive':>10}")
print("-" * 53)
for r in range(1, len(TICKERS) + 1):
    v = nxt[r]
    tag = f"{r}" + (" (best)" if r == 1 else " (worst)" if r == len(TICKERS) else "")
    print(f"{tag:<20}{v.mean() * 100:>+12.2f}%{np.median(v) * 100:>+9.2f}%"
          f"{(v > 0).mean() * 100:>9.0f}%")
print(f"{'All sectors':<20}{all_next * 100:>+12.2f}%")
w_vs_all = stats.ttest_rel(nxt[1], ret.iloc[1:].mean(axis=1).values)
print(f"Last quarter's winner against the all-sector average: "
      f"{(nxt[1].mean() - all_next) * 100:+.2f} points, t = {w_vs_all.statistic:+.2f}, "
      f"p = {w_vs_all.pvalue:.3f}")
print()

print("Sector record over the window")
print(f"{'Sector':<20}{'Ticker':>8}{'Mean ret rank':>15}{'Q at #1':>9}{'Q at #9':>9}"
      f"{'Mean vol rank':>15}")
print("-" * 76)
for t in sorted(TICKERS, key=lambda x: ret_rank[x].mean()):
    print(f"{SECTORS[t]:<20}{t:>8}{ret_rank[t].mean():>15.2f}"
          f"{int((ret_rank[t] == 1).sum()):>9}{int((ret_rank[t] == 9).sum()):>9}"
          f"{vol_rank[t].mean():>15.2f}")
print()
print(f"Best minus worst sector, per quarter: mean {spread.mean():.1f} points, "
      f"median {spread.median():.1f}, smallest {spread.min():.1f}, largest {spread.max():.1f}")

# ---------------------------------------------------------------- chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#333333")

x = np.arange(len(HORIZONS))
w = 0.36
for off, (label, color) in zip((-w / 2, w / 2),
                               [("Quarterly return", ACCENT),
                                ("Realised volatility", MUTED)]):
    means = [r[0] for r in per[label]]
    errs = [r[1] for r in per[label]]
    ax1.bar(x + off, means, w, yerr=errs, capsize=3, color=color, label=label,
            error_kw={"ecolor": "#777777", "lw": 1})
ax1.axhline(0, color="#555555", lw=1)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{h} quarter" if h == 1 else f"{h} quarters" for h in HORIZONS])
ax1.set_xlabel("How far ahead", color=FG, fontsize=9)
ax1.set_ylabel("Rank correlation with the later quarter", color=FG, fontsize=9)
ax1.set_title("Ordering that survives", color=FG, fontsize=11)
ax1.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, fontsize=8)

ranks = list(range(1, len(TICKERS) + 1))
ax2.bar(ranks, [nxt[r].mean() * 100 for r in ranks], color=ACCENT, width=0.7)
ax2.axhline(all_next * 100, color=MUTED, lw=1.2, ls="--",
            label=f"All sectors, {all_next * 100:.2f}%")
ax2.set_xticks(ranks)
ax2.set_xlabel("Rank this quarter (1 = best)", color=FG, fontsize=9)
ax2.set_ylabel("Mean return next quarter (%)", color=FG, fontsize=9)
ax2.set_title("What the winner earns next", color=FG, fontsize=11)
ax2.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, fontsize=8)

fig.suptitle("Does last quarter's best sector stay on top? 1999-2024",
             color=FG, fontsize=13)
plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor=BG)
