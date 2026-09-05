# Full write-up: https://xfinlink.com/blog/residual-momentum-vs-raw-momentum-python
#
# Raw 12-1 momentum ranks stocks on total past price return, part of which is
# just exposure to the market. Residual momentum strips the market component
# out first with a fitted market model, then ranks on what is left. Both
# signals are sorted into deciles on the same point-in-time S&P 500 sample.

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ANCHORS = pd.date_range("2018-01-31", "2026-07-31", freq="ME").strftime("%Y-%m-%d").tolist()
START, END = "2014-06-01", "2026-08-31"
FORM, SKIP, EST = 252, 21, 756          # formation, skip, market-model estimation (trading days)
PNG = "residual-momentum-vs-raw-momentum-python.png"

# ---------------------------------------------------------------- data ------


def roster(as_of):
    return sorted(int(e) for e in xfl.index("sp500", as_of=as_of)["entity_id"].dropna())


def series(entity):
    return xfl.prices(entity_id=entity, start=START, end=END,
                      fields=["close", "return_daily"], max_rows=200000)


with ThreadPoolExecutor(8) as pool:
    rosters = dict(zip(ANCHORS, pool.map(roster, ANCHORS)))
universe = sorted({e for ids in rosters.values() for e in ids})

with ThreadPoolExecutor(8) as pool:
    px = pd.concat([d for d in pool.map(series, universe) if len(d)], ignore_index=True)
spy = xfl.prices("SPY", start=START, end=END, fields=["close", "return_daily"], max_rows=200000)

px = px.drop_duplicates(["entity_id", "date"])
px = px[px["close"] > 0]
R = px.pivot(index="date", columns="entity_id", values="return_daily").sort_index()
priced = R.shape[1]
R = R[[c for c in R.columns if not ((R[c] > 1.0).any() or (R[c] < -0.5).any())]]

cal = R.index
vals = R.values                                     # daily price returns
logret = np.log1p(R.fillna(0.0)).cumsum().values    # cumulative log price return
live = R.notna().cumsum().values                    # cumulative count of traded days
mkt = spy.set_index("date")["return_daily"].reindex(cal).values
pos = {e: i for i, e in enumerate(R.columns)}
month_end = pd.Series(np.arange(len(cal))).groupby(np.asarray(cal.year * 100 + cal.month)).max()
days = [int(month_end[int(a[:4]) * 100 + int(a[5:7])]) for a in ANCHORS]

# ------------------------------------------------------------ the sorts -----

rows, counts, spearman, overlap, dec_raw, dec_res = [], [], [], [], [], []
netbeta, betacorr = [], []
for anchor, t, nxt in zip(ANCHORS, days, days[1:]):
    s, f, b = t - SKIP, t - FORM, t - SKIP - EST    # skip point, formation start, estimation start
    ids = np.array([pos[e] for e in rosters[anchor] if e in pos])
    usable = ((live[s, ids] - live[b, ids] == s - b)          # full estimation window
              & (live[nxt, ids] - live[t, ids] == nxt - t))   # traded through the holding month
    ids = ids[usable]

    X, mm = vals[b + 1:s + 1][:, ids], mkt[b + 1:s + 1]       # market model, 756 days to the skip point
    md = mm - mm.mean()
    beta = (X - X.mean(0)).T @ md / (md @ md)
    alpha = X.mean(0) - beta * mm.mean()
    resid = vals[f + 1:s + 1][:, ids] - alpha - np.outer(mkt[f + 1:s + 1], beta)

    raw = logret[s, ids] - logret[f, ids]           # 12-1 cumulative log price return
    res = resid.sum(0)                              # 12-1 cumulative residual
    held = np.expm1(logret[nxt, ids] - logret[t, ids])        # the month held
    n = len(ids)
    d_raw = np.argsort(np.argsort(raw, kind="stable"), kind="stable") * 10 // n
    d_res = np.argsort(np.argsort(res, kind="stable"), kind="stable") * 10 // n

    rows.append((cal[nxt], held[d_raw == 9].mean(), held[d_raw == 0].mean(),
                 held[d_res == 9].mean(), held[d_res == 0].mean(), held.mean()))
    counts.append(n)
    spearman.append(stats.spearmanr(raw, res).statistic)
    overlap.append(len(set(ids[d_raw == 9]) & set(ids[d_res == 9])) / (d_raw == 9).sum())
    dec_raw.append([held[d_raw == k].mean() for k in range(10)])
    dec_res.append([held[d_res == k].mean() for k in range(10)])
    netbeta.append((beta[d_raw == 9].mean() - beta[d_raw == 0].mean(),
                    beta[d_res == 9].mean() - beta[d_res == 0].mean()))
    betacorr.append((stats.spearmanr(raw, beta).statistic, stats.spearmanr(res, beta).statistic))

out = pd.DataFrame(rows, columns=["date", "raw_win", "raw_lose",
                                  "res_win", "res_lose", "univ"]).set_index("date")
out["raw"] = out["raw_win"] - out["raw_lose"]
out["res"] = out["res_win"] - out["res_lose"]
N = len(out)


def stat(x):
    x = np.asarray(x)
    return {"mean": x.mean(), "ann": np.prod(1 + x) ** (12 / len(x)) - 1,
            "vol": x.std(ddof=1) * np.sqrt(12),
            "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(12),
            "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
            "hit": (x > 0).mean(), "dollar": np.prod(1 + x)}


gap = out["res"] - out["raw"]
years = out.groupby(out.index.year)[["raw", "res"]].apply(lambda d: (1 + d).prod() - 1)
DR, DE = np.array(dec_raw).mean(0), np.array(dec_res).mean(0)

# --------------------------------------------------------------- print ------

print("Raw versus residual 12-1 momentum, point-in-time S&P 500")
print(f"Panel:    {len(ANCHORS)} month-end rosters covering {len(universe)} companies; {priced} carry a daily")
print(f"          price series for the window and {R.shape[1]} remain after the return screen")
print(f"Signals:  formation window {FORM} trading days ending {SKIP} days before the sort date;")
print(f"          market model fitted on the prior {EST} trading days against SPY")
print(f"Sorts:    {N} monthly holding periods, {out.index[0]:%Y-%m} to {out.index[-1]:%Y-%m}, "
      f"median {int(np.median(counts))} names ranked")
print(f"          (range {min(counts)} to {max(counts)}); equal-weighted top decile minus bottom decile")
print()
print("Signal          Monthly   A year      Vol   Sharpe       t   Hit rate   $1 becomes")
for label, key in [("Raw 12-1", "raw"), ("Residual 12-1", "res")]:
    r = stat(out[key])
    print(f"{label:14s}  {r['mean']:7.3%}  {r['ann']:7.2%}  {r['vol']:7.2%}   {r['sharpe']:6.2f}  "
          f"{r['t']:6.2f}     {r['hit']:6.1%}       {r['dollar']:6.2f}")
print()
print("Legs of each spread, average monthly return")
for label, key in [("Raw winners", "raw_win"), ("Raw losers", "raw_lose"),
                   ("Residual winners", "res_win"), ("Residual losers", "res_lose"),
                   ("Whole sample", "univ")]:
    r = stat(out[key])
    print(f"  {label:18s} {r['mean']:7.3%}   a year {r['ann']:7.2%}   vol {r['vol']:7.2%}   "
          f"Sharpe {r['sharpe']:5.2f}")
print()
print("How far apart are the two rankings?")
print(f"  Mean rank correlation of the two scores   {np.mean(spearman):6.3f}   "
      f"({np.min(spearman):.3f} to {np.max(spearman):.3f})")
print(f"  Top decile shared by both signals         {np.mean(overlap):6.1%}   "
      f"({np.min(overlap):.1%} to {np.max(overlap):.1%})")
print(f"  Correlation of the two monthly spreads    {out[['raw', 'res']].corr().iloc[0, 1]:6.3f}")
print(f"  Residual minus raw, mean monthly          {gap.mean():6.3%}   "
      f"(t = {gap.mean() / (gap.std(ddof=1) / np.sqrt(N)):.2f})")
print()
NB, BC = np.array(netbeta), np.array(betacorr)
print("Market exposure carried into the long-short book")
print(f"  Rank correlation of score with beta       raw {BC[:, 0].mean():+6.3f}   "
      f"residual {BC[:, 1].mean():+6.3f}")
print(f"  Net beta of winners minus losers, mean    raw {NB[:, 0].mean():+6.3f}   "
      f"residual {NB[:, 1].mean():+6.3f}")
print(f"  Net beta, standard deviation across sorts raw {NB[:, 0].std(ddof=1):6.3f}   "
      f"residual {NB[:, 1].std(ddof=1):6.3f}")
print(f"  Net beta, range across sorts              raw {NB[:, 0].min():+.2f} to {NB[:, 0].max():+.2f}   "
      f"residual {NB[:, 1].min():+.2f} to {NB[:, 1].max():+.2f}")
print()
print("Long-short return by calendar year")
print("  Year      Raw   Residual")
for y, r in years.iterrows():
    print(f"  {y}  {r['raw']:7.2%}   {r['res']:7.2%}")
print(f"  Residual ahead in {(years['res'] > years['raw']).sum()} of {len(years)} calendar years")
print()
print("Average monthly return by decile (1 = worst past score, 10 = best)")
print("  Decile      " + "".join(f"{k + 1:7d}" for k in range(10)))
print("  Raw       " + "".join(f"{v:6.2%} " for v in DR))
print("  Residual  " + "".join(f"{v:6.2%} " for v in DE))

# --------------------------------------------------------------- chart ------

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(out.index, np.cumprod(1 + out["raw"]), color="#f59e0b", linewidth=1.8, label="raw 12-1")
ax1.plot(out.index, np.cumprod(1 + out["res"]), color="#3b82f6", linewidth=1.8, label="residual 12-1")
ax1.axhline(1, color="#e0e0e0", linewidth=0.8)
ax1.set_ylabel("Value of one dollar")
ax1.set_title("Raw versus residual momentum: winners minus losers, S&P 500 2018-2026")
ax1.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper left")

w = np.arange(10)
ax2.bar(w - 0.2, DR * 100, 0.4, color="#f59e0b", label="raw 12-1")
ax2.bar(w + 0.2, DE * 100, 0.4, color="#3b82f6", label="residual 12-1")
ax2.axhline(out["univ"].mean() * 100, color="#e0e0e0", linewidth=0.8, linestyle="--")
ax2.set_xticks(w)
ax2.set_xticklabels([str(k + 1) for k in w])
ax2.set_xlabel("Decile of past score (1 = worst, 10 = best); dashed line is the whole sample")
ax2.set_ylabel("Average return, % a month")
ax2.set_title("Next-month return by decile, averaged over 102 sorts", fontsize=10)
ax2.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper left")
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
