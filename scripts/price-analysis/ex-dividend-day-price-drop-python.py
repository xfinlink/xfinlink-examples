# Full write-up: https://xfinlink.com/blog/ex-dividend-day-price-drop-python
#
# How much does a stock actually fall on its ex-dividend date?
# Measures the drop-to-dividend ratio on S&P 500 ex-dividend events, 2021-2025,
# using the overnight move (previous close to ex-date open) and adjusting for
# the market's own overnight move with each name's estimated beta.

import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2021-01-01", "2025-12-31"
FIELDS = ["open", "close", "dividend", "return_daily", "split_ratio"]
CHUNK, WORKERS = 10, 6
MAX_YIELD = 0.02        # dividend / previous close ceiling for a regular payer
MAX_GAP_DAYS = 5        # calendar days allowed between the two closes
MAX_OVERNIGHT = 0.10    # absolute overnight move ceiling


def fetch(**kwargs):
    last = None
    for attempt in range(4):
        try:
            return xfl.prices(max_rows=200_000, **kwargs)
        except xfl.XfinlinkError as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last


# ---------------------------------------------------------------- market leg
spy = fetch(ticker="SPY", start=START, end=END,
            fields=["open", "close", "return_daily"]).sort_values("date")
spy["mkt_overnight"] = spy["open"] / spy["close"].shift(1) - 1.0
spy["mkt_close"] = spy["close"] / spy["close"].shift(1) - 1.0
spy = spy[["date", "return_daily", "mkt_overnight", "mkt_close"]].rename(
    columns={"return_daily": "mkt_total"})

# ---------------------------------------------------------------- member leg
roster = xfl.index("sp500", as_of=END)
ids = sorted({int(e) for e in roster["entity_id"].dropna()})
chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]

with ThreadPoolExecutor(WORKERS) as pool:
    px = pd.concat(pool.map(
        lambda c: fetch(entity_id=c, start=START, end=END, fields=FIELDS), chunks),
        ignore_index=True)

px = (px.drop_duplicates(["entity_id", "date"])
        .sort_values(["entity_id", "date"])
        .reset_index(drop=True)
        .merge(spy, on="date", how="left"))

grp = px.groupby("entity_id", sort=False)
px["prev_close"] = grp["close"].shift(1)
px["prev_date"] = grp["date"].shift(1)
px["prev_split"] = grp["split_ratio"].shift(1)

# Beta on daily total returns, both legs clipped so one bad print cannot set it.
fit = px.dropna(subset=["return_daily", "mkt_total"]).copy()
fit = fit[fit["return_daily"].abs() <= 0.25]
beta = (fit.groupby("entity_id")
           .apply(lambda d: np.cov(d["return_daily"], d["mkt_total"])[0, 1]
                  / np.var(d["mkt_total"], ddof=1), include_groups=False)
           .rename("beta"))

# ------------------------------------------------------------------- events
ev = px[px["dividend"].notna() & (px["dividend"] > 0)].copy()
n0 = len(ev)
counts = {}

ev = ev.dropna(subset=["prev_close", "open", "close", "mkt_overnight"])
ev = ev[(ev["open"] > 0) & (ev["prev_close"] > 0)]
counts["incomplete price pair"] = n0 - len(ev)

n = len(ev)
ev = ev[(ev["date"] - ev["prev_date"]).dt.days <= MAX_GAP_DAYS]
counts["gap in the price series"] = n - len(ev)

n = len(ev)
ev = ev[ev["split_ratio"].isna() & ev["prev_split"].isna()]
counts["split on or beside the ex-date"] = n - len(ev)

ev["div_yield"] = ev["dividend"] / ev["prev_close"]
n = len(ev)
counts[f"dividend above {MAX_YIELD:.0%} of price"] = int((ev["div_yield"] > MAX_YIELD).sum())
ev = ev[ev["div_yield"] <= MAX_YIELD]

n = len(ev)
ev = ev[(ev["open"] / ev["prev_close"] - 1.0).abs() <= MAX_OVERNIGHT]
counts["overnight move beyond the artefact ceiling"] = n - len(ev)

ev = ev.join(beta, on="entity_id").dropna(subset=["beta"])


def ratios(d):
    """Raw and market-adjusted drop-to-dividend ratios, open and close conventions."""
    d = d.copy()
    d["raw"] = (d["prev_close"] - d["open"]) / d["dividend"]
    d["adj"] = (d["prev_close"] - d["open"]
                + d["beta"] * d["mkt_overnight"] * d["prev_close"]) / d["dividend"]
    d["adj_cc"] = (d["prev_close"] - d["close"]
                   + d["beta"] * d["mkt_close"] * d["prev_close"]) / d["dividend"]
    return d


ev = ratios(ev)


def summary(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return dict(n=len(s), mean=s.mean(), se=s.std(ddof=1) / np.sqrt(len(s)),
                median=s.median(), q1=q1, q3=q3,
                below1=(s < 1).mean(), below0=(s < 0).mean())


print(f"S&P 500 ex-dividend events, {START} to {END}")
print(f"members {len(ids)}  payers {ev['entity_id'].nunique()}  "
      f"raw events {n0}  kept {len(ev)}\n")
for k, v in counts.items():
    print(f"  dropped {v:>5}  {k}")

print("\n                        N     mean      SE   median      IQR   <1.0    <0")
for label, col in [("raw, open", "raw"), ("market-adjusted, open", "adj"),
                   ("market-adjusted, close", "adj_cc")]:
    s = summary(ev[col])
    print(f"{label:<22} {s['n']:>5} {s['mean']:>8.3f} {s['se']:>7.3f} "
          f"{s['median']:>8.3f}  {s['q1']:>5.2f}-{s['q3']:<5.2f} "
          f"{s['below1']:>5.1%} {s['below0']:>5.1%}")

drop = ev["adj"] * ev["dividend"]
pooled = drop.sum() / ev["dividend"].sum()
slope = (drop * ev["dividend"]).sum() / (ev["dividend"] ** 2).sum()
print(f"\npooled drop / pooled dividend  {pooled:.3f}"
      f"    slope of drop on dividend, through the origin  {slope:.3f}")

ev["bucket"] = pd.qcut(ev["div_yield"], 4,
                       labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
bk = ev.groupby("bucket", observed=True).agg(
    n=("adj", "size"), yld=("div_yield", "median"),
    mean=("adj", "mean"), median=("adj", "median"),
    se=("adj", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
print("\nmarket-adjusted ratio by dividend size (quartiles of dividend / price)")
print("bucket          N   median yield     mean      SE   median")
for name, r in bk.iterrows():
    print(f"{name:<12} {int(r['n']):>5} {r['yld']:>12.2%} {r['mean']:>8.3f} "
          f"{r['se']:>7.3f} {r['median']:>8.3f}")

# What the shortfall is worth: the part of the dividend the price never gives up.
ev["kept"] = (1.0 - ev["adj"]) * ev["div_yield"]
print(f"\nshortfall per event: {1e4 * ev['kept'].mean():.1f} bps of the share price, "
      f"per-event sd {1e4 * ev['kept'].std(ddof=1):.0f} bps, "
      f"t = {ev['kept'].mean() / (ev['kept'].std(ddof=1) / np.sqrt(len(ev))):.1f}")

yr = ev.groupby(ev["date"].dt.year).agg(n=("adj", "size"), median=("adj", "median"))
print("\nmarket-adjusted median by year")
print("  " + "  ".join(f"{y}: {r['median']:.3f} (N={int(r['n'])})" for y, r in yr.iterrows()))

# ------------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.6, 1]})

shown = ev.loc[ev["adj"].between(-2, 4), "adj"]
ax1.hist(shown, bins=72, color="#3b82f6", alpha=0.85)
ax1.axvline(1.0, color="#f59e0b", lw=1.6, label="theory: drop equals the dividend")
ax1.axvline(ev["adj"].median(), color="#e0e0e0", lw=1.6, ls="--",
            label=f"median {ev['adj'].median():.2f}")
ax1.set_xlabel(f"Price drop divided by the dividend, market-adjusted "
               f"({len(shown):,} of {len(ev):,} events fall between -2 and 4)")
ax1.set_ylabel("Ex-dividend events")
ax1.set_title(f"How far a stock falls on its ex-dividend date  "
              f"({len(ev):,} S&P 500 events, {START[:4]}-{END[:4]})", pad=12)
ax1.legend(frameon=False, labelcolor="#e0e0e0")
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

x = np.arange(len(bk))
ax2.errorbar(x, bk["mean"], yerr=1.96 * bk["se"], fmt="o", color="#3b82f6",
             ecolor="#3b82f6", capsize=4, ms=7, label="mean, 95% interval")
ax2.plot(x, bk["median"], "s", color="#e0e0e0", ms=6, label="median")
ax2.axhline(1.0, color="#f59e0b", lw=1.4)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{i}\n{y:.2%} of price" for i, y in zip(bk.index, bk["yld"])])
ax2.set_ylabel("Drop / dividend")
ax2.set_xlabel("Dividend size relative to share price")
ax2.legend(frameon=False, labelcolor="#e0e0e0", loc="lower right")
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig("ex-dividend-day-price-drop-python.png", dpi=150, facecolor="#0a0a0a")
