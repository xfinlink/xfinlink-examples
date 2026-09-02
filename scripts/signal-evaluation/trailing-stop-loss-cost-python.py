# Full write-up: https://xfinlink.com/blog/trailing-stop-loss-cost-python
#
# What a trailing stop-loss costs. One point-in-time S&P 500 roster, ten years of
# daily total returns, four stop thresholds, two exit rules: sell and stay in cash,
# or sell and buy back when the price regains the peak that triggered the exit.

import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2015-01-02", "2024-12-31"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25]
MIN_DAYS, SAMPLE, SEED = 500, 100, 20260902
PNG = "trailing-stop-loss-cost-python.png"

# ---------------------------------------------------------------- data ------

roster = xfl.index("sp500", as_of=START)
ids = sorted(int(e) for e in roster["entity_id"].dropna())
rng = np.random.default_rng(SEED)
sample = sorted(rng.choice(np.array(ids), size=SAMPLE, replace=False).tolist())
batches = [sample[i:i + 25] for i in range(0, len(sample), 25)]


def grab(b):
    d = xfl.prices(entity_id=b, start=START, end=END,
                   fields=["close", "return_daily"], max_rows=200000)
    print(f"batch of {len(b)} entities -> {len(d):,} rows", flush=True)
    return d


with ThreadPoolExecutor(4) as pool:
    px = pd.concat(list(pool.map(grab, batches)), ignore_index=True)

px = px.drop_duplicates(["entity_id", "date"]).sort_values(["entity_id", "date"])
px = px[px["close"] > 0]
raw_rows, raw_names = len(px), px["entity_id"].nunique()

bad = px.groupby("entity_id")["return_daily"].agg(lambda s: (s > 1.0).any() or (s < -0.75).any())
px = px[~px["entity_id"].isin(bad[bad].index)]
screened = px["entity_id"].nunique()

g = px.groupby("entity_id")
keep = g.size()[(g.size() >= MIN_DAYS) & (g["date"].min() <= pd.Timestamp("2015-01-15"))].index
px = px[px["entity_id"].isin(keep)]

# ----------------------------------------------------------- stop rule ------


def run(P, thr, reenter):
    """Trailing stop on a total-return index P. Sell at the close of the day the
    drawdown from the running peak first reaches thr. With reenter, buy back at
    the close of the first day P regains the peak that triggered the exit."""
    w, invested, entry, peak, ref, exits = 1.0, True, P[0], P[0], 0.0, 0
    gaps = []
    for t in range(1, len(P)):
        if invested:
            if P[t] > peak:
                peak = P[t]
            elif P[t] <= peak * (1.0 - thr):
                w *= P[t] / entry
                invested, ref, exits = False, peak, exits + 1
                gaps.append(P[t] / peak - 1.0)
        elif reenter and P[t] >= ref:
            invested, entry, peak = True, P[t], P[t]
    if invested:
        w *= P[-1] / entry
    return w, exits, gaps


rows, fills = [], []
for eid, d in px.groupby("entity_id"):
    d = d.sort_values("date")
    P = np.cumprod(1.0 + d["return_daily"].fillna(0.0).to_numpy())
    rec = {"ticker": d["ticker"].iloc[-1], "name": d["entity_name"].iloc[-1],
           "n": len(d), "years": (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25,
           "last": d["date"].iloc[-1], "bh": P[-1] - 1.0}
    for thr in THRESHOLDS:
        for tag, re_ in (("idle", False), ("re", True)):
            w, ex, gaps = run(P, thr, re_)
            rec[f"{tag}{int(thr * 100)}"] = w - 1.0
            rec[f"x{tag}{int(thr * 100)}"] = ex
            if tag == "idle":
                fills += [(thr, v) for v in gaps]
    rows.append(rec)

res = pd.DataFrame(rows)
res["bh_cagr"] = (1 + res["bh"]) ** (1 / res["years"]) - 1
fillfr = pd.DataFrame(fills, columns=["thr", "gap"])

summ = []
for thr in THRESHOLDS:
    k = int(thr * 100)
    row = {"thr": k}
    for tag in ("idle", "re"):
        d = res[f"{tag}{k}"] - res["bh"]
        cg = (1 + res[f"{tag}{k}"]) ** (1 / res["years"]) - 1
        row[f"{tag}_mean"], row[f"{tag}_med"] = d.mean(), d.median()
        row[f"{tag}_win"] = (res[f"{tag}{k}"] > res["bh"]).mean()
        row[f"{tag}_cagr"] = (cg - res["bh_cagr"]).median()
        row[f"{tag}_p5"] = res[f"{tag}{k}"].quantile(0.05)
        row[f"{tag}_min"] = res[f"{tag}{k}"].min()
        row[f"{tag}_lose50"] = (res[f"{tag}{k}"] < -0.5).mean()
    row["exits"] = res[f"xidle{k}"].mean()
    row["reexits"] = res[f"xre{k}"].mean()
    row["ever"] = (res[f"xidle{k}"] > 0).mean()
    row["fill"] = fillfr[fillfr["thr"] == thr]["gap"].median()
    row["fillw"] = fillfr[fillfr["thr"] == thr]["gap"].min()
    summ.append(row)
S = pd.DataFrame(summ).set_index("thr")

# --------------------------------------------------------------- print ------

print(f"Trailing stops against buy and hold, point-in-time S&P 500 roster of {START}")
print(f"Sample: {SAMPLE} of the 500 members drawn with seed {SEED}; {raw_names} carry a daily "
      f"series over the window ({raw_rows:,} rows),")
print(f"        {raw_names - screened} set aside by the return screen and {screened - len(res)} "
      f"by the {MIN_DAYS}-day minimum, leaving {len(res)} names and {int(px.groupby('entity_id').size().sum()):,} daily observations")
print(f"        {int((res['last'] < '2024-12-15').sum())} of them stop trading before the window ends "
      f"and are measured to their last traded day; median history {res['years'].median():.1f} years")
print(f"Buy and hold: 5th pct {res['bh'].quantile(0.05):+.1%}  median {res['bh'].median():+.1%}  "
      f"95th pct {res['bh'].quantile(0.95):+.1%}  worst {res['bh'].min():+.1%}  "
      f"share below -50%: {(res['bh'] < -0.5).mean():.1%}")
print()
print("Rule A: sell at the stop, proceeds sit in cash earning nothing")
print("stop   mean diff  median diff   stop beats   median CAGR   exits    5th pct     worst    share")
print(" (%)       (pp)         (pp)         hold      diff (pp)   /name     return    return  <  -50%")
for k, r in S.iterrows():
    print(f"{k:4d}  {r['idle_mean'] * 100:9.1f}  {r['idle_med'] * 100:11.1f}  {r['idle_win'] * 100:10.1f}%  "
          f"{r['idle_cagr'] * 100:11.2f}  {r['exits']:6.2f}  {r['idle_p5'] * 100:8.1f}%  "
          f"{r['idle_min'] * 100:8.1f}%  {r['idle_lose50'] * 100:6.1f}%")
print()
print("Rule B: sell at the stop, buy back when the index regains the peak that triggered the exit")
print("stop   mean diff  median diff   stop beats   median CAGR   round    5th pct     worst    share")
print(" (%)       (pp)         (pp)         hold      diff (pp)   trips     return    return  <  -50%")
for k, r in S.iterrows():
    print(f"{k:4d}  {r['re_mean'] * 100:9.1f}  {r['re_med'] * 100:11.1f}  {r['re_win'] * 100:10.1f}%  "
          f"{r['re_cagr'] * 100:11.2f}  {r['reexits']:6.2f}  {r['re_p5'] * 100:8.1f}%  "
          f"{r['re_min'] * 100:8.1f}%  {r['re_lose50'] * 100:6.1f}%")
print()
print("How often the stop fires, and how far past the trigger the sale actually lands")
for k, r in S.iterrows():
    print(f"  {k:2d}% stop fires on {r['ever'] * 100:5.1f}% of names   "
          f"median drawdown at the sale {r['fill'] * 100:6.2f}%   deepest single sale {r['fillw'] * 100:7.2f}%")
print()
print("Six worst buy-and-hold outcomes, total return %")
for _, r in res.nsmallest(6, "bh").iterrows():
    print(f"  {r['ticker']:>6}  {r['name'][:26]:<26} {r['years']:4.1f}y   hold {r['bh'] * 100:8.1f}   "
          f"20% stop cash {r['idle20'] * 100:8.1f}   20% stop buy back {r['re20'] * 100:8.1f}")
print("Six best buy-and-hold outcomes, total return %")
for _, r in res.nlargest(6, "bh").iterrows():
    print(f"  {r['ticker']:>6}  {r['name'][:26]:<26} {r['years']:4.1f}y   hold {r['bh'] * 100:8.1f}   "
          f"20% stop cash {r['idle20'] * 100:8.1f}   20% stop buy back {r['re20'] * 100:8.1f}")

# --------------------------------------------------------------- chart ------

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = np.arange(len(THRESHOLDS))
ax1.bar(x - 0.18, S["idle_med"] * 100, 0.34, color="#3b82f6", label="proceeds sit in cash")
ax1.bar(x + 0.18, S["re_med"] * 100, 0.34, color="#f59e0b", label="buy back at the old peak")
ax1.axhline(0, color="#e0e0e0", linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(t * 100)}%" for t in THRESHOLDS])
ax1.set_xlabel("Trailing stop, percent below the running peak")
ax1.set_ylabel("Median cost per stock,\npercentage points of total return")
ax1.set_title("What a trailing stop costs: S&P 500 members, 2015 to 2024")
ax1.set_ylim(-155, 0)
ax1.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="lower right")

q = np.linspace(0, 1, 201)
ax2.plot(q * 100, res["bh"].quantile(q) * 100, color="#e0e0e0", linewidth=1.8, label="buy and hold")
ax2.plot(q * 100, res["idle20"].quantile(q) * 100, color="#3b82f6", linewidth=1.8, label="20% stop, cash")
ax2.plot(q * 100, res["re20"].quantile(q) * 100, color="#f59e0b", linewidth=1.8, label="20% stop, buy back")
ax2.axhline(0, color="#3f3f3f", linewidth=0.8)
ax2.set_yscale("symlog", linthresh=100)
ax2.set_xlabel("Stocks ranked worst to best (percentile)")
ax2.set_ylabel("Total return over the window (%)")
ax2.set_title("Every outcome, sorted: the stop lifts the left tail and cuts the right", fontsize=10)
ax2.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper left")
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
