# Full write-up: https://xfinlink.com/blog/bollinger-band-squeeze-volatility-expansion-python
#
# Does a Bollinger Band squeeze predict a big move?
# Band width deciles against forward 20-day realised volatility, S&P 500, 2015-2024.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2014-01-02", "2024-12-31"
CHART = "bollinger-band-squeeze-volatility-expansion-python.png"

# ── Universe: point-in-time S&P 500 roster, carried by entity id ──────────────
roster = xfl.index("sp500", as_of="2014-12-31")
ids = sorted({int(i) for i in roster["entity_id"].dropna()})

frames = []
for i in range(0, len(ids), 25):
    frames.append(xfl.prices(entity_id=ids[i:i + 25], start=START, end=END,
                             fields=["adj_close"], max_rows=200000))
px = pd.concat(frames, ignore_index=True)
wide = px.pivot(index="date", columns="entity_id", values="adj_close").sort_index()
ret = wide.pct_change()

# Sessions moving more than 50% are dominated by corporate actions rather than
# trading, and one of them distorts a 20-day band for a month. Drop those names.
n_all = wide.shape[1]
keep = ~(ret.abs() > 0.50).any()
wide, ret = wide.loc[:, keep], ret.loc[:, keep]

# ── Bollinger band width, its own trailing percentile, forward outcomes ───────
width = 4.0 * wide.rolling(20).std() / wide.rolling(20).mean()
rank = width.rolling(252).rank(pct=True)
vol_now = ret.rolling(20).std() * np.sqrt(252)
vol_next = vol_now.shift(-20)
move_next = wide.shift(-20) / wide - 1.0

p = pd.DataFrame({"rank": rank.stack(), "width": width.stack(),
                  "vol_now": vol_now.stack(), "vol_next": vol_next.stack(),
                  "move": move_next.stack()}).dropna()
p = p[p.index.get_level_values(0) >= "2015-01-02"]
p["dec"] = np.ceil(p["rank"] * 10).clip(1, 10).astype(int)

g = p.groupby("dec").agg(n=("vol_next", "size"), width=("width", "median"),
                         vol_now=("vol_now", "median"), vol_next=("vol_next", "median"),
                         move=("move", lambda s: s.abs().median()),
                         up=("move", lambda s: (s > 0).mean()))
g["ratio"] = g["vol_next"] / g["vol_now"]

# ── Robustness: per company, and per calendar year ────────────────────────────
ends = p[p["dec"].isin([1, 10])].copy()
ends["eid"] = ends.index.get_level_values(1)
by_name = ends.groupby(["eid", "dec"])[["vol_now", "vol_next"]].median().dropna()
r1 = by_name.xs(1, level="dec")
r10 = by_name.xs(10, level="dec")
both = r1.join(r10, how="inner", lsuffix="_1", rsuffix="_10")
expands = int((both["vol_next_1"] / both["vol_now_1"] > 1).sum())
calmer = int((both["vol_next_1"] < both["vol_next_10"]).sum())

sq = p[p["dec"] == 1].copy()
sq["yr"] = sq.index.get_level_values(0).year
by_year = sq.groupby("yr").apply(
    lambda d: d["vol_next"].median() / d["vol_now"].median(), include_groups=False)

# ── Output ───────────────────────────────────────────────────────────────────
dates = p.index.get_level_values(0)
print(f"S&P 500 members as of 2014-12-31, daily closes {START} to {END}")
print(f"{int(keep.sum())} of {n_all} companies after the extreme-session screen, "
      f"{len(p):,} company-days")
print(f"Signal dates {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
      f"{p.index.get_level_values(1).nunique()} companies\n")

label = {1: " 1 tightest", 10: "10 widest"}
print(f"{'Band width decile':<18}{'n':>10}{'Width':>8}{'Vol now':>9}{'Vol next':>10}"
      f"{'Ratio':>8}{'|20d move|':>12}{'Up':>8}")
for d, x in g.iterrows():
    print(f"{label.get(d, f'{d:2d}'):<18}{int(x.n):>10,}{x.width:>8.2%}{x.vol_now:>9.1%}"
          f"{x.vol_next:>10.1%}{x.ratio:>8.2f}{x.move:>12.2%}{x.up:>8.1%}")
print(f"{'panel median':<18}{len(p):>10,}{p['width'].median():>8.2%}"
      f"{p['vol_now'].median():>9.1%}{p['vol_next'].median():>10.1%}"
      f"{p['vol_next'].median() / p['vol_now'].median():>8.2f}"
      f"{p['move'].abs().median():>12.2%}{(p['move'] > 0).mean():>8.1%}")

print(f"\nAfter a squeeze, volatility rises {g.loc[1, 'ratio'] - 1:.0%} from its "
      f"starting point and still lands at {g.loc[1, 'vol_next']:.1%}, the lowest "
      f"of the ten deciles.")
print(f"Per company: {expands} of {len(both)} expand after a squeeze; "
      f"{calmer} of {len(both)} are calmer after a squeeze than after a wide band.")
print("\nSqueeze-decile ratio of forward to current volatility, by year")
print("  ".join(f"{y} {v:.2f}" for y, v in by_year.items()))

# ── Chart ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 11)
ax.bar(x - 0.19, g["vol_now"] * 100, width=0.38, color="#4b5563", label="Volatility now")
ax.bar(x + 0.19, g["vol_next"] * 100, width=0.38, color="#3b82f6",
       label="Volatility over the next 20 trading days")
ax.axhline(p["vol_next"].median() * 100, color="#e0e0e0", lw=1, ls="--", alpha=0.7)
ax.text(0.62, p["vol_next"].median() * 100 + 0.7, "panel median",
        color="#e0e0e0", ha="left", fontsize=9, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["1\ntightest"] + [str(i) for i in range(2, 10)] + ["10\nwidest"])
ax.set_xlabel("Bollinger band width, decile of the stock's own past year")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("A Bollinger Band squeeze is followed by the calmest month, not the wildest")
ax.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
