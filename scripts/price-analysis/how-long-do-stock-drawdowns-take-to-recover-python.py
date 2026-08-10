# Full write-up: https://xfinlink.com/blog/how-long-do-stock-drawdowns-take-to-recover-python
"""Drawdown depth and recovery time for individual large-cap US stocks, 1996-2026.

The universe is the union of point-in-time S&P 500 rosters, one per year end, so
companies that later left the index stay in the sample and recoveries that never
happened are not quietly deleted. Prices are monthly split-adjusted closes and
each company is tracked from the month it joined the index.
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "1996-01-01", "2026-07-31"
END_M = pd.Period("2026-07", "M")
SLUG = "how-long-do-stock-drawdowns-take-to-recover-python"
BUCKETS = [("20-30%", 0.20, 0.30), ("30-50%", 0.30, 0.50),
           ("50-70%", 0.50, 0.70), ("70%+", 0.70, 1.01)]

# ------------------------------------------------------- point-in-time universe
roster_dates = [f"{y}-12-31" for y in range(1995, 2026)] + [END]
rosters = []
for d in roster_dates:
    r = xfl.index("sp500", as_of=d)
    if not r.empty:
        rosters.append(r[["entity_id", "added_date"]])
roster = pd.concat(rosters).dropna(subset=["entity_id"])
roster["entity_id"] = roster["entity_id"].astype(int)
joined = roster.groupby("entity_id")["added_date"].min()
joined = pd.to_datetime(joined).dt.to_period("M")
ids = sorted(joined.index)
print(f"point-in-time S&P 500 rosters, {len(roster_dates)} dates 1995-2026: "
      f"{len(ids)} distinct companies")

# ------------------------------------------------------- monthly prices
frames, queue = [], [ids[i:i + 40] for i in range(0, len(ids), 40)]
while queue:
    batch = queue.pop(0)
    try:
        d = xfl.prices(entity_id=batch, start=START, end=END, interval="1mo",
                       fields=["adj_close"], max_rows=500000)
        if not d.empty:
            frames.append(d)
    except Exception:                       # split the batch and retry smaller
        if len(batch) == 1:
            continue
        h = len(batch) // 2
        queue = [batch[:h], batch[h:]] + queue

px = pd.concat(frames, ignore_index=True)
px["m"] = pd.to_datetime(px["date"]).dt.to_period("M")
print(f"monthly split-adjusted closes 1996-01 to 2026-07: {len(px):,} bars "
      f"on {px['entity_id'].nunique()} companies")

# ------------------------------------------------------- sample construction
px["join_m"] = px["entity_id"].map(joined)
px = px[px["m"] >= px["join_m"]].sort_values(["entity_id", "m"]).reset_index(drop=True)
print(f"each series starts the month the company joined the index: {len(px):,} bars")

# a monthly series that stops for more than a month has stopped; keep the first block
px["gap"] = px.groupby("entity_id")["m"].diff().map(lambda x: x.n if pd.notna(x) else 1)
px["block"] = px.groupby("entity_id")["gap"].transform(lambda s: (s > 1).cumsum())
n_trunc = px.loc[px["block"] > 0, "entity_id"].nunique()
px = px[px["block"] == 0].copy()

px["ret"] = px.groupby("entity_id")["adj_close"].pct_change()
px["prev_tick"] = px.groupby("entity_id")["ticker"].shift()


def symbol_reappears(seq):
    runs = [k for k, _ in itertools.groupby(seq)]
    return len(runs) != len(set(runs))


s1 = set(px.groupby("entity_id")["ticker"].apply(list)
         .loc[lambda s: s.map(symbol_reappears)].index)
s2 = set(px.loc[(px["ticker"] != px["prev_tick"]) & px["prev_tick"].notna()
                & (px["ret"].abs() > 0.5), "entity_id"])
s3 = set(px.loc[(px["ret"] > 2.0) | (px["ret"] < -0.90), "entity_id"])
nobs = px.groupby("entity_id").size()
s4 = set(nobs[nobs < 36].index)

sub = px[~px["entity_id"].isin(s1 | s2 | s3 | s4)]
names = sub.groupby("entity_id")["entity_name"].last()
ticks = sub.groupby("entity_id")["ticker"].last()
last_m = sub.groupby("entity_id")["m"].max()
print(f"{n_trunc} companies truncated at a break in trading")
print(f"set aside: {len(s1)} where a symbol reappears after another, {len(s2)} "
      f"with a price step above 50% at a symbol change,")
print(f"           {len(s3)} with a single month beyond +200% or -90%, "
      f"{len(s4)} with under 36 months")
print(f"sample: {sub['entity_id'].nunique()} companies, {len(sub):,} monthly bars, "
      f"{int((last_m >= END_M - 1).sum())} still trading at 2026-07")


# ------------------------------------------------------- drawdown episodes
def episodes(values):
    """Peak -> low -> recovery episodes against the running high of the series."""
    out, peak, pi, low, li, live = [], values[0], 0, values[0], 0, False
    for i in range(1, len(values)):
        x = values[i]
        if x >= peak:
            if live:
                out.append((pi, li, i, 1 - low / peak))
                live = False
            peak, pi, low, li = x, i, x, i
        elif not live:
            live, low, li = True, x, i
        elif x < low:
            low, li = x, i
    if live:
        out.append((pi, li, None, 1 - low / peak))
    return out


rows = []
for eid, gdf in sub.groupby("entity_id"):
    gdf = gdf.sort_values("m")
    ms = gdf["m"].tolist()
    for pi, li, ri, depth in episodes(gdf["adj_close"].values):
        if depth < 0.20:
            continue
        rows.append({"entity_id": eid, "peak_m": ms[pi], "low_m": ms[li],
                     "rec_m": ms[ri] if ri is not None else None, "depth": depth,
                     "months": (ri - li) if ri is not None else np.nan,
                     "watched": len(ms) - 1 - li, "last_m": ms[-1]})

ep = pd.DataFrame(rows)
ep["recovered"] = ep["months"].notna()
ep["trading"] = ep["last_m"] >= END_M - 1
ep["name"] = ep["entity_id"].map(names)
ep["ticker"] = ep["entity_id"].map(ticks)
ep["bucket"] = pd.cut(ep["depth"], [b[1] for b in BUCKETS] + [BUCKETS[-1][2]],
                      right=False, labels=[b[0] for b in BUCKETS])
print(f"\nfalls of 20% or deeper from a running high: {len(ep):,} episodes on "
      f"{ep['entity_id'].nunique()} companies\n")


def share_back(frame, h):
    """Share back above the old high within h months. An episode still open on a
    company still trading, watched for less than h months, is not yet countable."""
    hit = frame["recovered"] & (frame["months"] <= h)
    countable = hit | ~frame["trading"] | (frame["watched"] >= h)
    n = int(countable.sum())
    return int(hit.sum()), n, (hit.sum() / n * 100 if n else np.nan)


print("fall      episodes   within 1yr    within 2yr    within 5yr   median  never")
for lab, _, _ in BUCKETS:
    s = ep[ep["bucket"] == lab]
    cells = [f"{share_back(s, h)[2]:5.1f}% ({share_back(s, h)[1]:4d})"
             for h in (12, 24, 60)]
    med = s.loc[s["recovered"], "months"].median()
    never = (~s["recovered"]).mean() * 100
    print(f"{lab:<9} {len(s):8,}  " + "  ".join(cells) + f"  {med:6.0f}  {never:4.1f}%")
print("counts in brackets are the episodes countable at that horizon; "
      "median months is measured from the low, over recoveries only")

print("\nwhere every episode stands at 2026-07")
print("fall      episodes   back above the old high   below it, still trading   "
      "series ends first")
for lab, _, _ in BUCKETS:
    s = ep[ep["bucket"] == lab]
    r = int(s["recovered"].sum())
    u = int((~s["recovered"] & s["trading"]).sum())
    g = int((~s["recovered"] & ~s["trading"]).sum())
    print(f"{lab:<9} {len(s):8,}   {r:9,} ({r/len(s)*100:4.1f}%)          "
          f"{u:8,} ({u/len(s)*100:4.1f}%)         {g:6,} ({g/len(s)*100:4.1f}%)")

deep = ep[ep["depth"] >= 0.50]
print(f"\nfalls of 50% or more: {len(deep):,} episodes on "
      f"{deep['entity_id'].nunique()} companies")
for h in (12, 24, 60, 120):
    k, n, p = share_back(deep, h)
    unit = "year " if h == 12 else "years"
    print(f"  back above the old high within {h // 12:2d} {unit}: {p:5.1f}%  ({k}/{n})")
print(f"  median months from the low, over recoveries only: "
      f"{deep.loc[deep['recovered'], 'months'].median():.0f}")
print(f"  never got back: {int((~deep['recovered']).sum()):,} of {len(deep):,} "
      f"({(~deep['recovered']).mean()*100:.1f}%)")

view = ep.copy()
view["fall"] = (view["depth"] * 100).round(1)
head = ["company", "sym", "peak", "low", "back", "fall %", "months"]
cols = ["name", "ticker", "peak_m", "low_m", "rec_m", "fall", "months"]
print("\nlongest waits from the low back to the old high")
print(view[view["recovered"]].nlargest(8, "months")[cols]
      .to_string(index=False, header=head))
print("\ndeepest falls on companies whose price series ends before recovery")
print(view[~view["recovered"] & ~view["trading"]].nlargest(8, "fall")
      [["name", "ticker", "peak_m", "low_m", "fall", "last_m"]]
      .to_string(index=False, header=["company", "sym", "peak", "low", "fall %",
                                      "last month"]))

# ------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
shades = ["#bfdbfe", "#60a5fa", "#3b82f6", "#1d4ed8"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

horizons = np.arange(0, 121)
for (lab, _, _), col in zip(BUCKETS, shades):
    s = ep[ep["bucket"] == lab]
    ax1.plot(horizons, [share_back(s, h)[2] for h in horizons],
             color=col, lw=2, label=f"fell {lab}")
ax1.set_xlabel("Months since the low")
ax1.set_ylabel("Back above the old high (%)")
ax1.set_title("How long large-cap stocks take to recover a fall, 1996-2026")
ax1.set_xlim(0, 120)
ax1.set_ylim(0, 100)
ax1.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
ax1.spines[["top", "right"]].set_visible(False)

labels = [b[0] for b in BUCKETS]
parts, left = {}, np.zeros(len(labels))
for lab in labels:
    s = ep[ep["bucket"] == lab]
    p12, p24, p60 = (share_back(s, h)[2] for h in (12, 24, 60))
    for key, val in [("within 1 year", p12), ("1 to 2 years", p24 - p12),
                     ("2 to 5 years", p60 - p24), ("still down after 5 years",
                                                   100 - p60)]:
        parts.setdefault(key, []).append(val)
for (key, vals), col in zip(parts.items(), shades[:3] + ["#4b5563"]):
    ax2.barh(labels, vals, left=left, color=col, label=key, height=0.6)
    left += np.array(vals)
ax2.set_xlabel("Share of falls (%)")
ax2.set_ylabel("Size of the fall")
ax2.set_xlim(0, 100)
ax2.invert_yaxis()
ax2.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart saved to {SLUG}.png")
