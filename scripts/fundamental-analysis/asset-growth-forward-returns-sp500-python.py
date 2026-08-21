# Full write-up: https://xfinlink.com/blog/asset-growth-forward-returns-sp500-python
"""Does a fast-growing balance sheet predict weak forward returns?

Sorts point-in-time S&P 500 rosters on year-over-year total asset growth at ten
annual 30 June formation dates, then measures the forward 12-month price return
of every member. Universe, fundamentals and prices are all addressed by
entity id, so a symbol later reassigned to another company cannot enter.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

CHART = "asset-growth-forward-returns-sp500-python.png"
YEARS = list(range(2016, 2026))        # formation dates, 30 June
CHUNK = 60                             # entity ids per request
MIN_BASE = 10.0                        # $10m floor on both ends of the asset base
BIG = 0.50                             # "fast growth" threshold


def chunked(seq, n=CHUNK):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))


# ------------------------------------------------------- 1. point-in-time rosters
rosters = {y: retry(xfl.index, "sp500", as_of=f"{y}-06-30") for y in YEARS}
universe = sorted({int(e) for r in rosters.values() for e in r["entity_id"]})
print(f"Point-in-time S&P 500 rosters, 30 June {YEARS[0]} to 30 June {YEARS[-1]}")
print(f"  distinct companies that were members at least once: {len(universe)}")

# ------------------------------------------------------- 2. annual balance sheets
fun = pd.concat([retry(xfl.fundamentals, entity_id=c, period_type="annual",
                       fields=["total_assets", "revenue"],
                       start="2013-01-01", end="2025-12-31", max_rows=200000)
                 for c in chunked(universe)], ignore_index=True)
fun["period_end"] = pd.to_datetime(fun["period_end"])
fun = fun.dropna(subset=["total_assets"]).sort_values(["entity_id", "period_end"])

fun["prev_assets"] = fun.groupby("entity_id")["total_assets"].shift(1)
fun["prev_end"] = fun.groupby("entity_id")["period_end"].shift(1)
gap = (fun["period_end"] - fun["prev_end"]).dt.days
fun = fun[(gap >= 300) & (gap <= 430)
          & (fun["prev_assets"] >= MIN_BASE) & (fun["total_assets"] >= MIN_BASE)].copy()
fun["asset_growth"] = fun["total_assets"] / fun["prev_assets"] - 1.0

# ------------------------------------------------------- 3. signal at each formation date
signal = []
for year in YEARS:
    ids = {int(e) for e in rosters[year]["entity_id"]}
    known = fun[fun["entity_id"].isin(ids) & (fun["period_end"] <= f"{year - 1}-12-31")]
    latest = known.sort_values("period_end").groupby("entity_id").tail(1).copy()
    latest = latest[latest["period_end"] >= f"{year - 2}-01-01"]
    latest["form_year"] = year
    signal.append(latest[["entity_id", "ticker", "gics_sector", "form_year",
                          "asset_growth"]])
signal = pd.concat(signal, ignore_index=True)

# ------------------------------------------------------- 4. anchor prices, by entity id
anchors = {}
for year in YEARS + [YEARS[-1] + 1]:
    ids = sorted({int(e) for e in rosters[min(year, YEARS[-1])]["entity_id"]})
    px = pd.concat([retry(xfl.prices, entity_id=c, start=f"{year}-06-16",
                          end=f"{year}-07-10", fields=["adj_close"], max_rows=200000)
                    for c in chunked(ids)], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] <= f"{year}-06-30"].dropna(subset=["adj_close"])
    anchors[year] = px.sort_values("date").groupby("entity_id").tail(1) \
                      .set_index("entity_id")["adj_close"]

signal["p0"] = [anchors[y].get(e, np.nan) for e, y in
                zip(signal.entity_id, signal.form_year)]
signal["p1"] = [anchors[y + 1].get(e, np.nan) for e, y in
                zip(signal.entity_id, signal.form_year)]

# Members that left the index or stopped trading inside a holding year have no
# 30 June anchor in the following roster. Take their last close in the window.
missing = signal[signal["p1"].isna() & signal["p0"].notna()]
tails = []
for year, group in missing.groupby("form_year"):
    ids = sorted(set(group["entity_id"]))
    for c in chunked(ids, 30):
        t = retry(xfl.prices, entity_id=c, start=f"{year}-07-01",
                  end=f"{year + 1}-06-30", fields=["adj_close"], max_rows=200000)
        t["form_year"] = year
        tails.append(t)
tails = pd.concat(tails, ignore_index=True).dropna(subset=["adj_close"])
tails["date"] = pd.to_datetime(tails["date"])
last = tails.sort_values("date").groupby(["form_year", "entity_id"]).tail(1) \
            .set_index(["form_year", "entity_id"])["adj_close"]
signal["p1"] = signal["p1"].fillna(
    pd.Series([last.get((y, e), np.nan) for e, y in
               zip(signal.entity_id, signal.form_year)], index=signal.index))

# ------------------------------------------------------- 5. forward returns
d = signal.dropna(subset=["p0", "p1"]).copy()
d["raw"] = d["p1"] / d["p0"] - 1.0
d["fwd"] = d.groupby("form_year")["raw"].transform(
    lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
d["quintile"] = d.groupby("form_year")["asset_growth"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=range(1, 6)).astype(int))
d["decile"] = d.groupby("form_year")["asset_growth"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=range(1, 11)).astype(int))

print(f"  company-years with a signal and a forward return: {len(d):,}")
print(f"  forward returns winsorised at the 1st and 99th percentile of each "
      f"cohort: {(d['fwd'] != d['raw']).sum()} values adjusted")

q = d.groupby("quintile").agg(n=("fwd", "size"), growth=("asset_growth", "median"),
                              mean=("fwd", "mean"), median=("fwd", "median"))
print("\nForward 12-month return by asset-growth quintile, all cohorts pooled")
print("  quintile          median asset growth   mean return   median return      n")
names = {1: "Q1 slowest", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 fastest"}
for i, row in q.iterrows():
    print(f"  {names[i]:<16s} {row.growth * 100:15.1f}%  {row['mean'] * 100:11.2f}%  "
          f"{row['median'] * 100:13.2f}%  {int(row.n):5d}")

coh = d.groupby(["form_year", "quintile"])["fwd"].mean().unstack()
spread = coh[1] - coh[5]
print(f"  Q1 minus Q5: mean {spread.mean() * 100:+.2f} pp, "
      f"median {spread.median() * 100:+.2f} pp, "
      f"positive in {int((spread > 0).sum())} of {len(spread)} cohorts")

dec = d.groupby("decile").agg(n=("fwd", "size"), growth=("asset_growth", "median"),
                              mean=("fwd", "mean"), median=("fwd", "median"))
print("\nSame sort into deciles")
print("  decile   median asset growth   mean return   median return      n")
for i, row in dec.iterrows():
    print(f"  D{i:<7d} {row.growth * 100:14.1f}%  {row['mean'] * 100:11.2f}%  "
          f"{row['median'] * 100:13.2f}%  {int(row.n):5d}")

big, rest = d[d.asset_growth > BIG], d[d.asset_growth <= BIG]
by_year = d.groupby([d.form_year, d.asset_growth > BIG])["fwd"].mean().unstack()
gap_year = by_year[True] - by_year[False]
print(f"\nCompanies whose total assets grew more than {BIG:.0%} in one fiscal year")
print(f"  {len(big)} company-years across {big.gics_sector.nunique()} sectors")
print(f"  mean forward return {big.fwd.mean() * 100:6.2f}%   "
      f"median {big.fwd.median() * 100:6.2f}%")
print(f"  rest of the index   {rest.fwd.mean() * 100:6.2f}%   "
      f"median {rest.fwd.median() * 100:6.2f}%")
print(f"  trailed the rest of the index in {int((gap_year < 0).sum())} of "
      f"{gap_year.notna().sum()} cohorts")
print("  cohort gap, percentage points:")
for year, v in gap_year.items():
    print(f"    {year}  {v * 100:+7.2f}")

# ------------------------------------------------------- 6. chart
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor="#0a0a0a")
for ax in (ax1, ax2):
    ax.set_facecolor("#0a0a0a")
    for s in ax.spines.values():
        s.set_color("#333333")
    ax.tick_params(colors="#e0e0e0")

bars = ax1.bar(dec.index, dec["mean"] * 100, color="#3b82f6", width=0.7)
bars[9].set_color("#f87171")
ax1.axhline(d["fwd"].mean() * 100, color="#e0e0e0", lw=1, ls="--")
ax1.set_title("Forward 12-month return by asset-growth decile, S&P 500 members "
              "2016-2025", color="#e0e0e0", fontsize=12)
ax1.set_xlabel("Asset-growth decile (1 = slowest, 10 = fastest)", color="#e0e0e0")
ax1.set_ylabel("Mean return, per cent", color="#e0e0e0")
ax1.set_xticks(dec.index)

colors = ["#f87171" if v < 0 else "#3b82f6" for v in gap_year]
ax2.bar(gap_year.index, gap_year * 100, color=colors, width=0.6)
ax2.axhline(0, color="#e0e0e0", lw=1)
ax2.set_title(f"Assets up more than {BIG:.0%} in a year, minus the rest of the "
              "index, by formation year", color="#e0e0e0", fontsize=12)
ax2.set_xlabel("Formation year (30 June)", color="#e0e0e0")
ax2.set_ylabel("Return gap, percentage points", color="#e0e0e0")
ax2.set_xticks(list(gap_year.index))

plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {CHART}")
