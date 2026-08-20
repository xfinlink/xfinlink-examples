# Full write-up: https://xfinlink.com/blog/share-of-stocks-beating-the-index-python
"""How many S&P 500 members beat the index itself?

Point-in-time rosters, monthly price returns, June-to-June holding years.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORMATION_YEARS = list(range(2018, 2026))
START, END = "2017-12-01", "2026-08-19"

# 1. Point-in-time membership on each 30 June formation date.
rosters = {}
for year in FORMATION_YEARS + [2026]:
    roster = xfl.index("sp500", as_of=f"{year}-06-30")
    rosters[year] = roster.dropna(subset=["ticker"])

universe = sorted({t for r in rosters.values() for t in r["ticker"]})

# 2. Monthly prices for every name that was ever a member over the window.
frames = []
for i in range(0, len(universe), 100):
    frames.append(
        xfl.prices(universe[i : i + 100], start=START, end=END, interval="1mo",
                   fields=["close", "adj_close"], max_rows=500000)
    )
px = pd.concat(frames, ignore_index=True)
spy = xfl.prices(["SPY"], start=START, end=END, interval="1mo",
                 fields=["close", "adj_close"])

# 3. Collapse to one observation per entity per calendar month, keeping the
#    last bar in each month, then index the panel on month start.
def month_end_panel(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df = df.sort_values("date").groupby(["entity_id", "month"], as_index=False).last()
    return df.pivot(index="month", columns="entity_id", values="adj_close")

panel = month_end_panel(px)
bench = month_end_panel(spy).iloc[:, 0]
symbol = px.sort_values("date").groupby("entity_id")["ticker"].last()


def window_return(series_frame, start_month, end_month):
    """Price return between two month ends, NaN where either end is missing."""
    if start_month not in series_frame.index or end_month not in series_frame.index:
        return None
    return series_frame.loc[end_month] / series_frame.loc[start_month] - 1


# 4. One holding year per formation date: 30 June to 30 June.
rows = []
for year in FORMATION_YEARS:
    m0, m1 = pd.Period(f"{year}-06", "M"), pd.Period(f"{year + 1}-06", "M")
    ids = [i for i in rosters[year]["entity_id"] if i in panel.columns]
    ret = window_return(panel[ids], m0, m1).dropna()
    spy_ret = bench.loc[m1] / bench.loc[m0] - 1
    rows.append({
        "year": f"{year}-{str(year + 1)[2:]}",
        "n": len(ret),
        "index_pct": spy_ret * 100,
        "median_pct": ret.median() * 100,
        "mean_pct": ret.mean() * 100,
        "beat_pct": (ret > spy_ret).mean() * 100,
    })
annual = pd.DataFrame(rows)

# 5. Full-period buy and hold for the June 2018 roster.
m0, m1 = pd.Period("2018-06", "M"), pd.Period("2026-06", "M")
ids_2018 = [i for i in rosters[2018]["entity_id"] if i in panel.columns]
full = window_return(panel[ids_2018], m0, m1).dropna()
spy_full = bench.loc[m1] / bench.loc[m0] - 1

ranked = full.sort_values(ascending=False)
top_decile = ranked.head(int(round(len(ranked) * 0.1)))
share_top_decile = top_decile.sum() / ranked.sum() * 100

print(f"June-to-June holding years, point-in-time S&P 500 members")
print(annual.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))
print()
print(f"Eight-year buy and hold, June 2018 roster, {len(full)} names with a full price history")
print(f"  index price return        {spy_full * 100:8.1f}%")
print(f"  member median             {full.median() * 100:8.1f}%")
print(f"  member mean               {full.mean() * 100:8.1f}%")
print(f"  beat the index            {(full > spy_full).mean() * 100:8.1f}% of names")
print(f"  best decile share of the aggregate profit {share_top_decile:5.1f}%")
print()
print("  strongest five, current symbol")
for eid, r in ranked.head(5).items():
    print(f"    {symbol.get(eid, eid):<6}{r * 100:8.0f}%")
print("  weakest five, current symbol")
for eid, r in ranked.tail(5).items():
    print(f"    {symbol.get(eid, eid):<6}{r * 100:8.0f}%")

# 6. Chart: the distribution behind the headline number.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, ax = plt.subplots(figsize=(10, 5))
clipped = np.clip(full.values * 100, -100, 600)
ax.hist(clipped, bins=40, color="#3b82f6", edgecolor="#0a0a0a", linewidth=0.6)
ax.axvline(spy_full * 100, color="#f59e0b", linewidth=2,
           label=f"index {spy_full * 100:.0f}%")
ax.axvline(full.median() * 100, color="#e0e0e0", linewidth=2, linestyle="--",
           label=f"median member {full.median() * 100:.0f}%")
ax.set_xlabel("Eight-year price return, per cent (capped at 600)")
ax.set_ylabel("Number of companies")
ax.set_title("How many S&P 500 stocks beat the index, June 2018 to June 2026")
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("share-of-stocks-beating-the-index-python.png", dpi=150)
