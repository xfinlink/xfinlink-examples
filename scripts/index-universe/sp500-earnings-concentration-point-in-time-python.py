# Full write-up: https://xfinlink.com/blog/sp500-earnings-concentration-point-in-time-python
"""How concentrated are S&P 500 earnings? Point-in-time index analysis in Python.

Rebuilds the S&P 500 roster for every year end from 2010 to 2025 using
survivorship-bias-free historical membership, joins each member to its annual
income statement by entity id, and measures how concentrated aggregate revenue
and aggregate profit have become.
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xfinlink as xfl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = range(2010, 2026)


def roster(as_of):
    """Entity ids of S&P 500 members on a given date (None = today)."""
    df = xfl.index("sp500", as_of=as_of).dropna(subset=["entity_id"])
    return sorted(set(df["entity_id"].astype(int)))


def fiscal_year(ids, year):
    """One annual income statement per member for fiscal year `year`."""
    frames = []
    for i in range(0, len(ids), 100):
        frames.append(
            xfl.fundamentals(
                entity_id=ids[i:i + 100],
                period_type="annual",
                start=f"{year}-06-01",
                end=f"{year + 1}-05-31",
                fields=["revenue", "net_income"],
                max_rows=100000,
            )
        )
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    f = pd.concat(frames, ignore_index=True).dropna(subset=["revenue", "net_income"])
    f = f[f["revenue"] > 0]
    return f.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)


def concentration(values):
    """Top-10 share and effective count over the positive part of a pool."""
    pos = np.sort(np.asarray(values, dtype=float)[np.asarray(values) > 0])[::-1]
    w = pos / pos.sum()
    return 100 * w[:10].sum(), 1.0 / np.sum(w ** 2), len(pos)


rows = []
snapshots = {}
for year in YEARS:
    ids = roster(f"{year}-12-31")
    f = fiscal_year(ids, year)
    rev_top10, rev_neff, _ = concentration(f["revenue"])
    ni_top10, ni_neff, earners = concentration(f["net_income"])
    rows.append(
        dict(
            year=year,
            members=len(ids),
            covered=len(f),
            rev_top10=rev_top10,
            rev_neff=rev_neff,
            ni_top10=ni_top10,
            ni_neff=ni_neff,
            earners=earners,
            pool_bn=f.loc[f["net_income"] > 0, "net_income"].sum() / 1e3,
            losers=int((f["net_income"] < 0).sum()),
        )
    )
    snapshots[year] = f

t = pd.DataFrame(rows)

print("S&P 500 earnings vs revenue concentration, point-in-time rosters")
print(f"{'FY':<5}{'members':>8}{'covered':>9}{'rev top10':>11}{'rev Neff':>10}"
      f"{'NI top10':>10}{'NI Neff':>9}{'profit pool':>13}{'loss-makers':>13}")
for r in rows:
    print(f"{r['year']:<5}{r['members']:>8}{r['covered']:>9}{r['rev_top10']:>10.1f}%"
          f"{r['rev_neff']:>10.1f}{r['ni_top10']:>9.1f}%{r['ni_neff']:>9.1f}"
          f"{r['pool_bn']:>12,.0f}b{r['losers']:>13}")

print("\nTop 10 earners, FY2010 vs FY2025 ($m)")
a = snapshots[2010].nlargest(10, "net_income").reset_index(drop=True)
b = snapshots[2025].nlargest(10, "net_income").reset_index(drop=True)
# One display label per company, taken from its most recent appearance, so a
# company holding a place in both columns is shown under a single symbol.
label = {}
for year in sorted(snapshots):
    for eid, tic in zip(snapshots[year]["entity_id"], snapshots[year]["ticker"]):
        label[eid] = tic
print(f"{'#':<3}{'FY2010':<10}{'net income':>12}   {'FY2025':<10}{'net income':>12}")
for i in range(10):
    print(f"{i + 1:<3}{label[a.loc[i, 'entity_id']]:<10}{a.loc[i, 'net_income']:>12,.0f}   "
          f"{label[b.loc[i, 'entity_id']]:<10}{b.loc[i, 'net_income']:>12,.0f}")

# Survivorship check: measure FY2010 with today's roster instead of the 2010 one.
cur = fiscal_year(roster(None), 2010)
pit = snapshots[2010]
pit_top10, pit_neff, _ = concentration(pit["net_income"])
cur_top10, cur_neff, _ = concentration(cur["net_income"])
print("\nFY2010 measured two ways")
print(f"point-in-time 2010 roster : {len(pit)} names, top-10 share {pit_top10:.1f}%, Neff {pit_neff:.1f}")
print(f"today's roster backdated  : {len(cur)} names, top-10 share {cur_top10:.1f}%, Neff {cur_neff:.1f}")

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(t["year"], t["ni_top10"], color="#3b82f6", lw=2.2, marker="o", ms=4, label="Profit")
ax1.plot(t["year"], t["rev_top10"], color="#9ca3af", lw=2.2, marker="s", ms=4, label="Revenue")
ax1.set_ylabel("Share held by 10 largest (%)")
ax1.set_title("S&P 500 profit concentration has risen; revenue concentration has not",
              color="#e0e0e0", fontsize=12, pad=12)
ax1.legend(frameon=False, loc="upper left")

ax2.plot(t["year"], t["ni_neff"], color="#3b82f6", lw=2.2, marker="o", ms=4, label="Profit")
ax2.plot(t["year"], t["rev_neff"], color="#9ca3af", lw=2.2, marker="s", ms=4, label="Revenue")
ax2.set_ylabel("Effective number of companies")
ax2.set_xlabel("Fiscal year")
ax2.legend(frameon=False, loc="lower left")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("sp500-earnings-concentration-point-in-time-python.png", dpi=150,
            facecolor="#0a0a0a")
