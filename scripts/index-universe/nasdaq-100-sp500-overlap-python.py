# Full write-up: https://xfinlink.com/blog/nasdaq-100-sp500-overlap-python
"""How much of the Nasdaq 100 is already inside the S&P 500?

Point-in-time membership for both indices at the year ends between 2005 and 2024
whose Nasdaq 100 roster resolves to exactly one hundred distinct companies, keyed
on entity identifiers rather than tickers, plus the market-cap weight the overlap
carries inside the Nasdaq 100 at the final year end.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = range(2005, 2025)
FINAL = "2024-12-31"
CHART = "nasdaq-100-sp500-overlap-python.png"


def members(index_name, as_of):
    """Point-in-time membership as a set of entity identifiers."""
    df = xfl.index(index_name, as_of=as_of)
    return set(int(x) for x in df["entity_id"].dropna())


rows = []
for year in YEARS:
    as_of = f"{year}-12-31"
    ndx, spx = members("ndx100", as_of), members("sp500", as_of)
    # keep the year ends whose roster resolves to exactly one hundred companies
    if len(ndx) == 100:
        rows.append({"year": year, "ndx": len(ndx), "both": len(ndx & spx)})

panel = pd.DataFrame(rows)
panel["outside"] = panel["ndx"] - panel["both"]

ndx_final = sorted(members("ndx100", FINAL))
spx_final = members("sp500", FINAL)
cap = xfl.metrics(entity_id=ndx_final, period_type="daily", fields=["market_cap"],
                  start=FINAL, end=FINAL, max_rows=100000)
cap["weight"] = 100 * cap["market_cap"] / cap["market_cap"].sum()
cap["in_sp500"] = cap["entity_id"].isin(spx_final)
outside = cap[~cap["in_sp500"]].sort_values("weight", ascending=False)
shared_weight = cap.loc[cap["in_sp500"], "weight"].sum()

print("Nasdaq 100 members that are also S&P 500 members, point-in-time rosters")
print(f"{'year':<6}{'NDX members':>13}{'in S&P 500':>12}{'outside':>9}")
for r in panel.itertuples():
    print(f"{r.year:<6}{r.ndx:>13}{r.both:>12}{r.outside:>9}")

print(f"\nMarket-cap weight inside the Nasdaq 100 at {FINAL}")
print(f"  {int(cap['in_sp500'].sum())} shared members   {shared_weight:.2f}% of index weight")
print(f"  {len(outside)} members held only by the Nasdaq 100   "
      f"{100 - shared_weight:.2f}% of index weight")
print(f"  aggregate market value  ${cap['market_cap'].sum() / 1e6:,.2f}tn")

print(f"\nNasdaq 100 members outside the S&P 500 at {FINAL}")
print(f"{'ticker':<8}{'company':<34}{'market cap $bn':>15}{'NDX weight':>12}")
for r in outside.itertuples():
    print(f"{r.ticker:<8}{r.entity_name[:33]:<34}{r.market_cap / 1e3:>15,.1f}{r.weight:>11.2f}%")

print("\nA blended portfolio: how much sits in securities the S&P 500 fund does not hold")
for split in (10, 20, 30, 40):
    print(f"  {100 - split}% S&P 500 fund / {split:>2}% Nasdaq 100 fund"
          f"{split * (100 - shared_weight) / 100:>8.2f}%")

# ── chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(panel["year"], panel["both"], color="#3b82f6", linewidth=2, marker="o", markersize=4)
ax1.set_title("Nasdaq 100 members that are also in the S&P 500", color="#e0e0e0", fontsize=12)
ax1.set_ylabel("Companies in both indices")
ax1.set_ylim(40, 95)
ax1.set_xticks(range(2005, 2025, 5))
ax1.grid(axis="y", color="#2a2a2a", linewidth=0.6)
ax1.set_axisbelow(True)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

order = outside.sort_values("weight")
ax2.barh(order["ticker"], order["weight"], color="#3b82f6")
ax2.set_title(f"Weight of the 16 Nasdaq 100 members held outside the S&P 500, {FINAL}",
              color="#e0e0e0", fontsize=12)
ax2.set_xlabel("Percent of Nasdaq 100 market value")
ax2.grid(axis="x", color="#2a2a2a", linewidth=0.6)
ax2.set_axisbelow(True)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
