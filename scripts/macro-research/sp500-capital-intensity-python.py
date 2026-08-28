# Full write-up: https://xfinlink.com/blog/sp500-capital-intensity-python
"""
Is the S&P 500 getting more capital intensive?

Capital intensity is capital expenditure measured against revenue. This script
tracks it across point-in-time S&P 500 membership from fiscal 2012 to fiscal
2025, separating what the index does in aggregate from what the median member
does, and measures how concentrated capital spending has become.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

SLUG = "sp500-capital-intensity-python"
# Financials and Real Estate are excluded: capital expenditure against revenue
# does not describe a balance-sheet-driven business the same way.
EXCLUDE = {"Financials", "Real Estate"}


def fetch(fn, **kw):
    for attempt in range(3):
        try:
            return fn(**kw)
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(8)
    return None


# 1. The index as it stood at each year end.
rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
           for y in range(2012, 2026)}
universe = sorted(set().union(*rosters.values()))
print(f"point-in-time universe: {len(universe)} entities")

parts = []
for i in range(0, len(universe), 80):
    got = fetch(xfl.fundamentals, entity_id=universe[i:i + 80], start="2011-01-01",
                end="2026-08-28", period_type="annual", max_rows=60000,
                fields=["revenue", "capital_expenditures", "total_assets"])
    if got is not None:
        parts.append(got)
fund = pd.concat(parts, ignore_index=True)

# 2. One row per company-year, index members only, real revenue only.
fund = fund.sort_values(["entity_id", "period_end"])
fund = fund.drop_duplicates(subset=["entity_id", "fiscal_year"], keep="last")
fund = fund[fund["fiscal_year"].between(2012, 2025)]
fund = fund.dropna(subset=["revenue", "capital_expenditures", "gics_sector"])
fund = fund[fund["revenue"] > 0]
fund = fund[[e in rosters.get(int(y), set())
             for e, y in zip(fund["entity_id"], fund["fiscal_year"])]]
panel = fund[~fund["gics_sector"].isin(EXCLUDE)].copy()
panel["intensity"] = panel["capital_expenditures"] / panel["revenue"]
print(f"panel: {len(panel):,} company-years, {panel['entity_id'].nunique()} entities")


def summarise(g):
    total = g["capital_expenditures"].sum()
    top10 = g.nlargest(10, "capital_expenditures")["capital_expenditures"].sum()
    return pd.Series({
        "companies": len(g),
        "aggregate_pct": total / g["revenue"].sum() * 100,
        "median_pct": g["intensity"].median() * 100,
        "top10_share_pct": top10 / total * 100})


by_year = panel.groupby("fiscal_year").apply(summarise, include_groups=False)
print("\ncapital expenditure as a share of revenue, S&P 500 ex Financials and Real Estate")
print(by_year.round(2).to_string())

sector = panel.groupby(["fiscal_year", "gics_sector"]).apply(
    lambda g: g["capital_expenditures"].sum() / g["revenue"].sum() * 100,
    include_groups=False).unstack()
change = (sector.loc[2025] - sector.loc[2015]).sort_values()
print("\naggregate capital intensity by sector, change fiscal 2015 to 2025 (pp)")
print(pd.DataFrame({"2015_pct": sector.loc[2015].round(2),
                    "2025_pct": sector.loc[2025].round(2),
                    "change_pp": change.round(2)}).sort_values("change_pp").to_string())

print("\nten largest capital spenders, fiscal 2025")
top = panel[panel["fiscal_year"] == 2025].nlargest(10, "capital_expenditures")
top = top.assign(pct_of_revenue=(top["intensity"] * 100).round(1))
print(top[["ticker", "gics_sector", "capital_expenditures", "revenue",
           "pct_of_revenue"]].to_string(index=False))

# 3. Constant-sample check: the same companies in both years, so the move is
#    not an artefact of which companies the panel happens to hold.
print("\nconstant-sample check (companies present in both years)")
for a, b in [(2015, 2025), (2024, 2025)]:
    both = set(panel[panel["fiscal_year"] == a]["entity_id"]) & \
           set(panel[panel["fiscal_year"] == b]["entity_id"])
    for year in (a, b):
        g = panel[(panel["fiscal_year"] == year) & (panel["entity_id"].isin(both))]
        agg = g["capital_expenditures"].sum() / g["revenue"].sum() * 100
        t10 = (g.nlargest(10, "capital_expenditures")["capital_expenditures"].sum()
               / g["capital_expenditures"].sum() * 100)
        print(f"  {a} vs {b}: FY{year}  n={len(g)}  aggregate={agg:.2f}%  "
              f"median={g['intensity'].median() * 100:.2f}%  top10={t10:.1f}%")

# 4. Chart.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
years = by_year.index.astype(int)

ax1.plot(years, by_year["aggregate_pct"], color="#3b82f6", linewidth=2,
         marker="o", markersize=4, label="Index total (all capex / all revenue)")
ax1.plot(years, by_year["median_pct"], color="#9ca3af", linewidth=2,
         marker="o", markersize=4, label="Median company")
ax1.set_ylabel("Capex as % of revenue")
ax1.set_title("S&P 500 capital intensity, fiscal 2012-2025 (ex Financials and Real Estate)")
ax1.legend(frameon=False, labelcolor="#e0e0e0")
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(years, by_year["top10_share_pct"], color="#3b82f6", linewidth=2,
         marker="o", markersize=4)
ax2.set_ylabel("Top 10 spenders'\nshare of capex (%)")
ax2.set_xlabel("Fiscal year")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart written to {SLUG}.png")
