# Full write-up: https://xfinlink.com/blog/operating-lease-hidden-leverage-python
"""How much leverage is hidden in operating lease obligations?

Compares reported debt/EBITDA with lease-adjusted debt/EBITDA across the
current S&P 500 outside Financials and Real Estate.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIELDS = [
    "revenue", "ebitda", "total_debt", "current_portion_long_term_debt",
    "operating_lease_liabilities_current", "operating_lease_liabilities_noncurrent",
]

roster = xfl.index("sp500")
ids = roster["entity_id"].dropna().astype(int).tolist()

frames = []
for i in range(0, len(ids), 100):
    frames.append(xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                                   start="2024-09-01", fields=FIELDS))
f = pd.concat(frames, ignore_index=True)

# One row per company: the most recent annual filing.
f = f.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)
f = f[~f["gics_sector"].isin(["Financials", "Real Estate"])]
f = f[f["gics_sector"].notna()]
f = f.dropna(subset=["ebitda", "total_debt",
                     "operating_lease_liabilities_current",
                     "operating_lease_liabilities_noncurrent"])

# A leverage ratio needs a denominator that means something.
f = f[(f["ebitda"] / f["revenue"]) >= 0.05].copy()

f["lease"] = (f["operating_lease_liabilities_current"]
              + f["operating_lease_liabilities_noncurrent"])
f["borrowings"] = f["total_debt"]
f["borrowings_inc_current"] = f["total_debt"] + f["current_portion_long_term_debt"].fillna(0)

results = {}
for label, debt in [("borrowings", f["borrowings"]),
                    ("borrowings_inc_current", f["borrowings_inc_current"])]:
    reported = debt / f["ebitda"]
    adjusted = (debt + f["lease"]) / f["ebitda"]
    results[label] = pd.DataFrame({"reported": reported, "adjusted": adjusted,
                                   "delta": adjusted - reported,
                                   "lease_share": 100 * f["lease"] / (debt + f["lease"])})

base = results["borrowings"]
alt = results["borrowings_inc_current"]
f["reported"], f["adjusted"] = base["reported"], base["adjusted"]
f["delta"], f["lease_share"] = base["delta"], base["lease_share"]

print(f"Companies in sample: {len(f)}")
print(f"Median reported debt/EBITDA:        {base['reported'].median():.2f}x")
print(f"Median lease-adjusted debt/EBITDA:  {base['adjusted'].median():.2f}x")
print(f"Median increase:                    {base['delta'].median():.2f} turns")
print(f"Gain 1.00 turns or more:            {(base['delta'] >= 1).sum()} companies")
print(f"Owe more in leases than borrowings: {(f['lease'] > f['borrowings']).sum()} companies")

# Which companies cross 3.0x, under BOTH definitions of borrowings.
crossed = set(f.loc[(base["reported"] < 3) & (base["adjusted"] >= 3), "ticker"]) & \
          set(f.loc[(alt["reported"] < 3) & (alt["adjusted"] >= 3), "ticker"])
print(f"\nCross 3.0x once leases count (either debt definition): {len(crossed)}")
print(", ".join(sorted(crossed)))

sector = f.groupby("gics_sector").agg(n=("ticker", "size"),
                                      reported=("reported", "median"),
                                      adjusted=("adjusted", "median"),
                                      lease_share=("lease_share", "median"))
sector["delta"] = sector["adjusted"] - sector["reported"]
sector = sector.sort_values("delta", ascending=False)
print("\nSector medians:")
print(sector.round(2).to_string())

print("\nLargest increases:")
top = f.nlargest(10, "delta")[["ticker", "gics_sector", "borrowings", "lease",
                               "reported", "adjusted", "delta", "lease_share"]]
print(top.round(2).to_string(index=False))

# Chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
y = range(len(sector))
ax.barh([i + 0.2 for i in y], sector["reported"], height=0.4,
        color="#4b5563", label="Reported borrowings only")
ax.barh([i - 0.2 for i in y], sector["adjusted"], height=0.4,
        color="#3b82f6", label="Including operating lease obligations")
ax.set_yticks(list(y))
ax.set_yticklabels(sector.index)
ax.invert_yaxis()
ax.set_xlabel("Median debt to EBITDA (turns)")
ax.set_title("Operating leases add most to leverage in consumer sectors")
ax.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", loc="upper right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("operating-lease-hidden-leverage-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart saved")
