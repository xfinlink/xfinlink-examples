# Full write-up: https://xfinlink.com/blog/variance-risk-premium-vrp-python

import xfinlink as xfl
import pandas as pd
import numpy as np
import os
from fredapi import Fred
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# -- VIX from FRED -------------------------------------------------------------
vix = fred.get_series("VIXCLS", observation_start="2018-01-01").rename("vix").dropna()
vix.index = pd.to_datetime(vix.index)

# -- SPY daily returns from xfinlink -------------------------------------------
spy_df = xfl.prices("SPY", start="2018-01-01", fields=["close", "return_daily"])
spy_df["date"] = pd.to_datetime(spy_df["date"])
spy = spy_df.set_index("date")

# -- 21-day realized vol, annualized -------------------------------------------
spy["rv_21d"] = spy["return_daily"].rolling(21).std() * np.sqrt(252) * 100

# -- Merge on date -------------------------------------------------------------
merged = pd.concat([spy["rv_21d"], vix], axis=1, join="inner").dropna()
merged["vrp"] = merged["vix"] - merged["rv_21d"]

# -- Summary statistics --------------------------------------------------------
pct_positive = (merged["vrp"] > 0).mean() * 100
print(
    f"Mean VIX: {merged['vix'].mean():.1f}%  Mean RV: {merged['rv_21d'].mean():.1f}%  "
    f"Mean VRP: {merged['vrp'].mean():+.1f}%  VRP positive: {pct_positive:.0f}% of days"
)

# -- By year -------------------------------------------------------------------
merged["year"] = merged.index.year
yearly = merged.groupby("year").agg(
    vix=("vix", "mean"), rv=("rv_21d", "mean"), vrp=("vrp", "mean")
).round(1)
print(f"\nVRP by Year:\n{yearly}")

# -- Sector realized vol (1Y) -------------------------------------------------
sector_etfs = {"Tech": "XLK", "Energy": "XLE", "ConsStaples": "XLP",
               "Financials": "XLF", "Healthcare": "XLV", "Industrials": "XLI"}
sector_df = xfl.prices(list(sector_etfs.values()), period="1y", fields=["return_daily"])
sector_vol = (
    sector_df.groupby("ticker")["return_daily"]
    .std()
    .mul(np.sqrt(252))
    .mul(100)
    .round(1)
)
print("\nSector Realized Vol (1Y):")
for name, etf in sector_etfs.items():
    if etf in sector_vol.index:
        print(f"  {name}: {sector_vol[etf]:.1f}%")

# -- Chart: VIX vs realized vol with VRP shading ------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

# Top panel: VIX and RV time series
ax1.plot(merged.index, merged["vix"], color="#dc2626", linewidth=1.2, label="VIX (Implied)", alpha=0.8)
ax1.plot(merged.index, merged["rv_21d"], color="#2563eb", linewidth=1.2, label="21d Realized Vol", alpha=0.8)
ax1.fill_between(
    merged.index, merged["rv_21d"], merged["vix"],
    where=merged["vrp"] > 0, alpha=0.15, color="#16a34a", label="VRP > 0 (sellers overpaid)",
)
ax1.fill_between(
    merged.index, merged["rv_21d"], merged["vix"],
    where=merged["vrp"] <= 0, alpha=0.15, color="#dc2626", label="VRP < 0 (sellers underpaid)",
)
ax1.set_ylabel("Volatility (%)")
ax1.set_title(
    "Variance Risk Premium: VIX vs 21-Day Realized Volatility\n"
    f"Mean VRP = {merged['vrp'].mean():+.1f}% | Positive {pct_positive:.0f}% of days",
    fontsize=13,
    fontweight="bold",
)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Bottom panel: VRP over time
ax2.bar(merged.index, merged["vrp"], width=3,
        color=["#16a34a" if v > 0 else "#dc2626" for v in merged["vrp"]], alpha=0.7)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("VRP (percentage points)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("variance-risk-premium-vrp-python.png", dpi=150, bbox_inches="tight")
plt.show()
