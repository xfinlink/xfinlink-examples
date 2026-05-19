# Full write-up: https://xfinlink.com/blog/inflation-surprise-factor-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup
fred = Fred(api_key="YOUR_FRED_KEY")

# -- CPI from FRED -----------------------------------------------------------
cpi = fred.get_series("CPIAUCSL", observation_start="2018-01-01")
cpi_mom = cpi.pct_change().dropna()
cpi_mom.index = cpi_mom.index.to_period("M")
cpi_mom = cpi_mom.rename("cpi_mom")
expected = cpi_mom.rolling(12).mean().rename("expected")
surprise = (cpi_mom - expected).rename("surprise")

# -- XLE and TLT monthly returns from xfinlink ------------------------------
tickers = ["XLE", "TLT"]
df = xfl.prices(tickers, start="2018-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])
monthly = (
    df.groupby(["ticker", df["date"].dt.to_period("M")])["close"]
    .last()
    .unstack("ticker")
)
rets = monthly.pct_change().dropna()

# -- Merge and construct factor ----------------------------------------------
merged = pd.concat([rets, surprise], axis=1, join="inner").dropna()
merged["factor"] = merged["XLE"] - merged["TLT"]

# -- Classify regimes --------------------------------------------------------
merged["regime"] = pd.cut(
    merged["surprise"],
    bins=[-np.inf, -0.001, 0.001, np.inf],
    labels=["COOL", "NEUTRAL", "HOT"],
)

n_months = len(merged)
start_dt = merged.index[0]
end_dt = merged.index[-1]
print(f"Period: {start_dt} to {end_dt} ({n_months} months)\n")

# -- Regime statistics -------------------------------------------------------
for regime in ["HOT", "NEUTRAL", "COOL"]:
    sub = merged[merged["regime"] == regime]
    n = len(sub)
    ann_factor = sub["factor"].mean() * 12 * 100
    ann_xle = sub["XLE"].mean() * 12 * 100
    ann_tlt = sub["TLT"].mean() * 12 * 100
    sharpe = sub["factor"].mean() / sub["factor"].std() * np.sqrt(12) if sub["factor"].std() > 0 else 0
    print(
        f"{regime:8s} ({n:2d} months):  factor={ann_factor:+.1f}%  "
        f"XLE={ann_xle:+.1f}%  TLT={ann_tlt:+.1f}%  Sharpe={sharpe:+.2f}"
    )

# -- Full sample statistics --------------------------------------------------
ann_ret = merged["factor"].mean() * 12 * 100
ann_vol = merged["factor"].std() * np.sqrt(12) * 100
sharpe_full = ann_ret / ann_vol if ann_vol > 0 else 0
corr = merged["factor"].corr(merged["surprise"])
print(
    f"\nFull sample:  ann_return={ann_ret:+.1f}%  vol={ann_vol:.1f}%  Sharpe={sharpe_full:+.2f}"
)
print(f"Correlation with CPI surprise: {corr:+.3f}")

# -- Chart: cumulative factor return with regime shading ---------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

# Top panel: cumulative returns
cum_factor = (1 + merged["factor"]).cumprod() - 1
cum_xle = (1 + merged["XLE"]).cumprod() - 1
cum_tlt = (1 + merged["TLT"]).cumprod() - 1

dates = merged.index.to_timestamp()
ax1.plot(dates, cum_factor * 100, color="#2563eb", linewidth=2, label="Factor (XLE - TLT)")
ax1.plot(dates, cum_xle * 100, color="#16a34a", linewidth=1, alpha=0.6, label="XLE")
ax1.plot(dates, cum_tlt * 100, color="#dc2626", linewidth=1, alpha=0.6, label="TLT")

# Shade regimes
for i, (idx, row) in enumerate(merged.iterrows()):
    ts = idx.to_timestamp()
    color = "#fee2e2" if row["regime"] == "HOT" else "#dbeafe" if row["regime"] == "COOL" else None
    if color:
        ax1.axvspan(ts, ts + pd.Timedelta(days=28), alpha=0.3, color=color, linewidth=0)

ax1.set_ylabel("Cumulative Return (%)")
ax1.set_title(
    "Inflation Surprise Factor: Long XLE / Short TLT\n"
    "Red shading = HOT CPI months | Blue shading = COOL CPI months",
    fontsize=13,
    fontweight="bold",
)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color="black", linewidth=0.5)

# Bottom panel: CPI surprise
colors = []
for _, row in merged.iterrows():
    if row["regime"] == "HOT":
        colors.append("#dc2626")
    elif row["regime"] == "COOL":
        colors.append("#2563eb")
    else:
        colors.append("#9ca3af")

ax2.bar(dates, merged["surprise"] * 100, color=colors, width=20, alpha=0.8)
ax2.set_ylabel("CPI Surprise (%)")
ax2.set_xlabel("Date")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("inflation-surprise-factor-python.png", dpi=150, bbox_inches="tight")
plt.show()
