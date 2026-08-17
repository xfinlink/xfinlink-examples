# Full write-up: https://xfinlink.com/blog/dupont-roe-decomposition-sp500-python
"""Which part of the DuPont identity explains why S&P 500 returns on equity differ?

ROE = net margin x asset turnover x equity multiplier. The identity is exact, so the
cross-sectional variance of log ROE splits cleanly into the three components.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Index members are identified by their permanent entity id, not by ticker,
# so a company that has changed symbol is still picked up correctly.
members = xfl.index("sp500")
ids = members["entity_id"].dropna().astype(int).tolist()

frames = []
for i in range(0, len(ids), 100):
    frames.append(xfl.fundamentals(
        entity_id=ids[i:i + 100], period_type="annual", start="2024-01-01", end="2025-06-30",
        fields=["revenue", "net_income", "total_assets", "total_equity", "gics_sector"]))
f = pd.concat(frames, ignore_index=True)
f["period_end"] = pd.to_datetime(f["period_end"])
f = f[(f["period_end"] >= "2024-01-01") & (f["period_end"] <= "2024-12-31")]

f["margin"] = f["net_income"] / f["revenue"]
f["turnover"] = f["revenue"] / f["total_assets"]
f["leverage"] = f["total_assets"] / f["total_equity"]
f["roe"] = f["net_income"] / f["total_equity"]

d = f.dropna(subset=["margin", "turnover", "leverage", "roe", "gics_sector"])
pos = d[(d["net_income"] > 0) & (d["revenue"] > 0)
        & (d["total_equity"] / d["total_assets"] >= 0.01)].copy()
for src, dst in [("margin", "lm"), ("turnover", "lt"), ("leverage", "ll"), ("roe", "lr")]:
    pos[dst] = np.log(pos[src])


def decompose(frame, label):
    """Share of the cross-sectional variance of log ROE carried by each component."""
    var = frame["lr"].var(ddof=1)
    print(f"\n{label}  n={len(frame)}  sd(log ROE)={np.sqrt(var):.3f}")
    for col, name in [("lm", "Net margin"), ("lt", "Asset turnover"), ("ll", "Equity multiplier")]:
        print(f"  {name:<18} {frame[col].cov(frame['lr']) / var * 100:5.1f}%")


decompose(pos, "All sectors")
NON_OPERATING = ["Financials", "Real Estate", "Utilities"]
decompose(pos[~pos["gics_sector"].isin(NON_OPERATING)], "Operating companies only")
decompose(pos[pos["gics_sector"] == "Financials"], "Financials only")

med = (pos.groupby("gics_sector")
         .agg(n=("entity_id", "size"), roe=("roe", "median"), margin=("margin", "median"),
              turnover=("turnover", "median"), leverage=("leverage", "median"))
         .sort_values("roe", ascending=False))
print("\nSector medians, fiscal 2024")
print(med.round(3).to_string())

# ---- chart: the margin/turnover trade-off, with iso-return-on-assets curves ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.004, 0.95)
ax.set_ylim(0.02, 4.5)
ISO_LEVELS = [(0.02, "2%"), (0.06, "6%"), (0.18, "18%")]
for roa, lab in ISO_LEVELS:
    xs = np.linspace(0.004, 0.95, 120)
    ax.plot(xs, roa / xs, color="#4a4a4a", lw=0.9, ls="--", zorder=1)
ax.scatter(pos["margin"], pos["turnover"], s=18, c="#3b82f6", alpha=0.6, edgecolors="none", zorder=3)
for t in ["WMT", "AAPL", "KO", "NVDA", "JPM", "XOM"]:
    r = pos[pos["ticker"] == t]
    if len(r):
        ax.annotate(t, (r.iloc[0]["margin"], r.iloc[0]["turnover"]), color="#e0e0e0", fontsize=9,
                    xytext=(5, 4), textcoords="offset points", zorder=4)
ax.set_xlabel("Net profit margin (log scale)")
ax.set_ylabel("Asset turnover, revenue per dollar of assets (log scale)")
ax.set_title("Two routes to the same profit: margin against asset turnover, fiscal 2024")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout()

# Label each iso-curve at the on-screen angle of the line, which needs the
# final axes geometry, so this runs after tight_layout.
fig.canvas.draw()
for roa, lab in ISO_LEVELS:
    x0, x1 = 0.06, 0.30
    p0 = ax.transData.transform((x0, roa / x0))
    p1 = ax.transData.transform((x1, roa / x1))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    xm = np.sqrt(x0 * x1)
    ax.annotate("return on assets " + lab, xy=(xm, roa / xm), color="#8a8a8a", fontsize=8,
                ha="center", va="bottom", rotation=angle, rotation_mode="anchor", zorder=5)

plt.savefig("dupont-roe-decomposition-sp500-python.png", dpi=150)
