# Full write-up: https://xfinlink.com/blog/operating-margin-stability-stock-volatility-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

universe = [
    "AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "ADBE", "CRM", "TXN", "QCOM", "ACN",
    "GOOGL", "NFLX", "DIS", "CMCSA",
    "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "KO", "PEP", "PG", "WMT", "COST", "CL", "MDLZ",
    "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "MDT", "AMGN",
    "HON", "UNP", "CAT", "DE", "LMT", "RTX", "UPS",
    "XOM", "CVX", "COP",
    "LIN", "SHW", "NEE", "DUK", "SO",
]

# annual operating margin, 2015-2024
f = xfl.fundamentals(universe, start="2015-01-01", end="2024-12-31",
                     period_type="annual", fields=["revenue", "operating_income"])
f = f[f["fiscal_year"].between(2015, 2024)].copy()
f["op_margin"] = 100 * f["operating_income"] / f["revenue"]

# daily returns, 2015-2025
px = xfl.prices(universe, start="2015-01-01", end="2025-12-31",
                fields=["close", "return_daily"], max_rows=400000)

rows = []
for t in universe:
    ft = f[f["ticker"] == t].sort_values("fiscal_year")
    pt = px[px["ticker"] == t]
    if ft["fiscal_year"].nunique() < 10:            # full decade of annual filings
        continue
    if pt["ticker"].nunique() != 1 or len(pt) < 2000:  # continuous single-label prices
        continue
    rows.append((t, pt["gics_sector"].iloc[0],
                 ft["op_margin"].std(ddof=1),
                 pt["return_daily"].std() * np.sqrt(252) * 100))

d = pd.DataFrame(rows, columns=["ticker", "sector", "margin_sd", "stock_vol"])
d = d.sort_values("margin_sd").reset_index(drop=True)

rho, p = stats.spearmanr(d["margin_sd"], d["stock_vol"])
print(f"{len(d)} non-financial large caps with 10 annual margins and continuous prices")
print("margin_sd = standard deviation of annual operating margin, 2015-2024 (pct points)")
print("stock_vol = annualised volatility of daily returns, 2015-2025 (%)")
print()
print(f"Spearman rank correlation (margin instability vs stock volatility): "
      f"rho = {rho:.3f}, p = {p:.4f}")
print()

d["tercile"] = pd.qcut(d["margin_sd"], 3, labels=["steadiest", "middle", "most variable"])
grp = d.groupby("tercile", observed=True)["stock_vol"].agg(["mean", "median", "count"])
print("average stock volatility by operating-margin-stability tercile")
print("tercile          mean vol   median vol   n")
for name, r in grp.iterrows():
    print(f"{name:14s}   {r['mean']:7.1f}%   {r['median']:8.1f}%   {int(r['count']):2d}")
print()
print("five steadiest margins:", ", ".join(d.head(5)["ticker"]))
print("five most variable    :", ", ".join(d.tail(5)["ticker"]))

# chart: margin instability vs stock volatility
BG, FG, ACC = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.scatter(d["margin_sd"], d["stock_vol"], color=ACC, s=45, alpha=0.85, zorder=3)
b, a = np.polyfit(d["margin_sd"], d["stock_vol"], 1)
xs = np.linspace(d["margin_sd"].min(), d["margin_sd"].max(), 100)
ax.plot(xs, a + b * xs, color="#f59e0b", lw=1.8, zorder=2, label="Least-squares fit")
for _, r in pd.concat([d.head(3), d.tail(3)]).iterrows():
    ax.annotate(r["ticker"], (r["margin_sd"], r["stock_vol"]),
                textcoords="offset points", xytext=(6, 4), color=FG, fontsize=9)
ax.set_xlabel("Operating-margin instability, 2015-2024 (std of annual margin, pct points)", color=FG)
ax.set_ylabel("Annualised stock volatility, 2015-2025 (%)", color=FG)
ax.set_title("Steadier operating margins go with calmer stocks", color=FG, fontsize=13)
for spine in ax.spines.values():
    spine.set_color("#3f3f46")
ax.tick_params(colors=FG)
ax.legend(facecolor=BG, edgecolor="#3f3f46", labelcolor=FG, loc="upper left")
plt.tight_layout()
plt.savefig("operating-margin-stability-stock-volatility-python.png", dpi=150, facecolor=BG)
