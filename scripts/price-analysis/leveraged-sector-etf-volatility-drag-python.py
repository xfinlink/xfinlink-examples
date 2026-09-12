# Full write-up: https://xfinlink.com/blog/leveraged-sector-etf-volatility-drag-python
"""Can 2x leveraged sector ETFs beat the sector they double?

Nine ProShares Ultra funds against their Select Sector SPDR counterparts,
2010-01-04 to 2024-12-31. Splits the multi-year shortfall into the
mechanical cost of the daily reset and everything else, then turns the
shortfall into a break-even hurdle for the sector.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

PAIRS = {"XLB": "UYM", "XLE": "DIG", "XLF": "UYG", "XLI": "UXI", "XLK": "ROM",
         "XLP": "UGE", "XLU": "UPW", "XLV": "RXL", "XLY": "UCC"}
NAME = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
        "XLI": "Industrials", "XLK": "Technology", "XLP": "Staples",
        "XLU": "Utilities", "XLV": "Health Care", "XLY": "Discretionary"}

px = xfl.prices(list(PAIRS) + list(PAIRS.values()), start="2010-01-04",
                end="2024-12-31", fields=["return_daily"], max_rows=200000)
r = px.pivot(index="date", columns="ticker", values="return_daily").dropna()
years = len(r) / 252.0

rows = []
for sec, lev in PAIRS.items():
    ru, rl = r[sec], r[lev]
    g_sec = np.log1p(ru).sum()                 # sector, buy and hold
    g_lev = np.log1p(rl).sum()                 # the live 2x fund
    g_reset = np.log1p(2 * ru).sum()           # costless daily-reset 2x
    rows.append({
        "sector": NAME[sec], "fund": lev,
        "vol": ru.std(ddof=1) * np.sqrt(252),
        "mult": np.polyfit(ru, rl, 1)[0],      # realised daily multiple
        "sec_x": np.exp(g_sec), "lev_x": np.exp(g_lev), "tgt_x": np.exp(2 * g_sec),
        "shortfall": (2 * g_sec - g_lev) / years,
        "drag": (2 * g_sec - g_reset) / years,
        "sigma2": ru.var(ddof=1) * 252,
    })
t = pd.DataFrame(rows).sort_values("vol", ascending=False).reset_index(drop=True)
t["cost"] = t["shortfall"] - t["drag"]
t["growth"] = np.log(t["sec_x"]) / years        # sector log growth per year
t["beat"] = t["growth"] > t["shortfall"]        # 2x fund ahead of the sector

print(f"{len(r)} trading days, {years:.1f} years, 2010-01-04 to 2024-12-31\n")
print(f"{'Sector':<14}{'Fund':<6}{'Vol':>6}{'Mult':>6}{'Sector':>9}"
      f"{'Fund':>9}{'Target':>10}{'Short':>8}{'Drag':>8}{'Sig2':>7}{'Cost':>7}")
for _, x in t.iterrows():
    print(f"{x.sector:<14}{x['fund']:<6}{x.vol:>5.1%}{x['mult']:>6.2f}"
          f"{x.sec_x:>8.2f}x{x.lev_x:>8.2f}x{x.tgt_x:>9.2f}x"
          f"{x.shortfall:>8.2%}{x.drag:>8.2%}{x.sigma2:>7.2%}{x.cost:>7.2%}")
print(f"\nDrag vs sigma^2: mean gap {(t.drag - t.sigma2).mean():>+.3%}, "
      f"correlation {np.corrcoef(t.drag, t.sigma2)[0, 1]:.4f}")
print(f"Cost after drag: {t.cost.min():.2%} to {t.cost.max():.2%}, "
      f"median {t.cost.median():.2%}\n")

print("Break-even: the sector must compound faster than the shortfall")
print(f"{'Sector':<14}{'Growth':>8}{'Hurdle':>9}{'Margin':>9}  Fund beat sector")
for _, x in t.sort_values("growth", ascending=False).iterrows():
    print(f"{x.sector:<14}{x.growth:>8.2%}{x.shortfall:>9.2%}"
          f"{x.growth - x.shortfall:>+9.2%}  {'yes' if x.beat else 'no'}")
print(f"{t.beat.sum()} of {len(t)} funds beat their sector over the window")

# ── chart ────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

lim = [0, max(t.sigma2.max(), t.drag.max()) * 105]
ax1.plot(lim, lim, "--", color="#6b7280", lw=1)
ax1.scatter(t.sigma2 * 100, t.drag * 100, s=55, color="#3b82f6", zorder=3)
for _, x in t.iterrows():
    ax1.annotate(x.sector, (x.sigma2 * 100, x.drag * 100), textcoords="offset points",
                 xytext=(6, -3), fontsize=7.5, color="#9ca3af")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Annual variance of the sector (%)")
ax1.set_ylabel("Measured cost of the daily reset (% per year)")
ax1.set_title("Variance predicts the reset cost", fontsize=10)

o = t.sort_values("shortfall")
y = np.arange(len(o))
ax2.barh(y, o.drag * 100, color="#3b82f6", label="Daily reset")
ax2.barh(y, o.cost * 100, left=o.drag * 100, color="#94a3b8", label="Fees and financing")
ax2.set_yticks(y); ax2.set_yticklabels(o.sector)
ax2.set_xlabel("Annual shortfall against twice the sector (%)")
ax2.set_title("Where the missing return goes", fontsize=10)
ax2.legend(frameon=False, fontsize=8, loc="lower right")
for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("leveraged-sector-etf-volatility-drag-python.png", dpi=150,
            facecolor="#0a0a0a")
