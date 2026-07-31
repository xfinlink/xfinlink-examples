# Full write-up: https://xfinlink.com/blog/macro-factor-betas-growth-inflation-python
"""Macro factor betas: which assets respond to growth and inflation shocks.

Growth factor    = weekly return(DBB) - weekly return(GLD)   (industrial metals vs gold)
Inflation factor = weekly return(TIP) - weekly return(IEF)   (breakeven inflation proxy)

Each asset's weekly total return is regressed on both factors with
Newey-West (HAC) standard errors.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START = "2015-01-01"
FACTOR_LEGS = ["DBB", "GLD", "TIP", "IEF"]
ASSETS = ["SPY", "IWM", "EFA", "EEM", "XLE", "XLF", "XLI", "XLK",
          "XLP", "XLU", "XLY", "VNQ", "TLT", "LQD", "HYG"]
LABELS = {
    "SPY": "S&P 500", "IWM": "US small cap", "EFA": "Developed ex-US",
    "EEM": "Emerging markets", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Staples",
    "XLU": "Utilities", "XLY": "Discretionary", "VNQ": "REITs",
    "TLT": "Long Treasuries", "LQD": "IG credit", "HYG": "High yield",
}

# ---------------------------------------------------------------- data
px = pd.concat(
    [xfl.prices(t, start=START, fields=["return_daily"]) for t in FACTOR_LEGS + ASSETS]
)
daily = px.pivot_table(index="date", columns="ticker", values="return_daily")

# keep weeks with complete daily coverage across every series
obs = daily.notna().resample("W-FRI").sum()
complete = obs.eq(obs.max(axis=1), axis=0).all(axis=1)
weekly = ((1 + daily.fillna(0)).resample("W-FRI").prod() - 1)[complete].iloc[1:]

growth = weekly["DBB"] - weekly["GLD"]
inflation = weekly["TIP"] - weekly["IEF"]


def betas(y, g, i):
    x = sm.add_constant(pd.DataFrame({"growth": g, "inflation": i}))
    m = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return m


# ------------------------------------------------------- full sample
rows = []
for a in ASSETS:
    m = betas(weekly[a], growth, inflation)
    rows.append({
        "asset": a,
        "b_growth": m.params["growth"], "t_growth": m.tvalues["growth"],
        "b_infl": m.params["inflation"], "t_infl": m.tvalues["inflation"],
        "r2": m.rsquared,
    })
res = pd.DataFrame(rows).set_index("asset").sort_values("b_infl", ascending=False)

# ----------------------------------------------------- sub-periods
PERIODS = [("2015-2020", slice("2015", "2020")), ("2021-2026", slice("2021", "2026"))]
sub = {}
for label, yrs in PERIODS:
    w = weekly.loc[yrs]
    g, i = w["DBB"] - w["GLD"], w["TIP"] - w["IEF"]
    fits = {a: betas(w[a], g, i) for a in ASSETS}
    sub[label] = pd.DataFrame({
        "growth": {a: f.params["growth"] for a, f in fits.items()},
        "inflation": {a: f.params["inflation"] for a, f in fits.items()},
    })

# ---------------------------------------------------------- output
sd_g, sd_i = growth.std(), inflation.std()
print(f"Sample: {weekly.index[0]:%Y-%m-%d} to {weekly.index[-1]:%Y-%m-%d}  ({len(weekly)} weeks)")
print(f"Growth factor    DBB-GLD   annualised vol {sd_g * np.sqrt(52) * 100:5.1f}%")
print(f"Inflation factor TIP-IEF   annualised vol {sd_i * np.sqrt(52) * 100:5.1f}%")
print(f"Factor correlation: {np.corrcoef(growth, inflation)[0, 1]:+.2f}\n")

print(f"{'':22}{'growth beta':>14}{'t':>7}{'inflation beta':>17}{'t':>7}{'R2':>7}")
for a, r in res.iterrows():
    print(f"{a:<6}{LABELS[a]:<16}{r.b_growth:>14.3f}{r.t_growth:>7.1f}"
          f"{r.b_infl:>17.3f}{r.t_infl:>7.1f}{r.r2:>7.2f}")

p1, p2 = sub["2015-2020"], sub["2021-2026"]
print()
for label, yrs in PERIODS:
    w = weekly.loc[yrs]
    i = w["TIP"] - w["IEF"]
    print(f"{label}: {len(w):>3} weeks   inflation factor vol {i.std() * np.sqrt(52) * 100:4.1f}%"
          f"   corr(SPY, inflation factor) {np.corrcoef(w['SPY'], i)[0, 1]:+.2f}")

print(f"\n{'':22}{'growth beta':>21}{'inflation beta':>25}")
print(f"{'':22}{'2015-2020':>10}{'2021-2026':>11}{'2015-2020':>13}{'2021-2026':>12}{'change':>10}")
for a in res.index:
    print(f"{a:<6}{LABELS[a]:<16}{p1.loc[a, 'growth']:>10.2f}{p2.loc[a, 'growth']:>11.2f}"
          f"{p1.loc[a, 'inflation']:>13.2f}{p2.loc[a, 'inflation']:>12.2f}"
          f"{p2.loc[a, 'inflation'] - p1.loc[a, 'inflation']:>10.2f}")

# ----------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
NUDGE = {"XLK": (7, 7), "XLY": (7, -11), "IWM": (7, 6), "XLF": (7, 3),
         "EEM": (7, -11), "SPY": (7, 4), "EFA": (7, -11)}
fig, ax = plt.subplots(figsize=(10, 7))
ax.axhline(0, color="#555555", lw=0.8)
ax.axvline(0, color="#555555", lw=0.8)
for a in ASSETS:
    x1, y1 = p1.loc[a, "growth"], p1.loc[a, "inflation"]
    x2, y2 = p2.loc[a, "growth"], p2.loc[a, "inflation"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.2, alpha=0.7))
    ax.scatter([x1], [y1], s=26, color="#3b82f6", alpha=0.45, zorder=3)
    ax.scatter([x2], [y2], s=60, color="#3b82f6", zorder=3)
    ax.annotate(a, (x2, y2), textcoords="offset points",
                xytext=NUDGE.get(a, (7, -3)), color="#e0e0e0", fontsize=9)
ax.text(0.015, 0.03, "arrow runs from the 2015-2020 estimate to the 2021-2026 estimate",
        transform=ax.transAxes, color="#9ca3af", fontsize=9)
ax.set_xlabel("Sensitivity to a growth shock (industrial metals minus gold)")
ax.set_ylabel("Sensitivity to an inflation shock (inflation-linked minus nominal Treasuries)")
ax.set_title("Macro factor betas before and after 2021")
plt.tight_layout()
plt.savefig("macro-factor-betas-growth-inflation-python.png", dpi=150)
