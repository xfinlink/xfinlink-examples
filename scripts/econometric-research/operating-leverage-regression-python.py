# Full write-up: https://xfinlink.com/blog/operating-leverage-regression-python
"""Operating leverage: how far operating income moves when revenue moves 1%.

For each company, regress the annual percentage change in operating income on
the annual percentage change in revenue across FY2015-FY2025 (ten paired
observations). The slope is the degree of operating leverage. The estimated
slopes are then compared with the size of each company's annual earnings swings
and with the realised volatility of its daily stock returns over the same
window.
"""

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST_FY, LAST_FY = 2015, 2025
PX_START, PX_END = "2015-01-01", "2025-12-31"
MIN_MARGIN = 0.03
OUT_PNG = "operating-leverage-regression-python.png"

# ---------------------------------------------------------------- universe --
roster = xfl.index("sp500")
roster_tickers = sorted(roster["ticker"].dropna().unique())

frames = []
for i in range(0, len(roster_tickers), 100):
    frames.append(
        xfl.fundamentals(roster_tickers[i:i + 100], period_type="annual",
                         start="2014-06-01", end="2026-06-30",
                         fields=["revenue", "operating_income"], max_rows=20000)
    )
fund = pd.concat(frames, ignore_index=True)

fund = fund[~fund["gics_sector"].isin(["Financials", "Real Estate"])]
fund = fund[fund["fiscal_year"].between(FIRST_FY, LAST_FY)]
fund = fund.dropna(subset=["revenue", "operating_income"])
fund = fund.drop_duplicates(subset=["ticker", "fiscal_year"], keep="last")
fund = fund[fund["ticker"].isin(roster_tickers)]

qualified = {}
for ticker, g in fund.groupby("ticker"):
    g = g.sort_values("fiscal_year")
    if len(g) != (LAST_FY - FIRST_FY + 1):
        continue
    # a near-zero or sign-flipping profit base makes a percentage change
    # meaningless, so require a positive and non-trivial operating margin
    # in every year of the window
    if g["revenue"].min() <= 0 or g["operating_income"].min() <= 0:
        continue
    if (g["operating_income"] / g["revenue"]).min() < MIN_MARGIN:
        continue
    qualified[ticker] = g

print(f"companies passing the fundamental screens: {len(qualified)}")

# ------------------------------------------------------- regression + vol --
rows = []
for ticker, g in qualified.items():
    d_rev = g["revenue"].pct_change().dropna().to_numpy() * 100
    d_opi = g["operating_income"].pct_change().dropna().to_numpy() * 100
    fit = stats.linregress(d_rev, d_opi)

    px = xfl.prices(ticker, start=PX_START, end=PX_END, fields=["return_daily"])
    px = px[px["ticker"] == ticker].dropna(subset=["return_daily"])
    if len(px) < 2500:
        continue

    rows.append({
        "ticker": ticker,
        "sector": g["gics_sector"].iloc[-1],
        "slope": fit.slope,
        "r2": fit.rvalue ** 2,
        "pval": fit.pvalue,
        "opinc_sd": d_opi.std(ddof=1),
        "vol": px["return_daily"].std(ddof=1) * np.sqrt(252) * 100,
    })

res = pd.DataFrame(rows).sort_values("slope", ascending=False).reset_index(drop=True)
sig = res[res["pval"] < 0.05]

rho_e, p_e = stats.spearmanr(res["slope"], res["opinc_sd"])
rho_v, p_v = stats.spearmanr(res["slope"], res["vol"])
res["quartile"] = pd.qcut(res["slope"], 4,
                          labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
qtab = res.groupby("quartile", observed=True).agg(
    n=("ticker", "size"), slope=("slope", "mean"),
    opinc_sd=("opinc_sd", "mean"), vol=("vol", "mean"))

# ------------------------------------------------------------------ output --
SEP = "-" * 74
print(SEP)
print("OPERATING LEVERAGE, FY2015-FY2025")
print("slope = percent change in operating income per 1 percent change in revenue")
print(SEP)
print(f"sample: {len(res)} current S&P 500 members outside Financials and Real Estate")
print(f"        with 11 consecutive annual filings, operating income positive and")
print(f"        operating margin at least {MIN_MARGIN:.0%} in every year")
print(f"slopes distinguishable from zero at the 5% level: {len(sig)} of {len(res)}")
print()

hdr = f"{'ticker':<7}{'sector':<24}{'slope':>7}{'R2':>6}{'op inc sd':>11}{'stock vol':>11}"


def block(title, frame):
    print(title)
    print(hdr)
    for _, r in frame.iterrows():
        print(f"{r['ticker']:<7}{r['sector'][:23]:<24}{r['slope']:>7.2f}"
              f"{r['r2']:>6.2f}{r['opinc_sd']:>10.1f}%{r['vol']:>10.1f}%")
    print()


block("HIGHEST SLOPES (5% significant only)", sig.head(10))
block("LOWEST SLOPES (5% significant only)", sig.tail(10).iloc[::-1])

print(f"median slope, all {len(res)} companies:  {res['slope'].median():.2f}")
print(f"median slope, {len(sig)} significant:      {sig['slope'].median():.2f}")
print(f"significant slopes above 1.0: {(sig['slope'] > 1).sum()} of {len(sig)}"
      f"   below 0: {(sig['slope'] < 0).sum()} of {len(sig)}")
print()
print("Spearman rank correlation with the slope")
print(f"  volatility of annual operating-income growth: rho = {rho_e:+.3f}, p = {p_e:.1e}")
print(f"  annualised stock volatility:                  rho = {rho_v:+.3f}, p = {p_v:.1e}")
print()

print("AVERAGES BY OPERATING-LEVERAGE QUARTILE")
print(f"{'quartile':<12}{'n':>4}{'mean slope':>12}{'op inc sd':>12}{'stock vol':>12}")
for q, r in qtab.iterrows():
    print(f"{str(q):<12}{int(r['n']):>4}{r['slope']:>12.2f}"
          f"{r['opinc_sd']:>11.1f}%{r['vol']:>11.1f}%")
print()

print("MEDIAN SLOPE BY SECTOR")
sec = res.groupby("sector")["slope"].agg(["size", "median"]).sort_values(
    "median", ascending=False)
for name, r in sec.iterrows():
    print(f"{name:<24}{int(r['size']):>4}{r['median']:>8.2f}")

# ------------------------------------------------------------------- chart --
BG, FG, ACCENT = "#0a0a0a", "#e0e0e0", "#3b82f6"
MUTED = "#6b7280"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.edgecolor": "#333333", "font.size": 9,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

ends = pd.concat([sig.head(10), sig.tail(10)]).iloc[::-1]
ax1.barh(ends["ticker"], ends["slope"],
         color=[ACCENT if s > 1 else MUTED for s in ends["slope"]])
ax1.axvline(1.0, color=FG, linewidth=0.8, linestyle="--")
ax1.set_xlabel("Change in operating income per\n1% change in revenue")
ax1.set_title("Ten highest and ten lowest slopes", fontsize=10, color=FG)
ax1.tick_params(labelsize=8)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

x = np.arange(len(qtab))
w = 0.38
ax2.bar(x - w / 2, qtab["opinc_sd"], w, color=ACCENT, label="Operating income")
ax2.bar(x + w / 2, qtab["vol"], w, color=MUTED, label="Share price")
ax2.set_xticks(x)
ax2.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"], fontsize=8)
ax2.set_xlabel("Operating leverage quartile")
ax2.set_ylabel("Annual variability (%)")
ax2.set_title("Where the leverage shows up", fontsize=10, color=FG)
ax2.legend(frameon=False, fontsize=8, labelcolor=FG)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

fig.suptitle("Operating leverage across large US companies, FY2015-FY2025",
             color=FG, fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nchart saved to {OUT_PNG}")
