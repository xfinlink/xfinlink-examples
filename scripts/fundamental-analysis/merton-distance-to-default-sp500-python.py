# Full write-up: https://xfinlink.com/blog/merton-distance-to-default-sp500-python
"""Merton distance to default across the S&P 500 (ex-financials).

Equity is a call option on the firm's assets. Solving the option relation for
asset value and asset volatility gives distance to default: how many standard
deviations of asset value separate the firm from its debt.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

R = 0.0377        # 1-year Treasury par yield, 5 August 2026
T = 1.0           # forecast horizon, years
END = "2026-08-05"
SLUG = "merton-distance-to-default-sp500-python"

# ── 1. universe and inputs ───────────────────────────────────────────────
idx = xfl.index("sp500")
tickers = sorted(idx["ticker"].dropna().unique())

fun = pd.concat([xfl.fundamentals(
    tickers[i:i + 100], period_type="quarterly", start="2025-09-01",
    fields=["short_term_debt", "current_portion_long_term_debt", "long_term_debt",
            "shares_outstanding"]) for i in range(0, len(tickers), 100)], ignore_index=True)

latest = fun.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")
latest = latest[~latest["gics_sector"].isin(["Financials", "Real Estate"])]

# Filers differ in what they call long-term debt. Where a company reports the figure
# inclusive of the current portion, the noncurrent line from its own filing is used, so
# that the current portion is not counted twice in the default point. Where a company
# reports no noncurrent long-term debt for its latest quarter, it leaves the sample.
NONCURRENT = {"NRG": 21744.0,   # 10-Q, quarter ended 2026-06-30, noncurrent long-term debt and finance leases
              "VST": 17264.0}   # 10-Q, quarter ended 2026-03-31, noncurrent long-term debt and finance leases
NO_LONG_TERM_DEBT_LINE = ["ED"]

for tk, value in NONCURRENT.items():
    if tk in latest.index:
        latest.loc[tk, "long_term_debt"] = value
latest = latest.drop(index=[t for t in NO_LONG_TERM_DEBT_LINE if t in latest.index])

latest["F"] = (latest["current_portion_long_term_debt"].fillna(0)
               + 0.5 * latest["long_term_debt"])
latest = latest[latest["F"] > 0]
names = sorted(latest.index)

mcap = pd.concat([xfl.metrics(names[i:i + 40], period_type="daily", fields=["market_cap"],
                              start="2026-07-27") for i in range(0, len(names), 40)],
                 ignore_index=True)
close = pd.concat([xfl.prices(names[i:i + 40], start="2026-07-29", fields=["close"])
                   for i in range(0, len(names), 40)], ignore_index=True)

px = []
for i in range(0, len(names), 15):
    for attempt in range(3):
        try:
            px.append(xfl.prices(names[i:i + 15], period="1y", fields=["adj_close"]))
            break
        except xfl.XfinlinkError:
            time.sleep(3 * (attempt + 1))
px = pd.concat(px, ignore_index=True)
px = px[px["adj_close"] > 0].drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])

# ── 2. sample screens ────────────────────────────────────────────────────
E0 = mcap.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")["market_cap"]
last_close = close.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["close"]
sessions = px.groupby("ticker")["date"].count()
jump = px.groupby("ticker")["adj_close"].apply(lambda s: np.abs(np.diff(np.log(s.values))).max())

d = latest.join(E0.rename("E0")).join(last_close.rename("close"))
d = d.join(sessions.rename("sessions")).join(jump.rename("jump"))
n = [len(d)]
d = d[d["E0"].notna() & d["close"].notna() & d["shares_outstanding"].notna()]
n.append(len(d))
d = d[d["sessions"] >= 200]
n.append(len(d))
d = d[d["jump"] <= 0.45]                                   # structural breaks in the price series
n.append(len(d))
d = d[(d["E0"] / (d["close"] * d["shares_outstanding"]) - 1).abs() <= 0.15]
n.append(len(d))

# ── 3. Merton solution ───────────────────────────────────────────────────
def asset_value(E, F, sigma_V, r=R, t=T):
    """Invert E = V N(d1) - F exp(-rt) N(d2) for V, one V per equity observation."""
    V = E + F
    for _ in range(100):
        sq = sigma_V * np.sqrt(t)
        d1 = (np.log(V / F) + (r + 0.5 * sigma_V ** 2) * t) / sq
        gap = V * norm.cdf(d1) - F * np.exp(-r * t) * norm.cdf(d1 - sq) - E
        step = gap / np.maximum(norm.cdf(d1), 1e-8)
        V = np.maximum(V - step, E * 1.000001)
        if np.max(np.abs(step) / V) < 1e-12:
            break
    return V


def merton(E, F, r=R, t=T):
    """Iterate asset volatility to a fixed point, then read off distance to default."""
    sigma_E = np.diff(np.log(E)).std(ddof=1) * np.sqrt(252)
    sigma_V = sigma_E * E[-1] / (E[-1] + F)                # KMV starting guess
    for _ in range(300):
        V = asset_value(E, F, sigma_V, r, t)
        new = np.diff(np.log(V)).std(ddof=1) * np.sqrt(252)
        if abs(new - sigma_V) < 1e-10:
            sigma_V = new
            break
        sigma_V = new
    V = asset_value(E, F, sigma_V, r, t)
    dd = (np.log(V[-1] / F) + (r - 0.5 * sigma_V ** 2) * t) / (sigma_V * np.sqrt(t))
    return V[-1], sigma_V, sigma_E, dd, norm.cdf(-dd)


series = {t: g["adj_close"].values for t, g in px.groupby("ticker")}
rows = []
for tk, s in d.iterrows():
    E = s["E0"] * series[tk] / series[tk][-1]              # equity value path, $m
    V, sigma_V, sigma_E, dd, pdef = merton(E, s["F"])
    rows.append(dict(ticker=tk, name=s["entity_name"], sector=s["gics_sector"],
                     period_end=s["period_end"], E=E[-1], F=s["F"], V=V,
                     sigma_E=sigma_E, sigma_V=sigma_V, lev=s["F"] / V, dd=dd, pdef=pdef))

res = pd.DataFrame(rows).sort_values("dd").reset_index(drop=True)
res["dd_rank"] = res["dd"].rank().astype(int)
res["lev_rank"] = (-res["lev"]).rank().astype(int)

for tk, r_ in res.head(12)[["ticker", "dd"]].values:
    print(f"{tk:6} DD {r_:5.2f}")
print("Spearman, distance to default against leverage: "
      f"{spearmanr(res['dd'], -res['lev']).statistic:.3f}")

# ── 4. full report ───────────────────────────────────────────────────────
W = 108
print("\n" + "=" * W)
print(f"MERTON DISTANCE TO DEFAULT | S&P 500 excluding financials and real estate | as at {END}")
print("=" * W)
print(f"Equity as a one-year call on assets, strike = default point, risk-free rate {R:.2%} "
      "(1-year Treasury, 5 Aug 2026)")
print("Default point = debt due within one year + half of longer-dated debt (KMV convention)")
print(f"Asset value and asset volatility solved jointly from 1 year of daily equity values\n")
print(f"S&P 500 members outside financials and real estate with reported debt   {n[0]:>4}")
print(f"  with market capitalisation, share count and closing price            {n[1]:>4}")
print(f"  with at least 200 trading sessions in the window                     {n[2]:>4}")
print(f"  with no session-to-session move beyond 45% in log terms              {n[3]:>4}")
print(f"  market capitalisation within 15% of shares x price                   {n[4]:>4}")
print(f"Filing quarters used: {res['period_end'].min().date()} to {res['period_end'].max().date()}")

print("\nTWELVE SHORTEST DISTANCES TO DEFAULT")
print(f"{'':6}{'company':32}{'equity':>11}{'default':>10}{'equity':>8}{'asset':>7}"
      f"{'F/V':>7}{'DD':>7}{'implied':>9}{'leverage':>10}")
print(f"{'':6}{'':32}{'$m':>11}{'point $m':>10}{'vol':>8}{'vol':>7}{'':7}{'':7}{'default':>9}{'rank':>10}")
print("-" * W)
for _, r_ in res.head(12).iterrows():
    print(f"{r_['ticker']:6}{r_['name'][:31]:32}{r_['E']:>11,.0f}{r_['F']:>10,.0f}"
          f"{r_['sigma_E']:>8.2f}{r_['sigma_V']:>7.2f}{r_['lev']:>7.2f}{r_['dd']:>7.2f}"
          f"{r_['pdef']:>9.2%}{r_['lev_rank']:>10}")

print("\nTEN HEAVIEST DEBT LOADS, AND WHERE THE MODEL PUTS THEM")
print(f"{'':6}{'company':32}{'sector':26}{'F/V':>7}{'asset vol':>11}{'DD':>7}{'DD rank':>9}")
print("-" * W)
for _, r_ in res.nlargest(10, "lev").iterrows():
    print(f"{r_['ticker']:6}{r_['name'][:31]:32}{r_['sector'][:25]:26}{r_['lev']:>7.2f}"
          f"{r_['sigma_V']:>11.2f}{r_['dd']:>7.2f}{r_['dd_rank']:>9}")

print("\nSECTOR MEDIANS")
print(f"{'sector':26}{'names':>7}{'F/V':>8}{'asset vol':>11}{'DD':>8}")
print("-" * 60)
sec = res.groupby("sector").agg(n=("dd", "size"), lev=("lev", "median"),
                                vol=("sigma_V", "median"), dd=("dd", "median")).sort_values("dd")
for s_, r_ in sec.iterrows():
    print(f"{s_:26}{r_['n']:>7.0f}{r_['lev']:>8.3f}{r_['vol']:>11.3f}{r_['dd']:>8.2f}")

top_dd = set(res.nsmallest(20, "dd")["ticker"])
top_lev = set(res.nlargest(20, "lev")["ticker"])
print(f"\nRanking agreement across {len(res)} companies")
print(f"  Spearman, DD against leverage                     {spearmanr(res['dd'], -res['lev']).statistic:>6.3f}")
print(f"  Spearman, DD against asset volatility             {spearmanr(res['dd'], -res['sigma_V']).statistic:>6.3f}")
print(f"  Names in both the riskiest 20 by DD and by leverage {len(top_dd & top_lev):>4} of 20")
print(f"  Riskiest by leverage only: {', '.join(sorted(top_lev - top_dd))}")
print(f"  Riskiest by DD only:       {', '.join(sorted(top_dd - top_lev))}")
print(f"  Distance to default: median {res['dd'].median():.2f}, "
      f"5th percentile {res['dd'].quantile(0.05):.2f}, 95th {res['dd'].quantile(0.95):.2f}")

print("\nSENSITIVITY OF THE RANKING")
base = res.set_index("ticker")["dd"]
for label, kwargs, F_col in [("risk-free rate 2%", dict(r=0.02), None),
                             ("risk-free rate 6%", dict(r=0.06), None),
                             ("default point + short-term debt", {}, "short_term_debt"),
                             ("6-month estimation window", {}, "half")]:
    alt = {}
    for tk, s in d.iterrows():
        a = series[tk][-126:] if F_col == "half" else series[tk]
        E = s["E0"] * a / a[-1]
        F = s["F"] + (np.nan_to_num(s["short_term_debt"]) if F_col == "short_term_debt" else 0)
        alt[tk] = merton(E, F, **kwargs)[3]
    alt = pd.Series(alt).reindex(base.index)
    print(f"  {label:34} mean |change| in DD {np.abs(alt - base).mean():>5.2f}   "
          f"riskiest-20 held {len(set(alt.nsmallest(20).index) & top_dd):>2} of 20   "
          f"Spearman {spearmanr(alt, base).statistic:.3f}")

# ── 5. chart ─────────────────────────────────────────────────────────────
BG, FG, AC = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
ax.set_facecolor(BG)
gx = np.linspace(0.002, 0.85, 400)
gy = np.linspace(0.05, 1.0, 400)
GX, GY = np.meshgrid(gx, gy)
GDD = (-np.log(GX) + (R - 0.5 * GY ** 2) * T) / (GY * np.sqrt(T))
cs = ax.contour(GX, GY, GDD, levels=[2, 4, 8, 16], colors="#3f3f46", linewidths=1)
ax.clabel(cs, fmt=lambda v: f"DD {v:.0f}", fontsize=9, colors="#9ca3af")
sc = ax.scatter(res["lev"], res["sigma_V"], c=np.clip(res["dd"], 0, 20), cmap="viridis_r",
                s=26, alpha=0.85, edgecolors="none")
riskiest = res.head(8)
ax.scatter(riskiest["lev"], riskiest["sigma_V"], s=60, facecolors="none", edgecolors=AC, linewidths=1.4)
for _, r_ in riskiest.iterrows():
    ax.annotate(r_["ticker"], (r_["lev"], r_["sigma_V"]), textcoords="offset points",
                xytext=(7, 5), color=AC, fontsize=9)
cb = fig.colorbar(sc, ax=ax, pad=0.015)
cb.set_label("Distance to default (capped at 20)", color=FG)
cb.ax.yaxis.set_tick_params(color=FG)
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
cb.outline.set_edgecolor("#3f3f46")
ax.set_xlabel("Debt as a share of asset value (default point / assets)", color=FG)
ax.set_ylabel("Asset volatility, annualised", color=FG)
ax.set_title("How close is the S&P 500 to default? Leverage, asset volatility and distance to default",
             color=FG, fontsize=12)
ax.tick_params(colors=FG)
for sp in ax.spines.values():
    sp.set_color("#3f3f46")
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor=BG)
