# Full write-up: https://xfinlink.com/blog/reverse-dcf-implied-growth-rate-python
#
# Reverse DCF: solve for the free-cash-flow growth rate that the current
# market price already implies, then compare it against realised growth.
#
# Model is EQUITY-side throughout: market capitalisation is discounted against
# levered free cash flow (operating cash flow - capital expenditure, which is
# already after interest paid) at a cost of equity. Enterprise value and
# unlevered cash flow are never mixed in.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["NVDA", "GOOGL", "MSFT", "AAPL", "META", "MA", "COST", "JNJ", "PG", "PEP"]
YEARS = 10          # explicit forecast horizon
TERMINAL_G = 0.025  # perpetual growth after year 10
RATES = [0.08, 0.10, 0.12]
BASE_R = 0.10
AS_OF = "2026-07-16"  # pinned valuation date, so the run is reproducible

# --- the model ------------------------------------------------------------
# Two-stage equity DCF. Base free cash flow F0 grows at g for 10 years, then
# at TERMINAL_G forever. Because TERMINAL_G is fixed below every discount rate
# used, the Gordon terminal term never divides by zero or flips negative --
# the model stays well defined even when implied g exceeds the discount rate.

def present_value(g, f0, r):
    t = np.arange(1, YEARS + 1)
    explicit = np.sum(f0 * (1 + g) ** t / (1 + r) ** t)
    terminal = f0 * (1 + g) ** YEARS * (1 + TERMINAL_G) / ((r - TERMINAL_G) * (1 + r) ** YEARS)
    return explicit + terminal


def implied_growth(market_cap, f0, r, lo=-0.50, hi=1.00):
    """Solve present_value(g) = market_cap for g. Returns NaN if no root in bracket."""
    fn = lambda g: present_value(g, f0, r) - market_cap
    if fn(lo) > 0 or fn(hi) < 0:
        return np.nan
    return brentq(fn, lo, hi, xtol=1e-8)


# --- data -----------------------------------------------------------------
# Base free cash flow = trailing twelve months (sum of the last 4 reported
# quarters). Guard: the same fiscal quarter can arrive under two period_end
# dates for firms with off-calendar fiscal years, so de-duplicate before summing.
q = xfl.fundamentals(TICKERS, period_type="quarterly", start="2023-01-01",
                     fields=["free_cash_flow"])
q = q.drop_duplicates(subset=["ticker", "fiscal_year", "fiscal_period"], keep="first")
q = q.sort_values(["ticker", "period_end"])
ttm_fcf = q.groupby("ticker").apply(lambda d: d.tail(4)["free_cash_flow"].sum(),
                                    include_groups=False)

# Realised growth = slope of a line fitted to log annual free cash flow.
# A log-linear trend over ~7 years is robust to a single spiky year in a way
# that an endpoint-to-endpoint CAGR is not.
ann = xfl.fundamentals(TICKERS, period_type="annual", start="2015-01-01",
                       fields=["free_cash_flow"])
ann = ann.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
ann = ann.sort_values(["ticker", "fiscal_year"])


def trend_growth(d, n=7):
    d = d.dropna(subset=["free_cash_flow"]).tail(n)
    d = d[d["free_cash_flow"] > 0]
    if len(d) < 5:
        return np.nan
    slope = np.polyfit(d["fiscal_year"].astype(float), np.log(d["free_cash_flow"]), 1)[0]
    return np.exp(slope) - 1


realised = {t: trend_growth(ann[ann["ticker"] == t]) for t in ann["ticker"].unique()}

# robustness base: mean of the last 3 annual free cash flow figures
avg3_fcf = ann.groupby("ticker").apply(
    lambda d: d.dropna(subset=["free_cash_flow"]).tail(3)["free_cash_flow"].mean(),
    include_groups=False)

# end= pins the valuation date; without it .last() silently rolls forward
mkt = xfl.metrics(TICKERS, period_type="daily", fields=["market_cap"],
                  start=AS_OF, end=AS_OF)
market_cap = mkt.sort_values("period_end").groupby("ticker")["market_cap"].last()

# --- solve ----------------------------------------------------------------
rows = []
for t in ttm_fcf.index:
    f0, cap = ttm_fcf[t], market_cap[t]
    row = {"ticker": t, "mcap_B": cap / 1e3, "ttm_fcf_B": f0 / 1e3,
           "fcf_yield": 100 * f0 / cap, "realised": realised[t],
           # same solve, but off a 3-year average base instead of TTM
           "g_avg3": implied_growth(cap, avg3_fcf[t], BASE_R)}
    for r in RATES:
        row[f"g_{int(r*100)}"] = implied_growth(cap, f0, r)
    rows.append(row)

df = pd.DataFrame(rows)
df["gap"] = df[f"g_{int(BASE_R*100)}"] - df["realised"]
df = df.sort_values(f"g_{int(BASE_R*100)}", ascending=False).reset_index(drop=True)

pct = lambda x: "no soln" if pd.isna(x) else f"{100*x:6.1f}%"
print(f"Reverse DCF | {YEARS}y explicit + {TERMINAL_G:.1%} terminal | equity-side (market cap vs levered FCF)")
print(f"{'':6} {'MktCap$B':>9} {'TTM FCF$B':>10} {'FCFyld':>7} "
      f"{'g@8%':>8} {'g@10%':>8} {'g@12%':>8} {'realised':>9} {'gap':>8}")
for _, r in df.iterrows():
    print(f"{r['ticker']:6} {r['mcap_B']:9,.0f} {r['ttm_fcf_B']:10,.1f} {r['fcf_yield']:6.2f}% "
          f"{pct(r['g_8']):>8} {pct(r['g_10']):>8} {pct(r['g_12']):>8} "
          f"{pct(r['realised']):>9} {pct(r['gap']):>8}")

spread = df["g_10"].max() - df["g_10"].min()
print(f"\nImplied g @10% spread: {100*spread:.1f}pp  "
      f"| rank corr g@8% vs g@12%: {df['g_8'].corr(df['g_12'], method='spearman'):.3f}")
# robustness: does the ranking survive a different free cash flow base?
print(f"Rank corr TTM base vs 3y-average base (both @10%): "
      f"{df['g_10'].corr(df['g_avg3'], method='spearman'):.3f}")

# --- chart ----------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
d = df.iloc[::-1].reset_index(drop=True)
y = np.arange(len(d))

# left: implied vs realised
for i, r in d.iterrows():
    ax1.plot([100 * r["realised"], 100 * r["g_10"]], [i, i], color="#3a3a3a", lw=1.4, zorder=1)
ax1.scatter(100 * d["realised"], y, s=52, color="#6b7280", zorder=2, label="Realised (7y trend)")
ax1.scatter(100 * d["g_10"], y, s=52, color="#3b82f6", zorder=3, label="Implied by price")
ax1.set_yticks(y); ax1.set_yticklabels(d["ticker"])
ax1.set_xlabel("Annual free cash flow growth (%)")
ax1.set_title("Implied vs realised growth", color="#e0e0e0", fontsize=10)
ax1.legend(frameon=False, loc="lower right", fontsize=8)
ax1.axvline(0, color="#3a3a3a", lw=0.8)
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# right: discount rate sensitivity
for r_, c, mk in zip(RATES, ["#93c5fd", "#3b82f6", "#1e40af"], ["o", "o", "o"]):
    ax2.scatter(100 * d[f"g_{int(r_*100)}"], y, s=46, color=c, marker=mk, label=f"r = {r_:.0%}")
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel("Implied growth (%)")
ax2.set_title("Sensitivity to discount rate", color="#e0e0e0", fontsize=10)
ax2.legend(frameon=False, loc="lower right", fontsize=8)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("What growth rate is the market pricing in? Reverse DCF",
             color="#e0e0e0", fontsize=12)
plt.tight_layout()
plt.savefig("worker/src/site/blog-images/reverse-dcf-implied-growth-rate-python.png",
            dpi=150, facecolor="#0a0a0a")
