# Full write-up: https://xfinlink.com/blog/do-high-accrual-companies-underperform-python
"""
Do high-accrual companies underperform? Accruals screening on the S&P 500.

Accruals are the part of reported profit that did not arrive as cash. Sloan
(1996) found that companies with high accruals go on to deliver lower stock
returns than companies whose profit is backed by cash. This script rebuilds
that sort on point-in-time S&P 500 membership from 2015 to 2024.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

SLUG = "do-high-accrual-companies-underperform-python"
FORM_YEARS = (2015, 2024)


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch(fn, **kw):
    """Three attempts, then give up on the chunk."""
    for attempt in range(3):
        try:
            return fn(**kw)
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(5)
    return None


# 1. Point-in-time membership: the roster as it stood at each year end.
rosters = {}
for year in range(2014, 2025):
    rosters[year] = set(xfl.index("sp500", as_of=f"{year}-12-31")["entity_id"])
universe = sorted(set().union(*rosters.values()))
print(f"point-in-time universe: {len(universe)} entities")

# 2. Annual fundamentals for the accruals calculation.
parts = []
for chunk in chunked(universe, 80):
    got = fetch(xfl.fundamentals, entity_id=chunk, start="2012-01-01",
                end="2026-08-01", period_type="annual", max_rows=60000,
                fields=["net_income", "operating_cash_flow", "total_assets"])
    if got is not None:
        parts.append(got)
fund = pd.concat(parts, ignore_index=True)
fund["period_end"] = pd.to_datetime(fund["period_end"])
fund = fund.sort_values(["entity_id", "period_end"])
fund = fund.drop_duplicates(subset=["entity_id", "fiscal_year"], keep="last")

# 3. Daily prices for the forward-return window.
parts = []
for chunk in chunked(universe, 50):
    got = fetch(xfl.prices, entity_id=chunk, start="2014-06-01",
                end="2026-08-28", fields=["close", "return_daily"],
                max_rows=500000)
    if got is not None:
        parts.append(got)
px = pd.concat(parts, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["entity_id", "date"])
print(f"price rows: {len(px):,}")

# 4. Accruals = (net income - operating cash flow) / average total assets.
fund["assets_lag"] = fund.groupby("entity_id")["total_assets"].shift(1)
fund["avg_assets"] = (fund["total_assets"] + fund["assets_lag"]) / 2
fund["accruals"] = (fund["net_income"] - fund["operating_cash_flow"]) / fund["avg_assets"]
fund = fund[fund["avg_assets"] > 0].dropna(subset=["accruals"])

# Portfolios form four months after the fiscal year closes, by which time the
# annual report has been filed. Membership is checked at the fiscal year end.
fund["formation"] = fund["period_end"] + pd.DateOffset(months=4)
fund["form_year"] = fund["formation"].dt.year
in_index = [eid in rosters.get(pe.year, set())
            for eid, pe in zip(fund["entity_id"], fund["period_end"])]
fund = fund[in_index]

# 5. Forward 12-month return from each formation date.
paths = {}
for eid, grp in px.groupby("entity_id"):
    logret = np.concatenate([[0.0], np.nancumsum(np.log1p(grp["return_daily"].values))])
    paths[eid] = (grp["date"].values, logret)

forward = []
for eid, start in zip(fund["entity_id"], fund["formation"]):
    if eid not in paths:
        forward.append(np.nan)
        continue
    dates, logret = paths[eid]
    i = np.searchsorted(dates, np.datetime64(start))
    j = np.searchsorted(dates, np.datetime64(start + pd.Timedelta(days=365)))
    if i >= len(dates) or j >= len(dates) or j <= i:
        forward.append(np.nan)
        continue
    forward.append(np.expm1(logret[j] - logret[i]))
fund["fwd12"] = forward

panel = fund.dropna(subset=["fwd12"])
panel = panel[panel["form_year"].between(*FORM_YEARS)].copy()

# 6. Winsorise within each formation year, then sort into quintiles.
panel["accruals_w"] = panel.groupby("form_year")["accruals"].transform(
    lambda s: s.clip(*s.quantile([0.01, 0.99])))
panel["quintile"] = panel.groupby("form_year")["accruals_w"].transform(
    lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]))

by_q = panel.groupby("quintile", observed=True).agg(
    firm_years=("fwd12", "size"),
    mean_accruals=("accruals_w", "mean"),
    mean_fwd12=("fwd12", "mean"),
    median_fwd12=("fwd12", "median"))

per_year = panel.groupby(["form_year", "quintile"], observed=True)["fwd12"].mean().unstack()
spread = per_year[1] - per_year[5]
t_stat, p_val = stats.ttest_1samp(spread.dropna(), 0)

print(f"\nfirm-years {len(panel):,}  entities {panel['entity_id'].nunique()}  "
      f"{panel['form_year'].min()}-{panel['form_year'].max()}")
print("\nforward 12-month return by accruals quintile (1 = lowest accruals)")
print((by_q * [1, 100, 100, 100]).round(2).to_string())
print(f"\nQ1 minus Q5, averaged over formation years: {spread.mean() * 100:.2f}pp")
print(f"t-statistic {t_stat:.2f}   p-value {p_val:.3f}   "
      f"positive in {int((spread > 0).sum())} of {spread.notna().sum()} years")

# 7. Chart.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([str(q) for q in by_q.index], by_q["mean_fwd12"] * 100, color="#3b82f6", width=0.62)
ax.set_xlabel("Accruals quintile (1 = lowest accruals, 5 = highest)")
ax.set_ylabel("Mean forward 12-month return (%)")
ax.set_title("Forward returns by accruals quintile, S&P 500, 2015-2024")
ax.spines[["top", "right"]].set_visible(False)
for spine, value in zip(ax.patches, by_q["mean_fwd12"] * 100):
    ax.text(spine.get_x() + spine.get_width() / 2, value + 0.25,
            f"{value:.1f}%", ha="center", color="#e0e0e0", fontsize=10)
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart written to {SLUG}.png")
