# Full write-up: https://xfinlink.com/blog/cash-balance-drawdown-cushion-python
"""Does cash on the balance sheet cushion a stock in a market selloff?

Sorts S&P 500 members into quintiles by cash as a share of total assets before
three market declines, then compares how far each quintile fell. Repeats the
sort with sector held fixed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Peak-to-trough declines in the S&P 500.
WINDOWS = [
    ("2018 Q4", "2018-09-20", "2018-12-24"),
    ("2020 COVID", "2020-02-19", "2020-03-23"),
    ("2022 bear", "2022-01-03", "2022-10-12"),
]
# Cash means something different for a bank or a landlord than for an operating
# company, so those two sectors sit outside the sample.
EXCLUDED_SECTORS = {"Financials", "Real Estate"}

panels = []
for name, start, end in WINDOWS:
    members = xfl.index("sp500", as_of=start)
    entity_ids = sorted(set(members["entity_id"]))

    # Only fiscal years ending at least six months before the window opens, the
    # standard reporting lag, so the sort uses figures a reader had at the time.
    lag_cut = (pd.Timestamp(start) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    fund = xfl.fundamentals(
        entity_id=entity_ids, start="2014-01-01", end=lag_cut, period_type="annual",
        fields=["cash_and_short_term_investments", "total_assets"],
    )
    fund = fund[fund["period_end"] <= pd.Timestamp(lag_cut)]
    fund = fund.dropna(subset=["cash_and_short_term_investments", "total_assets"])
    fund = fund[fund["total_assets"] > 0]
    # Take the whole latest row per company, never the latest value per column.
    fund = fund.loc[fund.groupby("entity_id")["period_end"].idxmax()]
    fund = fund[~fund["gics_sector"].isin(EXCLUDED_SECTORS)]
    fund["cash_ratio"] = fund["cash_and_short_term_investments"] / fund["total_assets"]

    # adj_close is the split-adjusted, gap-free series.
    px = xfl.prices(entity_id=fund["entity_id"].tolist(), start=start, end=end,
                    fields=["adj_close"], max_rows=500000)
    px = px[px["adj_close"] > 0].sort_values(["entity_id", "date"])
    grp = px.groupby("entity_id")["adj_close"]
    bars = grp.size()
    # Worst close of the window against the close it opened at.
    drawdown = (grp.min() / grp.first() - 1.0) * 100.0

    panel = fund.set_index("entity_id")[
        ["ticker", "entity_name", "gics_sector", "cash_ratio", "period_end"]].copy()
    panel["drawdown"] = drawdown
    panel["bars"] = bars
    # Require most of the window to be present so partial series cannot pose as
    # shallow declines.
    panel = panel[panel["bars"] >= 0.8 * bars.max()].dropna(subset=["drawdown"])
    panel["window"] = name
    panels.append(panel.reset_index())

panel = pd.concat(panels, ignore_index=True)

# Quintiles are formed inside each window so the three declines stay comparable.
panel["quintile"] = panel.groupby("window")["cash_ratio"].transform(
    lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]).astype(int))

by_window = panel.pivot_table(index="quintile", columns="window",
                              values="drawdown", aggfunc="mean")
by_window["Pooled"] = panel.groupby("quintile")["drawdown"].mean()

print("Mean drawdown (%) by cash quintile")
print(by_window.round(2).to_string())
print("\nMean cash / total assets by quintile")
print(panel.groupby("quintile")["cash_ratio"].mean().round(4).to_string())

print("\nQuintile 5 minus quintile 1")
for w in [n for n, _, _ in WINDOWS] + ["Pooled"]:
    d = panel if w == "Pooled" else panel[panel["window"] == w]
    hi, lo = d[d["quintile"] == 5]["drawdown"], d[d["quintile"] == 1]["drawdown"]
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    print(f"{w:11s} {hi.mean() - lo.mean():+6.2f}pp  t={t:+5.2f}  p={p:.4f}  n={len(hi)}/{len(lo)}")

# Hold sector fixed: strip the window-by-sector average out of both variables.
panel["dd_sn"] = panel["drawdown"] - panel.groupby(["window", "gics_sector"])["drawdown"].transform("mean")
panel["cr_sn"] = panel["cash_ratio"] - panel.groupby(["window", "gics_sector"])["cash_ratio"].transform("mean")
panel["q_sn"] = panel.groupby("window")["cr_sn"].transform(
    lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]).astype(int))

print("\nSector-neutral: drawdown relative to the window's sector average (pp)")
print(panel.groupby("q_sn")["dd_sn"].mean().round(2).to_string())
hi, lo = panel[panel["q_sn"] == 5]["dd_sn"], panel[panel["q_sn"] == 1]["dd_sn"]
t, p = stats.ttest_ind(hi, lo, equal_var=False)
print(f"Q5 minus Q1  {hi.mean() - lo.mean():+.2f}pp  t={t:+.2f}  p={p:.4f}")

print("\nSector averages, all three windows pooled")
print(panel.groupby("gics_sector")
      .agg(drawdown=("drawdown", "mean"), cash_ratio=("cash_ratio", "mean"), n=("drawdown", "size"))
      .round(3).sort_values("drawdown").to_string())

print(f"\n{len(panel)} company-window observations, {panel['entity_id'].nunique()} companies")

# ── chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
width, xs = 0.26, np.arange(5)
for i, (w, colour) in enumerate(zip([n for n, _, _ in WINDOWS], ["#6b7280", "#3b82f6", "#f59e0b"])):
    ax1.bar(xs + (i - 1) * width, by_window[w].values, width, label=w, color=colour)
ax1.set_xticks(xs, ["1\nleast", "2", "3", "4", "5\nmost"])
ax1.set_xlabel("Cash as a share of total assets, quintile")
ax1.set_ylabel("Mean fall from window open (%)")
ax1.set_title("Cash-rich stocks fell further, except in 2020")
ax1.legend(frameon=False)
ax1.axhline(0, color="#333333", lw=0.8)

ax2.bar(xs, panel.groupby("q_sn")["dd_sn"].mean().values, 0.6, color="#3b82f6")
ax2.set_xticks(xs, ["1\nleast", "2", "3", "4", "5\nmost"])
ax2.set_xlabel("Cash relative to sector, quintile")
ax2.set_ylabel("Fall vs sector average (pp)")
ax2.set_title("With sector held fixed, the gap closes")
ax2.axhline(0, color="#333333", lw=0.8)
ax2.set_ylim(-6, 6)

plt.tight_layout()
plt.savefig("cash-balance-drawdown-cushion-python.png", dpi=150, facecolor="#0a0a0a")
