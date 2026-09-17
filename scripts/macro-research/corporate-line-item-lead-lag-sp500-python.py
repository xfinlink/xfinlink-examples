# Full write-up: https://xfinlink.com/blog/corporate-line-item-lead-lag-sp500-python
"""Which corporate line item turns first? Lead-lag analysis of S&P 500 fundamentals.

Builds a bottom-up quarterly growth series for seven line items from company
filings, then measures where each one's correlation with revenue growth peaks.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SERIES = {
    "revenue": "Revenue",
    "cost_of_sales": "Cost of sales",
    "selling_general_admin": "SG&A expense",
    "inventory": "Inventory",
    "accounts_receivable": "Receivables",
    "accounts_payable": "Payables",
    "capital_expenditures": "Capital spending",
}
FIELDS = ["revenue", "cost_of_revenue", "cost_of_goods_sold", "selling_general_admin",
          "inventory", "accounts_receivable", "accounts_payable", "capital_expenditures"]
QS = pd.period_range("2010Q1", "2026Q2", freq="Q")
LAGS = range(-4, 5)

# ---- data ---------------------------------------------------------------
tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique())
frames = [xfl.fundamentals(tickers[i:i + 100], start="2008-06-01", end="2026-09-17",
                           period_type="quarterly", fields=FIELDS, max_rows=200000)
          for i in range(0, len(tickers), 100)]
raw = pd.concat(frames, ignore_index=True)

# Fiscal periods end on company-specific dates, so map each report to the
# calendar quarter whose end date it sits closest to, one report per quarter.
raw["quarter"] = (raw["period_end"] + pd.Timedelta(days=45)).dt.to_period("Q") - 1
raw["cost_of_sales"] = raw["cost_of_revenue"].fillna(raw["cost_of_goods_sold"])
raw = (raw.sort_values(["ticker", "quarter", "period_end"])
          .drop_duplicates(["ticker", "quarter"], keep="last"))

# Constant panel: companies with an unbroken record on every line item, so the
# growth rates reflect the same companies in every quarter.
names = sorted(raw["ticker"].dropna().unique())
wide = {f: raw.pivot_table(index="ticker", columns="quarter", values=f, aggfunc="first")
             .reindex(index=names, columns=QS) for f in SERIES}
keep = np.logical_and.reduce([wide[f].notna().all(axis=1).values for f in SERIES])
panel = pd.Index(names)[keep]

# Median company growth, not the dollar sum: one company's reporting-basis
# change in one quarter cannot move a median.
med = pd.DataFrame({f: (wide[f].loc[panel].pct_change(4, axis=1) * 100).median(axis=0)
                    for f in SERIES}).dropna()


def profile(frame):
    """Correlation of each line item's growth with revenue growth, by shift."""
    rev = frame["revenue"]
    return pd.DataFrame({f: {L: frame[f].shift(L).corr(rev) for L in LAGS}
                         for f in SERIES if f != "revenue"}).T


full = profile(med)
ex_covid = profile(med[(med.index < "2020Q1") | (med.index > "2021Q4")])

table = pd.DataFrame([{
    "Line item": SERIES[f],
    "Peak lag": full.loc[f].idxmax(),
    "Corr at peak": round(full.loc[f].max(), 3),
    "Corr same qtr": round(full.loc[f, 0], 3),
    "Peak ex-2020/21": ex_covid.loc[f].idxmax(),
    "Latest YoY %": round(med[f].iloc[-1], 1),
} for f in full.index]).sort_values("Peak lag")

# ---- output -------------------------------------------------------------
pd.set_option("display.width", 200)
sectors = raw[raw["ticker"].isin(panel)].drop_duplicates("ticker")["gics_sector"].value_counts()
print(f"Panel: {len(panel)} companies, unbroken quarterly records {QS[0]} to {QS[-1]}")
print("Sectors: " + ", ".join(f"{k} {v}" for k, v in sectors.items()))
print(f"Growth series: {len(med)} quarters, {med.index[0]} to {med.index[-1]}")
print("\nPositive lag = turns before revenue. Negative lag = turns after revenue.")
print(table.to_string(index=False))
print("\nCross-correlation against revenue growth, by lag in quarters:")
print(full.round(2).to_string())
print("\nSame profile excluding 2020Q1-2021Q4:")
print(ex_covid.round(2).to_string())
print("\nTurning points (median year-on-year growth):")
for f in ["revenue", "inventory", "capital_expenditures"]:
    up, dn = med[f].loc["2020Q1":"2023Q4"], med[f].loc["2022Q1":"2025Q4"]
    print(f"  {SERIES[f]:<17} cycle peak {up.idxmax()} at {up.max():5.1f}%   "
          f"trough {dn.idxmin()} at {dn.min():5.1f}%")
print(f"\nLatest quarter {med.index[-1]}: " +
      ", ".join(f"{SERIES[f]} {med[f].iloc[-1]:+.1f}%" for f in SERIES))

# ---- chart --------------------------------------------------------------
BG, FG, AC = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1, 1.2]})
fig.patch.set_facecolor(BG)

ax1.set_facecolor(BG)
t = med.index.to_timestamp()
ax1.plot(t, med["revenue"], color=AC, lw=2.2, label="Revenue")
ax1.plot(t, med["inventory"], color="#f59e0b", lw=1.6, label="Inventory")
ax1.plot(t, med["capital_expenditures"], color="#a78bfa", lw=1.6, label="Capital spending")
ax1.axhline(0, color="#555555", lw=0.8)
ax1.set_ylabel("Median growth vs year ago (%)", color=FG)
ax1.set_title(f"Inventory turns after revenue, not before ({len(panel)} S&P 500 companies)",
              color=FG, fontsize=12.5)
ax1.set_ylim(-21, 40)
ax1.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, fontsize=9, ncol=3,
           loc="upper center")
ax1.tick_params(colors=FG)

ax2.set_facecolor(BG)
order = [k for lab in table["Line item"] for k, v in SERIES.items() if v == lab]
mat = full.loc[order].values
im = ax2.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(LAGS)), [str(L) for L in LAGS], color=FG)
ax2.set_yticks(range(len(order)), [SERIES[k] for k in order], color=FG)
ax2.set_xlabel("Quarters shifted (positive = line item moves first)", color=FG)
ax2.set_title("Correlation with revenue growth at each shift", color=FG, fontsize=11)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax2.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                 color=FG if abs(mat[i, j]) > 0.6 else "#0a0a0a")
cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
cb.ax.tick_params(colors=FG)
cb.outline.set_edgecolor("#333333")
ax2.tick_params(colors=FG)

for ax in (ax1, ax2):
    for s in ax.spines.values():
        s.set_color("#333333")

plt.tight_layout()
plt.savefig("corporate-line-item-lead-lag-sp500-python.png", dpi=150, facecolor=BG)
