# Full write-up: https://xfinlink.com/blog/buybacks-funded-by-cash-flow-or-debt-python
"""Do S&P 500 companies buy back stock with cash they generated, or with borrowing?

Compares each company-year's shareholder payout (buybacks plus dividends) against the
free cash flow of the same year, then checks what happened to total debt when the payout
ran ahead of the cash.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = range(2014, 2025)
# Free cash flow is defined from operating cash flow and capital expenditure, neither of
# which carries its usual meaning for a bank or an insurer, so financials sit outside the
# sample along with real estate.
EXCLUDED_SECTORS = {"Financials", "Real Estate"}

# Point-in-time membership: each year is judged on the companies that were in the index
# that year, not on the survivors of 2024.
roster = {}
for y in YEARS:
    members = xfl.index("sp500", as_of=f"{y}-12-31")
    roster[y] = set(members["entity_id"])
all_ids = sorted(set().union(*roster.values()))

fund = xfl.fundamentals(
    entity_id=all_ids, start="2013-06-01", end="2025-06-30", period_type="annual",
    fields=["share_repurchases", "dividends_paid", "free_cash_flow", "total_debt"],
    max_rows=200000,
)
fund = fund[~fund["gics_sector"].isin(EXCLUDED_SECTORS)]
fund = fund.dropna(subset=["share_repurchases", "dividends_paid", "free_cash_flow", "total_debt"])

# Keep only company-years where the company was actually a member that year.
fund = fund[[eid in roster.get(y, ()) for eid, y in zip(fund["entity_id"], fund["fiscal_year"])]]
fund = fund[fund["fiscal_year"].isin(YEARS)]
fund = fund.sort_values(["entity_id", "fiscal_year"])

fund["payout"] = fund["share_repurchases"] + fund["dividends_paid"]
fund["gap"] = fund["payout"] - fund["free_cash_flow"]
fund["debt_change"] = fund.groupby("entity_id")["total_debt"].diff()
# A company's first observed year has no prior balance sheet to difference against.
panel = fund.dropna(subset=["debt_change"])
panel = panel[panel["fiscal_year"] > min(YEARS)]

print(f"{len(panel)} company-years, {panel['entity_id'].nunique()} companies, "
      f"{panel['fiscal_year'].min()}-{panel['fiscal_year'].max()}")

by_year = panel.groupby("fiscal_year").apply(lambda d: pd.Series({
    "buybacks_bn": d["share_repurchases"].sum() / 1000,
    "dividends_bn": d["dividends_paid"].sum() / 1000,
    "fcf_bn": d["free_cash_flow"].sum() / 1000,
    "payout_over_fcf": d["payout"].sum() / d["free_cash_flow"].sum(),
    "pct_cos_over": (d["gap"] > 0).mean() * 100,
    "n": len(d),
}), include_groups=False)

print("\nAggregate payout against free cash flow, by year ($bn)")
print(by_year.round(2).to_string())

# Did the companies that outspent their cash flow borrow?
over = panel[panel["gap"] > 0]
under = panel[panel["gap"] <= 0]
print("\nMedian change in total debt ($m), company-years split by payout vs free cash flow")
print(f"payout above cash flow  n={len(over):5d}  median {over['debt_change'].median():8.1f}"
      f"  share raising debt {100 * (over['debt_change'] > 0).mean():.1f}%")
print(f"payout within cash flow n={len(under):5d}  median {under['debt_change'].median():8.1f}"
      f"  share raising debt {100 * (under['debt_change'] > 0).mean():.1f}%")

print("\nBy sector: share of company-years with payout above free cash flow")
sec = panel.groupby("gics_sector").agg(
    pct_over=("gap", lambda s: (s > 0).mean() * 100),
    median_debt_change=("debt_change", "median"),
    n=("gap", "size")).round(1).sort_values("pct_over", ascending=False)
print(sec.to_string())

print("\nConcentration: share of all buyback dollars from the largest 20 spenders")
tot = panel.groupby("entity_id")["share_repurchases"].sum().sort_values(ascending=False)
print(f"top 20 of {len(tot)} companies = {100 * tot.head(20).sum() / tot.sum():.1f}% of buybacks")

# ── chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
yrs = by_year.index.astype(int)
ax1.bar(yrs - 0.2, by_year["buybacks_bn"] + by_year["dividends_bn"], 0.4,
        label="Buybacks + dividends", color="#3b82f6")
ax1.bar(yrs + 0.2, by_year["fcf_bn"], 0.4, label="Free cash flow", color="#6b7280")
ax1.set_xlabel("Fiscal year")
ax1.set_ylabel("Total, $ billion")
ax1.set_title("Payout against cash generated")
ax1.legend(frameon=False)

ax2.plot(yrs, by_year["pct_cos_over"], marker="o", color="#f59e0b")
ax2.set_xlabel("Fiscal year")
ax2.set_ylabel("Share of companies (%)")
ax2.set_title("Paying out more than free cash flow")
ax2.set_ylim(0, 60)

plt.tight_layout()
plt.savefig("buybacks-funded-by-cash-flow-or-debt-python.png", dpi=150, facecolor="#0a0a0a")
