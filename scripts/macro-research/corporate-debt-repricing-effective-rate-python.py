# Full write-up: https://xfinlink.com/blog/corporate-debt-repricing-effective-rate-python
"""How far has the cost of corporate borrowing actually repriced?

Computes an effective interest rate (interest expense over average debt) for
each S&P 500 non-financial, fiscal 2019 through 2025, and measures how much of
the move in market rates has reached the income statement.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

ids = xfl.index("sp500")["entity_id"].dropna().astype(int).tolist()
f = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                                start="2017-06-01",
                                fields=["revenue", "interest_expense", "total_debt", "ebit"])
               for i in range(0, len(ids), 100)], ignore_index=True)

f = f[~f["gics_sector"].isin(["Financials", "Real Estate"])]
f = f[f["gics_sector"].notna()]

# Fiscal year from period_end, not the fiscal_year label: a January year-end
# belongs to the year that just closed.
pe = pd.to_datetime(f["period_end"])
f["fy"] = pe.dt.year - (pe.dt.month < 6).astype(int)

p = f.pivot_table(index="entity_id", columns="fy",
                  values=["interest_expense", "total_debt", "ebit"], aggfunc="last")
meta = f.sort_values("period_end").groupby("entity_id").last()[["ticker", "entity_name", "gics_sector"]]

rate = {}
for y in YEARS:
    avg_debt = (p["total_debt"][y] + p["total_debt"][y - 1]) / 2
    r = 100 * p["interest_expense"][y] / avg_debt
    # A rate is only meaningful on a real debt balance, and a figure outside
    # this band is not describing a coupon.
    r = r.where((avg_debt > 500) & (r > 0.5) & (r < 15))
    rate[y] = r
rate = pd.DataFrame(rate).join(meta)

print("Effective interest rate on corporate debt, S&P 500 non-financials")
print(f"{'fiscal year':<12}{'n':>5}{'25th':>9}{'median':>9}{'75th':>9}")
for y in YEARS:
    s = rate[y].dropna()
    print(f"{y:<12}{len(s):>5}{s.quantile(.25):>9.2f}{s.median():>9.2f}{s.quantile(.75):>9.2f}")

both = rate[[2021, 2025]].dropna()
both = both.join(meta)
both["change"] = both[2025] - both[2021]
print(f"\nCompanies with a usable rate in both FY2021 and FY2025: {len(both)}")
print(f"Median FY2021 {both[2021].median():.2f}%  ->  median FY2025 {both[2025].median():.2f}%")
print(f"Median change per company: {both['change'].median():+.2f} percentage points")
print(f"Rate rose:  {(both['change'] > 0).sum()} companies "
      f"({100 * (both['change'] > 0).mean():.0f}%)")
print(f"Rose by more than 2 points: {(both['change'] > 2).sum()} companies")
print(f"Fell:       {(both['change'] < 0).sum()} companies")

sector = both.groupby("gics_sector").agg(n=("change", "size"),
                                         fy2021=(2021, "median"),
                                         fy2025=(2025, "median"),
                                         change=("change", "median"))
print("\nBy sector (median):")
print(sector.sort_values("change", ascending=False).round(2).to_string())

print("\nLargest increases:")
print(both.nlargest(8, "change")[["ticker", "gics_sector", 2021, 2025, "change"]]
      .round(2).to_string(index=False))
print("\nLargest decreases:")
print(both.nsmallest(6, "change")[["ticker", "gics_sector", 2021, 2025, "change"]]
      .round(2).to_string(index=False))

# Chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
med = [rate[y].median() for y in YEARS]
q25 = [rate[y].quantile(.25) for y in YEARS]
q75 = [rate[y].quantile(.75) for y in YEARS]
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(YEARS, q25, q75, color="#3b82f6", alpha=0.18,
                label="Middle half of companies")
ax.plot(YEARS, med, color="#3b82f6", linewidth=2.2, marker="o", label="Median company")
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Interest expense as a percent of average debt")
ax.set_title("Corporate borrowing cost repriced slowly and partially")
ax.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", loc="lower right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("corporate-debt-repricing-effective-rate-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart saved")
