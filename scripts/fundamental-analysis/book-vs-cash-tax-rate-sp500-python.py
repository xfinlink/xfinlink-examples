# Full write-up: https://xfinlink.com/blog/book-vs-cash-tax-rate-sp500-python

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy.stats import spearmanr

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# -- Universe: current S&P 500 constituents ---------------------------------
tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique().tolist())

FIELDS = [
    "period_end", "fiscal_year", "revenue", "pretax_income", "income_tax_expense",
    "cash_taxes_paid", "research_and_development", "total_assets", "gics_sector",
]

# -- Fetch annual filings in chunks, then verify the row count --------------
frames = []
for i in range(0, len(tickers), 50):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames.append(
            xfl.fundamentals(
                tickers[i:i + 50], period_type="annual",
                start="2021-06-01", end="2025-12-31", fields=FIELDS,
            )
        )
df = pd.concat(frames, ignore_index=True)
print(f"Fetched {len(df):,} annual rows for {df['ticker'].nunique()} companies\n")

# -- Sample: fiscal 2022-2025, positive pretax income, both tax lines present
d = df[df["fiscal_year"].between(2022, 2025)].copy()
d = d.dropna(subset=["pretax_income", "income_tax_expense", "cash_taxes_paid"])
d = d[d["pretax_income"] > 0]

d["book_rate"] = d["income_tax_expense"] / d["pretax_income"]
d["cash_rate"] = d["cash_taxes_paid"] / d["pretax_income"]
d["gap"] = d["book_rate"] - d["cash_rate"]

# -- Leg 1: single-year cross-section (fiscal 2025) -------------------------
f25 = d[d["fiscal_year"] == 2025]

print("=" * 66)
print("FISCAL 2025 CROSS-SECTION")
print("=" * 66)
print(f"Companies                          {len(f25):>10}")
print(f"Median book effective tax rate     {f25['book_rate'].median() * 100:>9.1f}%")
print(f"Median cash tax rate               {f25['cash_rate'].median() * 100:>9.1f}%")
print(f"Median company-level gap           {f25['gap'].median() * 100:>+9.1f}pp")
print(f"25th percentile of gap             {f25['gap'].quantile(.25) * 100:>+9.1f}pp")
print(f"75th percentile of gap             {f25['gap'].quantile(.75) * 100:>+9.1f}pp")
print(f"Share with gap wider than  5pp     {(f25['gap'].abs() > .05).mean() * 100:>9.1f}%")
print(f"Share with gap wider than 10pp     {(f25['gap'].abs() > .10).mean() * 100:>9.1f}%")

# -- Leg 2: four-year cumulative, aggregated in dollars ---------------------
years = d.groupby("ticker")["fiscal_year"].nunique()
full = d[d["ticker"].isin(years[years == 4].index)].copy()
full["rd"] = full["research_and_development"].fillna(0).abs()

c = full.groupby(["ticker", "gics_sector"]).agg(
    pretax=("pretax_income", "sum"),
    tax_expense=("income_tax_expense", "sum"),
    cash_tax=("cash_taxes_paid", "sum"),
    rev=("revenue", "sum"),
    rd=("rd", "sum"),
).reset_index()
c["cum_book"] = c["tax_expense"] / c["pretax"]
c["cum_cash"] = c["cash_tax"] / c["pretax"]
c["cum_gap"] = c["cum_book"] - c["cum_cash"]
c["rd_intensity"] = c["rd"] / c["rev"]

signs = full.assign(s=np.sign(full["gap"])).groupby("ticker")["s"].sum()

print()
print("=" * 66)
print("FISCAL 2022-2025 CUMULATIVE (dollars summed, then divided)")
print("=" * 66)
print(f"Companies with all four years      {len(c):>10}")
print(f"Aggregate book rate                {c['tax_expense'].sum() / c['pretax'].sum() * 100:>9.1f}%")
print(f"Aggregate cash rate                {c['cash_tax'].sum() / c['pretax'].sum() * 100:>9.1f}%")
print(f"Median cumulative gap              {c['cum_gap'].median() * 100:>+9.1f}pp")
print(f"Share with cumulative gap > 5pp    {(c['cum_gap'].abs() > .05).mean() * 100:>9.1f}%")
print(f"Same-signed gap in all four years  {(signs.abs() == 4).sum():>10} of {len(signs)}")
print(f"  book above cash all four years   {(signs == 4).sum():>10}")
print(f"  cash above book all four years   {(signs == -4).sum():>10}")

print()
print("Median cumulative gap by sector (percentage points)")
print(f"{'Sector':<24}{'n':>5}{'Book':>9}{'Cash':>9}{'Gap':>10}")
sec = c.groupby("gics_sector").agg(
    n=("cum_gap", "size"), book=("cum_book", "median"),
    cash=("cum_cash", "median"), gap=("cum_gap", "median"),
).sort_values("gap")
for name, r in sec.iterrows():
    print(f"{name:<24}{int(r['n']):>5}{r['book'] * 100:>8.1f}%{r['cash'] * 100:>8.1f}%{r['gap'] * 100:>+9.1f}pp")

# -- Leg 3: does research spending explain the gap? -------------------------
rho_all, p_all = spearmanr(c["rd_intensity"], c["cum_gap"])
spenders = c[c["rd_intensity"] > 0].copy()
rho_rd, p_rd = spearmanr(spenders["rd_intensity"], spenders["cum_gap"])

print()
print("=" * 66)
print("RESEARCH INTENSITY VS THE CUMULATIVE GAP")
print("=" * 66)
print(f"{'Spearman rho, full sample':<32}{'n=' + str(len(c)):>7}{rho_all:>+9.3f}   p={p_all:.2e}")
print(f"{'Spearman rho, R&D reporters only':<32}{'n=' + str(len(spenders)):>7}{rho_rd:>+9.3f}   p={p_rd:.2e}")
print()
none_rd = c[c["rd_intensity"] == 0]
print(f"{'Group':<36}{'n':>5}{'Median gap':>13}")
print(f"{'No reported research spend':<36}{len(none_rd):>5}{none_rd['cum_gap'].median() * 100:>+11.1f}pp")
spenders["quintile"] = pd.qcut(spenders["rd_intensity"], 5, labels=[1, 2, 3, 4, 5])
for q, g in spenders.groupby("quintile", observed=True):
    lo, hi = g["rd_intensity"].min() * 100, g["rd_intensity"].max() * 100
    label = f"R&D/revenue quintile {q} ({lo:.1f}-{hi:.1f}%)"
    print(f"{label:<36}{len(g):>5}{g['cum_gap'].median() * 100:>+11.1f}pp")

# -- Chart ------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.hist(f25["gap"].clip(-0.4, 0.4) * 100, bins=40, color="#3b82f6", edgecolor="#0a0a0a")
ax1.axvline(0, color="#e0e0e0", lw=1)
ax1.set_xticks([-40, -20, 0, 20, 40])
ax1.set_xticklabels(["≤-40", "-20", "0", "20", "≥40"])
ax1.set_xlabel("Reported tax rate minus cash tax rate (percentage points)")
ax1.set_ylabel("Number of companies")
ax1.set_title("Fiscal 2025: one year", fontsize=10)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

order = sec.sort_values("gap")
colors = ["#ef4444" if v < 0 else "#3b82f6" for v in order["gap"]]
ax2.barh(range(len(order)), order["gap"] * 100, color=colors)
ax2.set_yticks(range(len(order)))
ax2.set_yticklabels(order.index, fontsize=8)
ax2.tick_params(left=False)
ax2.axvline(0, color="#e0e0e0", lw=1)
ax2.set_xlabel("Median cumulative gap (percentage points)")
ax2.set_title("Fiscal 2022-2025: four years combined", fontsize=10)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)

fig.suptitle("Reported tax rates versus cash taxes actually paid, S&P 500", fontsize=12)
plt.tight_layout()
plt.savefig("book-vs-cash-tax-rate-sp500-python.png", dpi=130, facecolor="#0a0a0a")
