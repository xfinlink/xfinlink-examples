# Full write-up: https://xfinlink.com/blog/cash-vs-book-tax-rate-python
"""Reported tax expense versus cash tax actually paid, over a full decade.

Universe: the union of the point-in-time S&P 500 rosters at each year end
from 2013 to 2023, carried by entity_id. Financials and real estate are
excluded. A company enters the sample only if it reports ten consecutive
profitable fiscal years, 2014 through 2023.
"""
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "cash-vs-book-tax-rate-python"
YEARS = range(2014, 2024)

# ── 1. Point-in-time universe ────────────────────────────────────────
ids = set()
for year in range(2013, 2024):
    ids |= set(xfl.index("sp500", as_of=f"{year}-12-31")["entity_id"])
ids = sorted(ids)

# ── 2. Ten years of annual filings, carried by entity_id ─────────────
fields = ["revenue", "pretax_income", "income_tax_expense",
          "cash_taxes_paid", "net_income", "gics_sector"]
frames = [
    xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                     start="2014-01-01", end="2024-06-30", fields=fields)
    for i in range(0, len(ids), 100)
]
df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

# Fiscal year from the period end date: a year closing in January to May
# belongs to the previous calendar year.
df["fy"] = df["period_end"].dt.year - (df["period_end"].dt.month <= 5).astype(int)

# ── 3. Screens ───────────────────────────────────────────────────────
d = df[~df["gics_sector"].isin(["Financials", "Real Estate"])].copy()
d = d.dropna(subset=["revenue", "pretax_income", "income_tax_expense",
                     "cash_taxes_paid"])
d = d[(d["revenue"] > 0) & (d["pretax_income"] > 0) & d["fy"].isin(YEARS)]
d = d.drop_duplicates(["entity_id", "fy"], keep=False)

counts = d.groupby("entity_id")["fy"].nunique()
d = d[d["entity_id"].isin(counts[counts == len(YEARS)].index)]

# ── 4. Decade tax rates ──────────────────────────────────────────────
# Sum by entity, not by ticker: a company that changed symbol during the
# decade files under two tickers and one entity_id.
d = d.sort_values(["entity_id", "fy"])
labels = d.groupby("entity_id")[["ticker", "entity_name", "gics_sector"]].last()
firm = (d.groupby("entity_id")
          [["pretax_income", "income_tax_expense", "cash_taxes_paid"]]
          .sum().join(labels).reset_index())
firm["book_etr"] = firm["income_tax_expense"] / firm["pretax_income"]
firm["cash_etr"] = firm["cash_taxes_paid"] / firm["pretax_income"]
firm["gap_pp"] = (firm["book_etr"] - firm["cash_etr"]) * 100

sector = (firm.groupby("gics_sector")
               .agg(n=("ticker", "size"),
                    book=("book_etr", "median"),
                    cash=("cash_etr", "median"))
               .assign(gap_pp=lambda t: (t["book"] - t["cash"]) * 100)
               .sort_values("gap_pp", ascending=False))

widest = firm.nlargest(10, "gap_pp")

# ── 5. Output ────────────────────────────────────────────────────────
agg_book = firm["income_tax_expense"].sum() / firm["pretax_income"].sum()
agg_cash = firm["cash_taxes_paid"].sum() / firm["pretax_income"].sum()

print(f"Companies with 10 straight profitable years (FY2014-FY2023): {len(firm)}")
print(f"Pretax income summed over the decade: ${firm['pretax_income'].sum()/1e6:,.2f}tn")
print(f"Aggregate book tax rate: {agg_book:6.2%}")
print(f"Aggregate cash tax rate: {agg_cash:6.2%}")
print()

print("Firm-level gap, book tax rate minus cash tax rate (percentage points)")
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct:>2}th percentile: {firm['gap_pp'].quantile(pct/100):+6.1f}")
below = (firm["cash_etr"] < 0.5 * firm["book_etr"]).sum()
print(f"  Companies paying less than half their reported rate in cash: "
      f"{below} of {len(firm)}")
print()

print("Sector medians over the decade")
print(f"{'Sector':<24}{'n':>4}{'Book':>8}{'Cash':>8}{'Gap':>8}")
for name, row in sector.iterrows():
    print(f"{name:<24}{int(row['n']):>4}{row['book']:>8.1%}"
          f"{row['cash']:>8.1%}{row['gap_pp']:>+8.1f}")
print()

print("Ten widest decade gaps")
print(f"{'Ticker':<8}{'Sector':<24}{'Book':>8}{'Cash':>8}{'Gap':>8}")
for _, r in widest.iterrows():
    print(f"{r['ticker']:<8}{r['gics_sector']:<24}{r['book_etr']:>8.1%}"
          f"{r['cash_etr']:>8.1%}{r['gap_pp']:>+8.1f}")

# ── 6. Chart ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1, 1]})

for ax, frame, labels, title in [
    (ax1, sector, sector.index, "Sector medians"),
    (ax2, widest.iloc[::-1], widest.iloc[::-1]["ticker"], "Ten widest company gaps"),
]:
    y = range(len(frame))
    book = frame["book"] if "book" in frame else frame["book_etr"]
    cash = frame["cash"] if "cash" in frame else frame["cash_etr"]
    ax.hlines(y, cash * 100, book * 100, color="#3a3a3a", linewidth=2, zorder=1)
    ax.scatter(book * 100, y, color="#3b82f6", s=48, zorder=2, label="Reported tax rate")
    ax.scatter(cash * 100, y, color="#e0e0e0", s=48, zorder=2, label="Cash tax rate")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ten-year tax rate on pretax income (%)")
    ax.set_title(title, loc="left", fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

ax1.invert_yaxis()
ax1.legend(frameon=False, loc="lower right")
fig.suptitle("Reported tax rate versus cash tax paid, 2014-2023",
             fontsize=13, x=0.02, ha="left")
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
