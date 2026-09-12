# Full write-up: https://xfinlink.com/blog/pe-versus-ev-ebit-sp500-python
#
# Does the balance sheet change which S&P 500 companies look cheap?
# Ranks the 2025 fiscal-year cross-section on the equity multiple (price to
# earnings) and on the enterprise multiple (enterprise value to operating
# profit), then measures how far the two rankings disagree and what drives it.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF = "2025-12-31"
FY_LO, FY_HI = "2025-12-01", "2026-01-31"
EXCLUDE = {"Financials", "Real Estate"}
FIELDS = ["net_income", "ebit", "pretax_income", "income_tax_expense", "total_debt",
          "cash_and_short_term_investments", "weighted_avg_shares_diluted"]

# 1. Point-in-time index membership, carried by entity id.
ids = sorted(xfl.index("sp500", as_of=AS_OF)["entity_id"].dropna().astype(int))

# 2. Most recent annual statements for each member.
a = pd.concat(
    [xfl.fundamentals(entity_id=ids[i:i + 60], period_type="annual",
                      start="2024-06-01", fields=FIELDS, max_rows=200000)
     for i in range(0, len(ids), 60)], ignore_index=True)
a = a.dropna(subset=["period_end"]).drop_duplicates(["entity_id", "period_end"])
f = a.sort_values("period_end").groupby("entity_id").tail(1).set_index("entity_id")
f = f[~f["gics_sector"].isin(EXCLUDE)]
f = f[(f["period_end"] >= FY_LO) & (f["period_end"] <= FY_HI)]
f = f.dropna(subset=FIELDS)
f = f[(f["net_income"] > 0) & (f["ebit"] > 0) & (f["weighted_avg_shares_diluted"] > 0)]

# 3. Keep income statements that run straight from pre-tax income to net income.
gap = (f["net_income"] - (f["pretax_income"] - f["income_tax_expense"])).abs()
f = f[gap <= 0.10 * f["net_income"]]

# 4. Closing price on the fiscal year end, so price and balance sheet share a date.
px = pd.concat(
    [xfl.prices(entity_id=list(f.index)[i:i + 60], start="2025-12-15", end="2026-02-02",
                fields=["close"], max_rows=200000)
     for i in range(0, len(f), 60)], ignore_index=True)
px = px.dropna(subset=["close"]).sort_values("date")
d = pd.merge_asof(f.reset_index().sort_values("period_end"), px[["entity_id", "date", "close"]],
                  left_on="period_end", right_on="date", by="entity_id",
                  tolerance=pd.Timedelta("7D")).set_index("entity_id")
d = d.dropna(subset=["close"])

# 5. The two multiples.
d["market_cap"] = d["close"] * d["weighted_avg_shares_diluted"]
d["net_debt"] = d["total_debt"] - d["cash_and_short_term_investments"]
d["ev"] = d["market_cap"] + d["net_debt"]
d = d[d["ev"] > 0]
d["pe"] = d["market_cap"] / d["net_income"]
d["ev_ebit"] = d["ev"] / d["ebit"]
d["nd"] = d["net_debt"] / d["market_cap"]
d["r_pe"] = d["pe"].rank()
d["r_ev"] = d["ev_ebit"].rank()
d["move"] = d["r_pe"] - d["r_ev"]          # positive: cheaper on the enterprise multiple

# 6. How far apart are the two rankings?
n = len(d)
k = n // 5
rho = stats.spearmanr(d["pe"], d["ev_ebit"]).statistic
cheap_pe = set(d.nsmallest(k, "pe").index)
cheap_ev = set(d.nsmallest(k, "ev_ebit").index)
dear_pe = set(d.nlargest(k, "pe").index)
dear_ev = set(d.nlargest(k, "ev_ebit").index)
q = pd.qcut(d["nd"], 5, labels=False)
byq = d.groupby(q).agg(n=("pe", "size"), nd=("nd", "median"), pe=("pe", "median"),
                       ev=("ev_ebit", "median"), move=("move", "median"))
bysec = d.groupby("gics_sector").agg(n=("pe", "size"), nd=("nd", "median"),
                                     move=("move", "median")).sort_values("move")

print("Equity multiple against enterprise multiple, S&P 500 at 31 December 2025")
print(f"Universe: {len(ids)} index members on {AS_OF}, carried by entity id;")
print(f"          {n} non-financial companies with a fiscal year ending")
print("          1 December 2025 to 31 January 2026 and a reconciling profit line")
print("P/E     = diluted market capitalisation / net income")
print("EV/EBIT = (diluted market capitalisation + total debt - cash) / operating profit")
print()
print(f"Median P/E {d['pe'].median():.2f}      median EV/EBIT {d['ev_ebit'].median():.2f}"
      f"      median net debt / market cap {d['nd'].median():+.2f}")
print(f"Spearman rank correlation between the two multiples   {rho:+.4f}")
print()
print(f"Cheapest quintile ({k} names)   {len(cheap_pe & cheap_ev)} shared, "
      f"{k - len(cheap_pe & cheap_ev)} replaced")
print(f"Dearest quintile  ({k} names)   {len(dear_pe & dear_ev)} shared, "
      f"{k - len(dear_pe & dear_ev)} replaced")
print(f"Companies moving more than 50 rank places: {(d['move'].abs() > 50).sum()} of {n}")
print()
print("By net debt quintile (net debt / market capitalisation)")
print("  quintile        net debt   median P/E   median EV/EBIT   median rank move")
for i, r in byq.iterrows():
    print(f"  {int(i) + 1} ({int(r['n']):3d} names)  {r['nd']:+8.2f}   {r['pe']:10.2f}"
          f"   {r['ev']:14.2f}   {r['move']:+16.1f}")
print()
hdr = "  ticker  company                          P/E   EV/EBIT   net debt   P/E rank   EV rank"
print("Largest moves toward cheap on the enterprise multiple")
print(hdr)
for _, r in d.nlargest(8, "move").iterrows():
    print(f"  {r['ticker']:<6s}  {r['entity_name'][:28]:<28s}  {r['pe']:7.2f}  {r['ev_ebit']:7.2f}"
          f"  {r['nd']:+9.2f}   {r['r_pe']:8.0f}  {r['r_ev']:8.0f}")
print()
print("Largest moves toward expensive on the enterprise multiple")
print(hdr)
for _, r in d.nsmallest(8, "move").iterrows():
    print(f"  {r['ticker']:<6s}  {r['entity_name'][:28]:<28s}  {r['pe']:7.2f}  {r['ev_ebit']:7.2f}"
          f"  {r['nd']:+9.2f}   {r['r_pe']:8.0f}  {r['r_ev']:8.0f}")
print()
print("By sector")
print("  sector                     names   median net debt   median rank move")
for s, r in bysec.iterrows():
    print(f"  {s:<24s}  {int(r['n']):5d}   {r['nd']:+15.2f}   {r['move']:+16.1f}")

# ── Chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3f3f3f", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

rest, heavy = d[q < 4], d[q == 4]
ax1.plot([0, n], [0, n], color="#525252", linewidth=0.9, linestyle="--", zorder=1)
ax1.scatter(rest["r_pe"], rest["r_ev"], s=20, alpha=0.75, color="#9ca3af",
            label="net debt quintiles 1 to 4", linewidths=0, zorder=2)
ax1.scatter(heavy["r_pe"], heavy["r_ev"], s=24, alpha=0.95, color="#3b82f6",
            label="most indebted quintile", linewidths=0, zorder=3)
ax1.axvline(k, color="#525252", linewidth=0.7, alpha=0.6)
ax1.axhline(k, color="#525252", linewidth=0.7, alpha=0.6)
ax1.set_xlim(0, n + 2)
ax1.set_ylim(0, n + 2)
ax1.set_xlabel("Rank on price to earnings (1 = cheapest)")
ax1.set_ylabel("Rank on enterprise value to operating profit")
ax1.set_title("Where the two rankings disagree", color="#e0e0e0", fontsize=10)
ax1.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor="#e0e0e0")

meds = byq["move"].values
ax2.bar(range(1, 6), meds, color=["#9ca3af"] * 4 + ["#3b82f6"], width=0.62)
ax2.axhline(0, color="#525252", linewidth=0.9)
ax2.set_xticks(range(1, 6))
ax2.set_xticklabels([f"Q{i}\n{byq['nd'].iloc[i - 1]:+.2f}" for i in range(1, 6)])
ax2.set_ylim(meds.min() * 1.35, meds.max() * 1.9)
ax2.set_xlabel("Net debt to market capitalisation, quintile")
ax2.set_ylabel("Median change in rank places")
ax2.set_title("Only the most indebted quintile moves", color="#e0e0e0", fontsize=10)
span = meds.max() - meds.min()
for i, v in enumerate(meds):
    ax2.text(i + 1, v + (span * 0.05 if v >= 0 else -span * 0.11), f"{v:+.0f}",
             ha="center", color="#e0e0e0", fontsize=8.5)

plt.tight_layout()
plt.savefig("pe-versus-ev-ebit-sp500-python.png", dpi=150, facecolor="#0a0a0a")
