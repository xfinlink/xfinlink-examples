# Full write-up: https://xfinlink.com/blog/other-comprehensive-income-sp500-python
"""How large is other comprehensive income relative to reported net income?
Measured across the S&P 500 roster of 31 December 2024, annual reports with a
period end in each of the six calendar years 2019 to 2024."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF = "2024-12-31"
WIN_START, WIN_END = "2019-01-01", "2024-12-31"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
OUT_PNG = "other-comprehensive-income-sp500-python.png"

# 1. Point-in-time roster, carried by entity_id so no symbol change moves a company.
roster = xfl.index("sp500", as_of=AS_OF)
entity_ids = sorted({int(e) for e in roster["entity_id"].dropna()})

frames = []
for i in range(0, len(entity_ids), 50):
    frames.append(xfl.fundamentals(
        entity_id=entity_ids[i:i + 50], period_type="annual",
        start="2017-06-01", end="2025-06-30",
        fields=["comprehensive_income", "net_income", "total_equity",
                "accumulated_other_comprehensive_income"]))
raw = pd.concat(frames, ignore_index=True)

# 2. Other comprehensive income is the wedge between the two profit measures, and
#    the balance sheet carries the same amount as the movement in accumulated OCI.
df = raw.sort_values(["entity_id", "period_end"]).copy()
df["oci"] = df["comprehensive_income"] - df["net_income"]
df["d_aoci"] = df.groupby("entity_id")["accumulated_other_comprehensive_income"].diff()
df["year"] = df["period_end"].dt.year

# 3. Keep a year only where the two routes to OCI agree to within 1% of book equity.
w = df[(df["period_end"] >= WIN_START) & (df["period_end"] <= WIN_END)].copy()
w = w[(w["oci"] - w["d_aoci"]).abs() <= 0.01 * w["total_equity"].abs()]
w = w.sort_values("period_end").drop_duplicates(["entity_id", "year"], keep="last")

# 4. Balanced panel: one annual report ending in every one of the six years.
full = w.groupby("entity_id")["year"].transform("nunique") == len(YEARS)
p = w[full].copy()

print(f"S&P 500 members at {AS_OF}: {len(entity_ids)}   "
      f"annual rows retrieved: {len(raw):,}")
print(f"Company-years passing the reconciliation: {len(w):,}")
print(f"Balanced panel: {p['entity_id'].nunique()} companies x {len(YEARS)} years "
      f"= {len(p):,} company-years\n")

yr = p.groupby("year").agg(ni=("net_income", "sum"), oci=("oci", "sum"))
yr["pct"] = 100 * yr["oci"] / yr["ni"]
yr["neg"] = 100 * p.groupby("year")["oci"].apply(lambda s: (s < 0).mean())
print("Year   Net income $bn   OCI $bn   OCI as % of net income   Companies with OCI < 0")
for y, r in yr.iterrows():
    print(f"{y}        {r.ni / 1000:9,.1f}   {r.oci / 1000:7,.1f}   "
          f"{r.pct:20.1f}   {r.neg:19.0f}%")

co = p.groupby(["entity_id", "entity_name", "gics_sector"]).agg(
    ni=("net_income", "sum"), oci=("oci", "sum")).reset_index()
print(f"\nSix-year totals: net income ${co.ni.sum() / 1000:,.0f}bn, "
      f"OCI ${co.oci.sum() / 1000:,.0f}bn "
      f"({100 * co.oci.sum() / co.ni.sum():.2f}% of net income)")
print(f"Companies with negative six-year OCI: "
      f"{(co.oci < 0).sum()} of {len(co)} ({100 * (co.oci < 0).mean():.0f}%)")
pos = co[co.ni > 0].copy()
pos["wedge"] = 100 * pos["oci"] / pos["ni"]
print(f"Of the {len(pos)} with positive six-year net income, median OCI is "
      f"{pos.wedge.median():.2f}% of it; {100 * (pos.wedge.abs() > 10).mean():.1f}% "
      f"sit beyond plus or minus 10%")

sec = co.groupby("gics_sector").agg(n=("entity_id", "size"), ni=("ni", "sum"),
                                    oci=("oci", "sum"))
sec["pct"] = 100 * sec["oci"] / sec["ni"]
sec = sec.sort_values("pct")
print("\nSector                    N   Net income $bn   OCI $bn   OCI as % of net income")
for s, r in sec.iterrows():
    print(f"{s:<24}{r.n:3.0f}   {r.ni / 1000:12,.0f}   {r.oci / 1000:7,.1f}   {r.pct:20.2f}")

y22 = p[p.year == 2022]
print("\n2022, the six largest OCI losses and the six largest OCI gains ($m)")
for label, rows in (("loss", y22.nsmallest(6, "oci")), ("gain", y22.nlargest(6, "oci"))):
    for _, r in rows.iterrows():
        print(f"  {label:<5}{r.entity_name[:32]:<34}{r.gics_sector[:22]:<24}"
              f"net income {r.net_income:>9,.0f}   OCI {r.oci:>9,.0f}")

# 5. Chart: the year-by-year total, and where the six-year total lands by sector.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.bar([str(y) for y in yr.index], yr["oci"] / 1000, color="#3b82f6", width=0.6)
ax1.axhline(0, color="#e0e0e0", linewidth=1)
for x, v in enumerate(yr["oci"] / 1000):
    ax1.text(x, v + (8 if v >= 0 else -18), f"{v:,.0f}", ha="center",
             color="#e0e0e0", fontsize=10)
ax1.set_ylabel("Total OCI ($bn)")
ax1.set_ylim(-260, 90)
ax1.set_title("Profit that skips the income statement: S&P 500 other comprehensive income",
              color="#e0e0e0", fontsize=13, pad=12)

ax2.barh(range(len(sec)), sec["pct"], color="#3b82f6", height=0.62)
ax2.axvline(0, color="#e0e0e0", linewidth=1)
ax2.set_yticks(range(len(sec)))
ax2.set_yticklabels(sec.index)
ax2.set_xlabel("Six-year OCI as a percentage of six-year net income")

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nChart saved to {OUT_PNG}")
