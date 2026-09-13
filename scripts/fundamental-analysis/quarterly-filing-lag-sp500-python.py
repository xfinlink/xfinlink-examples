# Full write-up: https://xfinlink.com/blog/quarterly-filing-lag-sp500-python
"""How many days pass between a fiscal quarter ending and its Form 10-Q reaching
the public record? Measured across the current S&P 500, fiscal Q1-Q3 period ends
from 2024-07-01 to 2026-06-30."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2024-07-01", "2026-09-13"
PERIOD_MIN, PERIOD_MAX = "2024-07-01", "2026-06-30"
OUT_PNG = "quarterly-filing-lag-sp500-python.png"

# 1. Current S&P 500 roster, carried by entity_id so no ticker change can move a company.
roster = xfl.index("sp500")
entity_ids = sorted({int(e) for e in roster["entity_id"].dropna()})

# 2. Quarterly statements with their period end and filing date.
frames = []
for i in range(0, len(entity_ids), 50):
    frames.append(
        xfl.fundamentals(
            entity_id=entity_ids[i:i + 50],
            period_type="quarterly",
            start=START,
            end=END,
            fields=["revenue"],
            max_rows=50000,
        )
    )
raw = pd.concat(frames, ignore_index=True)

# 3. Screens.
#    - fiscal Q4 has no Form 10-Q: those figures arrive with the annual report
#    - a quarter reappears in later documents as a comparative; a report filed
#      more than 90 days after period end is one of those, not the first filing
#    - a company needs a substantially complete set of the six quarters in the
#      window to describe how it reports, so four is the minimum
df = raw[raw["source"] == "filing"].copy()
df = df[df["fiscal_period"].isin(["Q1", "Q2", "Q3"])]
df = df[(df["period_end"] >= PERIOD_MIN) & (df["period_end"] <= PERIOD_MAX)]
df = df.dropna(subset=["filing_date", "period_end"])
df["lag"] = (df["filing_date"] - df["period_end"]).dt.days
df = df[(df["lag"] > 0) & (df["lag"] <= 90)]
quarters_held = df.groupby("entity_id")["period_end"].transform("size")
df = df[quarters_held >= 4]

lag = df["lag"]
pcts = {
    "minimum": lag.min(), "5th pct": lag.quantile(0.05), "median": lag.median(),
    "75th pct": lag.quantile(0.75), "90th pct": lag.quantile(0.90),
    "95th pct": lag.quantile(0.95), "99th pct": lag.quantile(0.99), "maximum": lag.max(),
}
days = [20, 25, 30, 35, 40, 45, 50]
share = {d: 100 * (lag <= d).mean() for d in days}

per_company = df.groupby("entity_id").agg(
    ticker=("ticker", "last"),
    entity_name=("entity_name", "last"),
    gics_sector=("gics_sector", "last"),
    median=("lag", "median"),
    count=("lag", "size"),
).reset_index()
by_sector = (
    per_company.groupby("gics_sector")["median"]
    .agg(["count", "median"]).sort_values("median")
)

print(f"S&P 500 Form 10-Q filing lag | fiscal Q1-Q3 ending {PERIOD_MIN} to {PERIOD_MAX}")
print(f"Companies: {df['entity_id'].nunique()}   Company-quarters: {len(df):,}")
print()
print("Days from fiscal quarter end to filing")
for name, value in pcts.items():
    print(f"  {name:<10} {value:>5.0f}")
print()
print("Share of quarters on file by day N")
for d in days:
    print(f"  day {d:<3} {share[d]:>6.1f}%")
print()
print(f"Filed in the last five days before the 40-day deadline (36-40): "
      f"{100 * lag.between(36, 40).mean():.1f}%")
print(f"Filed after day 40: {(lag > 40).sum()} quarters "
      f"({100 * (lag > 40).mean():.1f}%) from {df.loc[lag > 40, 'ticker'].nunique()} companies")
tail = lag[lag > 40].value_counts().sort_index()
print("  tail by day: " + "  ".join(f"{d}d x{n}" for d, n in tail.items()))
print()
print("Fastest five companies (median days)")
for _, r in per_company.nsmallest(5, "median").iterrows():
    print(f"  {r['ticker']:<6} {r['entity_name'][:30]:<30} {r['median']:>5.1f}")
print()
print("Slowest five companies")
for _, r in per_company.nlargest(5, "median").iterrows():
    print(f"  {r['ticker']:<6} {r['entity_name'][:30]:<30} {r['median']:>5.1f}")
print()
print("Median company lag by sector")
for sector, r in by_sector.iterrows():
    print(f"  {sector:<24} {r['median']:>5.1f}  ({int(r['count'])} companies)")

# 4. Chart: the distribution, and the same data read as a cumulative curve.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.hist(lag, bins=np.arange(lag.min(), lag.max() + 2) - 0.5, color="#3b82f6")
ax1.axvline(40, color="#e0e0e0", linestyle="--", linewidth=1)
ax1.text(40.7, ax1.get_ylim()[1] * 0.88, "40-day deadline", color="#e0e0e0", fontsize=10)
ax1.set_ylabel("Number of quarters")
ax1.set_title("How long after a quarter ends do S&P 500 financials become public?",
              color="#e0e0e0", fontsize=13, pad=12)

order = np.sort(lag.values)
ax2.plot(order, 100 * np.arange(1, len(order) + 1) / len(order),
         color="#3b82f6", linewidth=2)
ax2.axvline(40, color="#e0e0e0", linestyle="--", linewidth=1)
ax2.set_xlabel("Days from fiscal quarter end to the filing reaching the public record")
ax2.set_ylabel("Share of quarters on file (%)")
ax2.set_ylim(0, 102)

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nChart saved to {OUT_PNG}")
