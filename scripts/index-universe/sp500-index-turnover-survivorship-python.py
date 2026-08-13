# Full write-up: https://xfinlink.com/blog/sp500-index-turnover-survivorship-python
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# --- 1. Every S&P 500 membership change, 1990-2025 -------------------------
parts, offset = [], 0
while True:
    page = xfl.index_events("sp500", start="1990-01-01", end="2025-12-31",
                            limit=1000, offset=offset)
    if page.empty:
        break
    parts.append(page)
    offset += len(page)
    if len(page) < 1000:
        break

events = pd.concat(parts, ignore_index=True)
events["year"] = pd.to_datetime(events["effective_date"]).dt.year
by_year = (events.groupby(["year", "event_type"]).size()
           .unstack(fill_value=0).reindex(range(1990, 2026), fill_value=0))
by_year["turnover_pct"] = by_year["added"] / 500 * 100

print(f"Membership events 1990-2025: {len(events):,}  "
      f"(added {by_year['added'].sum():,}, removed {by_year['removed'].sum():,})")

print("\nAdditions per year, five-year blocks")
blocks = by_year.loc[1991:2025].copy()
blocks["block"] = ((blocks.index - 1991) // 5).map(
    lambda b: f"{1991 + b * 5}-{1995 + b * 5}")
blk = blocks.groupby("block").agg(added=("added", "mean"),
                                  turnover=("turnover_pct", "mean"))
blk["implied_tenure"] = 100 / blk["turnover"]
for name, row in blk.iterrows():
    print(f"  {name}   {row['added']:5.1f} adds/yr   {row['turnover']:5.2f}% "
          f"turnover   {row['implied_tenure']:5.1f} yr implied tenure")

# --- 2. How much of the 2005 roster was still there later on ---------------
# Track companies by persistent entity identifier, so a symbol change or a
# renaming does not read as an exit.
def roster_ids(as_of):
    df = xfl.index("sp500", as_of=as_of, limit=1000)
    return set(df["entity_id"].dropna())


base = xfl.index("sp500", as_of="2005-12-31", limit=1000)
cohort = base.dropna(subset=["entity_id"])
cohort_ids = set(cohort["entity_id"])
n = len(cohort_ids)

print(f"\nSurvival of the {n} companies tracked from the 2005-12-31 roster")
mean_turnover = by_year.loc[2006:2025, "turnover_pct"].mean() / 100
print(f"  average turnover 2006-2025: {mean_turnover * 100:.2f}% a year")
print(f"  {'date':<12}{'years':>6}{'still in':>10}{'share':>9}{'flat-rate':>11}")

curve = []
for as_of, years in [("2010-12-31", 5), ("2015-12-31", 10),
                     ("2020-12-31", 15), ("2025-12-31", 20)]:
    left = len(cohort_ids & roster_ids(as_of))
    actual = left / n * 100
    flat = (1 - mean_turnover) ** years * 100
    curve.append((years, actual, flat))
    print(f"  {as_of:<12}{years:>6}{left:>10}{actual:>8.1f}%{flat:>10.1f}%")

# --- Chart -----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.bar(by_year.index, by_year["added"], color="#3b82f6")
ax1.plot(by_year.index, by_year["added"].rolling(5, center=True).mean(),
         color="#e0e0e0", linewidth=1.8, label="5-year average")
ax1.set_xlabel("Year")
ax1.set_ylabel("Companies added to the index")
ax1.set_title("Additions per year, 1990-2025")
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")

yrs = [0] + [c[0] for c in curve]
ax2.plot(yrs, [100] + [c[1] for c in curve], color="#3b82f6", marker="o",
         linewidth=2, label="Observed")
ax2.plot(yrs, [100] + [c[2] for c in curve], color="#e0e0e0", linestyle="--",
         linewidth=1.5, label="Flat 4.76% exit rate")
ax2.set_xlabel("Years after 31 December 2005")
ax2.set_ylabel("Share of the 2005 cohort still in the index (%)")
ax2.set_title("Survival of the 2005 cohort")
ax2.set_ylim(0, 100)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("sp500-index-turnover-survivorship-python.png", dpi=150)
