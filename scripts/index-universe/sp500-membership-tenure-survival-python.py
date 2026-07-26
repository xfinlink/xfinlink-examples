# Full write-up: https://xfinlink.com/blog/sp500-membership-tenure-survival-python
"""Is S&P 500 membership getting less durable?

Two independent estimators of index durability, 1980-2014 addition cohorts:
  Signal 1  cohort survival  - reconstruct membership spells from the event log
  Signal 2  roster retention - compare point-in-time rosters 10 years apart

Contrasted against the naive statistic that circulates in the press: the
average tenure of membership spells that have already ended.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from statsmodels.stats.proportion import proportions_ztest

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

HORIZON = pd.Timedelta(days=3652)  # 10 years
FIRST_COHORT, LAST_COHORT = 1980, 2014

# ---------------------------------------------------------------- event log
frames, offset = [], 0
while True:
    page = xfl.index_events("sp500", start="1957-01-01", limit=1000, offset=offset)
    if page.empty:
        break
    frames.append(page)
    offset += len(page)
    if len(page) < 1000:
        break

events = pd.concat(frames, ignore_index=True)
events["effective_date"] = pd.to_datetime(events["effective_date"])
events = events.sort_values(["entity_id", "effective_date"])
data_end = events["effective_date"].max()

# --------------------------------------------------------- membership spells
# The log is per security, so a company with two listed share classes produces
# two additions. Count open listings per company: a spell runs from the first
# addition that takes the count above zero to the removal that returns it to
# zero. Removals with no open listing are founding-roster members whose entry
# predates the log; they are skipped and fall outside the cohort window anyway.
spells = []
for entity_id, g in events.groupby("entity_id"):
    open_listings, spell_start = 0, None
    for _, row in g.iterrows():
        if row["event_type"] == "added":
            if open_listings == 0:
                spell_start = row["effective_date"]
            open_listings += 1
        elif open_listings > 0:
            open_listings -= 1
            if open_listings == 0:
                spells.append((entity_id, row["ticker"], spell_start,
                               row["effective_date"]))
    if open_listings > 0:  # still a member: right-censored
        spells.append((entity_id, g.iloc[-1]["ticker"], spell_start, pd.NaT))

sp = pd.DataFrame(spells, columns=["entity_id", "ticker", "added", "removed"])
sp["cohort"] = sp["added"].dt.year
sp["bucket"] = (sp["cohort"] // 5) * 5
sp["tenure"] = (sp["removed"] - sp["added"]).dt.days / 365.25

# every cohort below is fully observed at the 10-year horizon
w = sp[sp["cohort"].between(FIRST_COHORT, LAST_COHORT)].copy()
w["survived_10y"] = np.where(
    w["removed"].isna(), 1, (w["removed"] > w["added"] + HORIZON).astype(int)
)

cohorts = w.groupby("bucket").agg(
    additions=("added", "size"),
    ended=("tenure", "count"),
    naive_tenure=("tenure", "mean"),
    survival_10y=("survived_10y", "mean"),
)
cohorts["still_member"] = w.groupby("bucket")["removed"].apply(lambda s: s.isna().mean())

# ------------------------------------- signal 2: point-in-time roster retention
# Keyed on the membership spell (entity_id + added_date), not entity_id alone.
# One entity can hold several spells, so an entity-only key would score a match
# between two different memberships and overstate retention by roughly 2 points.
def spell_keys(as_of):
    snap = xfl.index("sp500", as_of=as_of)
    return set(zip(snap["entity_id"], snap["added_date"])), len(snap)

retention = []
for year in range(FIRST_COHORT, LAST_COHORT + 1):
    now, size = spell_keys(f"{year}-01-01")
    later, _ = spell_keys(f"{year + 10}-01-01")
    retention.append((year, size, len(now & later) / len(now)))

ret = pd.DataFrame(retention, columns=["year", "roster", "retained_10y"])
ret["bucket"] = (ret["year"] // 5) * 5
ret_b = ret.groupby("bucket")[["roster", "retained_10y"]].mean()
cohorts = cohorts.join(ret_b["retained_10y"])

# ------------------ signal 3: exit hazard from event counts, no entity linkage
# Removal events divided by roster size. No addition-to-removal pairing, so this
# estimator is unaffected by how membership rows attach to company records.
events["year"] = events["effective_date"].dt.year
removals = events[events["event_type"] == "removed"].groupby("year").size()
roster_size = ret.set_index("year")["roster"]
hazard = []
for b in cohorts.index:
    yrs = range(b, b + 5)
    exits = sum(int(removals.get(y, 0)) for y in yrs)
    seats = sum(int(roster_size.get(y, np.nan)) for y in yrs)
    rate = exits / seats
    hazard.append((b, exits, seats / 5, rate, (1 - rate) ** 10))
hz = pd.DataFrame(hazard, columns=["bucket", "exits", "roster", "exit_rate",
                                   "implied_10y"]).set_index("bucket")

# ------------------------------------------------- is the change significant?
early = w[w["cohort"] < 1990]
late = w[w["cohort"] >= 2005]
zstat, pval = proportions_ztest(
    [early["survived_10y"].sum(), late["survived_10y"].sum()],
    [len(early), len(late)],
)
slope = np.polyfit(cohorts.index.astype(float), cohorts["survival_10y"], 1)[0]

# ------------------------------------------------------------------- output
print("=== S&P 500 membership durability by addition cohort ===")
print(f"Event log: {len(events)} events, {events['effective_date'].min():%Y-%m-%d}"
      f" to {data_end:%Y-%m-%d}. Spells reconstructed: {len(sp)}\n")
print(f"{'Cohort':<11}{'Added':>7}{'Ended':>7}{'NaiveTenure':>13}"
      f"{'StillIn':>9}{'Surv10y':>9}{'Retain10y':>11}")
for b, r in cohorts.iterrows():
    print(f"{b}-{b % 100 + 4:02d}{'':<4}{int(r.additions):>7}{int(r.ended):>7}"
          f"{r.naive_tenure:>12.1f}y{r.still_member:>8.0%}"
          f"{r.survival_10y:>9.1%}{r.retained_10y:>11.1%}")

print(f"\nNaive mean tenure of ended spells: {cohorts['naive_tenure'].iloc[0]:.1f}y "
      f"(1980-84) -> {cohorts['naive_tenure'].iloc[-1]:.1f}y (2010-14), "
      f"{cohorts['naive_tenure'].iloc[-1] / cohorts['naive_tenure'].iloc[0] - 1:+.0%}")
print(f"10-year survival:                  {cohorts['survival_10y'].iloc[0]:.1%} "
      f"(1980-84) -> {cohorts['survival_10y'].iloc[-1]:.1%} (2010-14)")
print(f"Roster retention (independent):    {cohorts['retained_10y'].iloc[0]:.1%} "
      f"(1980-84) -> {cohorts['retained_10y'].iloc[-1]:.1%} (2010-14)")
print(f"Exit-rate implied survival:        {hz['implied_10y'].iloc[0]:.1%} "
      f"(1980-84) -> {hz['implied_10y'].iloc[-1]:.1%} (2010-14)")
print(f"Trend in 10-year survival: {slope * 1000:+.1f} pp per decade")
print(f"1980s cohorts {early['survived_10y'].mean():.1%} (n={len(early)}) vs "
      f"2005-14 cohorts {late['survived_10y'].mean():.1%} (n={len(late)}): "
      f"z={zstat:.2f}, p={pval:.3f}")
print(f"Censoring: {cohorts['still_member'].iloc[0]:.0%} of the 1980-84 cohort is still "
      f"a member and excluded from its tenure average; "
      f"{cohorts['still_member'].iloc[-1]:.0%} of the 2010-14 cohort.")

print(f"\n{'Period':<11}{'Exits':>7}{'Roster':>9}{'AnnualExit':>12}{'Implied10y':>12}")
for b, r in hz.iterrows():
    print(f"{b}-{b % 100 + 4:02d}{'':<4}{int(r.exits):>7}{r.roster:>9.0f}"
          f"{r.exit_rate:>11.2%}{r.implied_10y:>12.1%}")

print("\n--- validation: spells vs point-in-time rosters ---")
for d in ["1990-01-01", "2005-01-01", "2026-01-01"]:
    D = pd.Timestamp(d)
    recon = set(sp[(sp["added"] <= D)
                   & (sp["removed"].isna() | (sp["removed"] > D))]["entity_id"])
    roster = set(xfl.index("sp500", as_of=d)["entity_id"])
    print(f"{d}: roster {len(roster)}, spells open {len(recon)}, "
          f"open but not in roster {len(recon - roster)}, "
          f"in roster via founding entry {len(roster - recon)}")

# --------------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
labels = [f"{b}-{b % 100 + 4:02d}" for b in cohorts.index]
x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.bar(x, cohorts["naive_tenure"], color="#3b82f6", width=0.6)
ax1.set_ylabel("Average years in the index")
ax1.set_title("Average tenure of memberships that have already ended (biased)",
              color="#e0e0e0", fontsize=11)
for xi, v in zip(x, cohorts["naive_tenure"]):
    ax1.text(xi, v + 0.3, f"{v:.1f}", ha="center", color="#e0e0e0", fontsize=9)

ax2.plot(x, cohorts["survival_10y"] * 100, "o-", color="#3b82f6", lw=2,
         label="Cohort survival: added members still there 10 years later")
ax2.plot(x, cohorts["retained_10y"] * 100, "s--", color="#94a3b8", lw=2,
         label="Roster retention: full index membership 10 years later")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Percent still in the index")
ax2.set_xlabel("Year the company joined the index")
ax2.set_title("Share still in the index 10 years later (censoring-free)",
              color="#e0e0e0", fontsize=11)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

fig.suptitle("Is S&P 500 membership getting less durable?", color="#e0e0e0", fontsize=13)
plt.tight_layout()
plt.savefig("sp500-membership-tenure-survival-python.png", dpi=150,
            facecolor="#0a0a0a")
