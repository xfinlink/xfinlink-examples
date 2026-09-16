# Full write-up: https://xfinlink.com/blog/sp500-membership-kaplan-meier-hazard-python
#
# How long does an S&P 500 membership last, and does exit risk depend on how
# long a company has already been in the index? Kaplan-Meier survival estimate
# and annual hazard rate, built from the index membership event log.

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

warnings.simplefilter("ignore")

SLUG = "sp500-membership-kaplan-meier-hazard-python"
MIN_SPELL_DAYS = 30  # spin-off price-discovery placements are not memberships

# ---------------------------------------------------------------- event log
frames, offset = [], 0
while True:
    page = xfl.index_events("SP500", start="1957-01-01", limit=1000, offset=offset)
    if page.empty:
        break
    frames.append(page)
    offset += len(page)
    if len(page) < 1000:
        break

events = pd.concat(frames, ignore_index=True)
events["effective_date"] = pd.to_datetime(events["effective_date"])
n_events = len(events)
log_start, log_end = events["effective_date"].min(), events["effective_date"].max()

events = events.dropna(subset=["entity_id"]).copy()
events["entity_id"] = events["entity_id"].astype(int)
events = events.sort_values(
    ["entity_id", "effective_date", "event_type"], kind="mergesort"
)

# ------------------------------------------------- event log -> spells
# A company with two listed share classes generates two additions, so track how
# many listings are open. A spell runs from the addition that opens membership
# to the removal that closes the last one.
spells, unpaired = [], 0
for entity_id, g in events.groupby("entity_id"):
    depth, start = 0, None
    for _, row in g.iterrows():
        if row["event_type"] == "added":
            if depth == 0:
                start = row["effective_date"]
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                spells.append((entity_id, start, row["effective_date"]))
        else:
            unpaired += 1  # removal with no observed opening addition
    if depth > 0:
        spells.append((entity_id, start, pd.NaT))

sp = pd.DataFrame(spells, columns=["entity_id", "added", "removed"])

# ----------------------------------------------------- censoring set
# A spell is right-censored only if the company is in the roster as it stands.
roster = xfl.index("sp500")
open_now = sp["removed"].isna()
sp = sp[~open_now | sp["entity_id"].isin(set(roster["entity_id"]))].copy()

sp["days"] = (sp["removed"].fillna(log_end) - sp["added"]).dt.days
sp["exited"] = sp["removed"].notna().astype(int)
sp["years"] = sp["days"] / 365.25
dropped = sp[sp["days"] < MIN_SPELL_DAYS].copy()
sp = sp[sp["days"] >= MIN_SPELL_DAYS].copy()
n_short = len(dropped)
n_exit, n_cens = int(sp["exited"].sum()), int((1 - sp["exited"]).sum())

# ------------------------------------------------------- Kaplan-Meier
km = SurvfuncRight(sp["years"], sp["exited"])


def surv(fn, t, max_follow=np.inf):
    """Survival probability at tenure t years, NaN if follow-up is too short."""
    if t > max_follow:
        return np.nan
    s = fn.surv_prob[fn.surv_times <= t]
    return float(s[-1]) if len(s) else 1.0


def median(fn):
    below = fn.surv_times[fn.surv_prob <= 0.5]
    return float(below[0]) if len(below) else np.nan


km_median = median(km)

# the naive estimator: completed spells only, censored memberships thrown away
done = sp.loc[sp["exited"] == 1, "years"]
naive_median, naive_mean = done.median(), done.mean()

# ------------------------------------------- annual hazard by tenure year
hazard = []
for y in range(0, 26):
    at_risk = int((sp["years"] >= y).sum())
    exits = int(((sp["exited"] == 1) & (sp["years"] >= y) & (sp["years"] < y + 1)).sum())
    hazard.append((y, at_risk, exits, exits / at_risk))
haz = pd.DataFrame(hazard, columns=["year", "at_risk", "exits", "rate"])

# ------------------------------------------------------ decade cohorts
sp["decade"] = (sp["added"].dt.year // 10) * 10
rows = []
for d in sorted(sp["decade"].unique()):
    g = sp[sp["decade"] == d]
    if len(g) < 30:
        continue
    f = SurvfuncRight(g["years"], g["exited"])
    mf = g["years"].max()  # a cohort cannot report survival past its follow-up
    rows.append((d, len(g), int(g["exited"].sum()),
                 surv(f, 5, mf), surv(f, 10, mf), median(f)))
coh = pd.DataFrame(
    rows, columns=["decade", "n", "exits", "s5", "s10", "median"]
).set_index("decade")

grp = np.where(sp["decade"] < 1990, 0, np.where(sp["decade"] >= 2000, 1, 2))
keep = grp < 2
chi2, pval = survdiff(sp["years"][keep], sp["exited"][keep], grp[keep])

# ------------------------------------------------------------- output
print("=== S&P 500 membership tenure: Kaplan-Meier survival ===")
print(
    f"Event log: {n_events} events, {log_start:%Y-%m-%d} to {log_end:%Y-%m-%d}. "
    f"Spells: {len(sp)} ({n_exit} ended, {n_cens} still running)"
)
print(f"Removals not matched to a complete spell: {unpaired}. "
      f"Spells under {MIN_SPELL_DAYS} days excluded: {n_short}")
print()
print(f"Kaplan-Meier median tenure : {km_median:.1f} years")
print(f"Naive median, ended spells : {naive_median:.1f} years  "
      f"(understates by {km_median - naive_median:.1f}y, "
      f"{(1 - naive_median / km_median) * 100:.0f}%)")
print(f"Naive mean,   ended spells : {naive_mean:.1f} years")
print()
print("Tenure    Survived   Left by then")
for t in (1, 3, 5, 10, 15, 20, 25):
    s = surv(km, t)
    print(f"{t:>3} years    {s * 100:>6.1f}%       {(1 - s) * 100:>5.1f}%")
print()
def band_rate(lo, hi):
    b = haz[(haz["year"] >= lo) & (haz["year"] <= hi)]
    return b["exits"].sum() / b["at_risk"].sum()


print("Annual chance of leaving, by how long the company has already been in")
print("Tenure band    At risk   Exits   Hazard per year")
for lo, hi in [(0, 1), (2, 4), (5, 9), (10, 14), (15, 19), (20, 25)]:
    b = haz[(haz["year"] >= lo) & (haz["year"] <= hi)]
    print(f"{lo:>2}-{hi:<2} years   {int(b['at_risk'].sum()):>8}   "
          f"{int(b['exits'].sum()):>5}   {band_rate(lo, hi) * 100:>10.2f}%")
plateau_rate = band_rate(5, 25)
print()
print(f"Flat-hazard benchmark: at {plateau_rate * 100:.2f}% a year, half a cohort "
      f"is gone after ln(2)/h = {np.log(2) / plateau_rate:.1f} years")
print(f"Kaplan-Meier median actually observed: {km_median:.1f} years")
print()
print("Joining cohort by decade")
print("Decade    Spells   Ended   5y surv   10y surv   Median")
def pct(v):
    return f"{v * 100:>5.1f}%" if np.isfinite(v) else "   n/e"


for d, r in coh.iterrows():
    med = f"{r['median']:.1f}y" if np.isfinite(r["median"]) else "  n/r"
    print(f"{int(d)}s     {int(r['n']):>6}   {int(r['exits']):>5}    "
          f"{pct(r['s5'])}     {pct(r['s10'])}    {med:>6}")
print("n/e: follow-up does not yet reach that horizon. n/r: median not yet reached.")
print(f"\nLog-rank, pre-1990 joiners vs 2000+ joiners: chi2={chi2:.2f}, p={pval:.4f}")

# --------------------------------------------------------- validation
print("\n--- validation ---")
print(f"Negative or zero durations: {int((sp['days'] <= 0).sum())}")
print(f"Open spells not in the roster: "
      f"{int((sp['removed'].isna() & ~sp['entity_id'].isin(set(roster['entity_id']))).sum())}")
print(f"Additions {int((events['event_type'] == 'added').sum())}, "
      f"removals {int((events['event_type'] == 'removed').sum())}")
kept = sp[["years", "exited"]]
allsp = pd.concat([kept, dropped[["years", "exited"]]], ignore_index=True)
e01 = ((allsp["exited"] == 1) & (allsp["years"] < 2)).sum()
r01 = (allsp["years"] >= 0).sum() + (allsp["years"] >= 1).sum()
print(f"Years 0-1 hazard with short spells put back in: {e01 / r01 * 100:.2f}% per year")
oldest = sp[sp["exited"] == 0].nlargest(5, "years").merge(
    roster[["entity_id", "ticker"]], on="entity_id", how="left"
)
print("Longest running memberships: "
      + ", ".join(f"{r.ticker} {r.years:.0f}y" for r in oldest.itertuples()))

# ------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.5, 1]})

ax1.step(km.surv_times, km.surv_prob * 100, where="post",
         color="#3b82f6", lw=2, label="Kaplan-Meier (censoring handled)")
nd = np.sort(done.values)
ax1.step(nd, (1 - np.arange(1, len(nd) + 1) / len(nd)) * 100, where="post",
         color="#f59e0b", lw=1.6, ls="--",
         label="Ended memberships only (naive)")
ax1.axhline(50, color="#555555", lw=0.8, ls=":")
ax1.plot([km_median], [50], "o", color="#3b82f6", ms=7)
ax1.plot([naive_median], [50], "o", color="#f59e0b", ms=7)
ax1.annotate(f"{km_median:.1f}y", (km_median, 50), textcoords="offset points",
             xytext=(8, 8), color="#3b82f6", fontsize=10)
ax1.annotate(f"{naive_median:.1f}y", (naive_median, 50), textcoords="offset points",
             xytext=(-34, 8), color="#f59e0b", fontsize=10)
ax1.set_xlim(0, 45)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Still in the index (%)")
ax1.set_title("How long an S&P 500 membership lasts", color="#e0e0e0",
              fontsize=13, pad=10)
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           frameon=True, loc="upper right")
for s in ax1.spines.values():
    s.set_color("#333333")

ax2.bar(haz["year"], haz["rate"] * 100, color="#3b82f6", width=0.75)
plateau = (haz.loc[haz["year"] >= 5, "exits"].sum()
           / haz.loc[haz["year"] >= 5, "at_risk"].sum()) * 100
ax2.axhline(plateau, color="#f59e0b", lw=1.4, ls="--",
            label=f"Average after year 5: {plateau:.1f}% a year")
ax2.set_xlabel("Years already spent in the index")
ax2.set_ylabel("Chance of leaving\nin the next year (%)")
ax2.set_xlim(-0.8, 25.8)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           frameon=True, loc="lower right")
for s in ax2.spines.values():
    s.set_color("#333333")

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nChart saved: {SLUG}.png")
