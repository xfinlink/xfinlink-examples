# Full write-up: https://xfinlink.com/blog/sp500-addition-institutional-ownership-13f-python
"""Does joining the S&P 500 bring new institutional owners?

Event study on Form 13F holder counts around S&P 500 additions, 2018-2024,
against a control group of continuing index members.
"""

import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from PIL import Image
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.simplefilter("ignore")

BIG3 = {2432, 10139, 486}          # BlackRock, Vanguard, State Street
OFFSETS = [-2, -1, 0, 1, 2]        # quarters relative to the pre-inclusion snapshot
OUT_PNG = "sp500-addition-institutional-ownership-13f-python.png"
qend = lambda p: str(p.end_time.date())

# ---------------------------------------------------------------- events
ev = xfl.index_events("sp500", start="2018-01-01", end="2024-12-31",
                      event_type="added", limit=1000)
ev["effective_date"] = pd.to_datetime(ev["effective_date"])
# anchor = last quarter end at least 10 days before the stock actually joined
ev["q0"] = ev["effective_date"].apply(
    lambda d: pd.Period(d - pd.Timedelta(days=10), freq="Q") - 1)
assert ev["entity_id"].notna().all() and not ev["entity_id"].duplicated().any()
assert ev["effective_date"].is_monotonic_increasing

quarters = sorted({q0 + o for q0 in ev["q0"] for o in OFFSETS})
with ThreadPoolExecutor(8) as pool:
    rosters = dict(pool.map(lambda q: (q, xfl.index("sp500", as_of=qend(q))), quarters))
for q, r in rosters.items():
    ids = r["entity_id"].dropna()
    assert 480 <= len(r) <= 520 and not ids.duplicated().any(), qend(q)

# entities that ever entered or left the index are ineligible as controls
changed = set(xfl.index_events("sp500", start="2016-01-01", end="2026-08-01",
                               limit=1000)["entity_id"].dropna().astype(int))

rng = random.Random(7)
subjects, need = [], set()
for _, r in ev.iterrows():
    subjects.append(dict(group="added", eid=int(r["entity_id"]),
                         ticker=r["ticker"], name=r["entity_name"],
                         eff=r["effective_date"], q0=r["q0"]))
for q0 in sorted(set(ev["q0"])):
    pool_ids = (set(rosters[q0 - 2]["entity_id"].dropna().astype(int))
                & set(rosters[q0 + 2]["entity_id"].dropna().astype(int))) - changed
    for e in rng.sample(sorted(pool_ids), 8):
        subjects.append(dict(group="member", eid=e, ticker=None, name=None,
                             eff=pd.NaT, q0=q0))
for s in subjects:
    need.update((s["eid"], qend(s["q0"] + o)) for o in OFFSETS)

# ------------------------------------------------------------- 13F panel
def snapshot(pair):
    """Holder count and index-manager share for one company-quarter."""
    eid, q = pair
    h = xfl.holdings(entity_id=eid, quarter=q, security_class="COM",
                     fields=["manager_id", "shares"], max_rows=5000)
    if len(h) == 0 or h["manager_id"].duplicated().any():
        return pair, None            # no comparable snapshot for this quarter
    if not 50 <= len(h) < 5000:
        return pair, None
    total = h["shares"].sum()
    big3 = h.loc[h["manager_id"].isin(BIG3), "shares"].sum()
    return pair, (len(h), big3 / total if total > 0 else np.nan)

with ThreadPoolExecutor(8) as pool:
    snaps = dict(pool.map(snapshot, sorted(need)))

rows = []
for s in subjects:
    vals = [snaps[(s["eid"], qend(s["q0"] + o))] for o in OFFSETS]
    if any(v is None for v in vals):
        continue
    rec = dict(s)
    for o, (n, b) in zip(OFFSETS, vals):
        rec[f"h{o}"], rec[f"b{o}"] = n, b
    rows.append(rec)
p = pd.DataFrame(rows)
for o in OFFSETS:
    p[f"lh{o}"] = np.log(p[f"h{o}"])
p["w1"] = p["lh-1"] - p["lh-2"]
p["w2"] = p["lh0"] - p["lh-1"]
p["w3"] = p["lh1"] - p["lh0"]          # the inclusion quarter
p["w4"] = p["lh2"] - p["lh1"]
assert p[[f"h{o}" for o in OFFSETS] + ["w1", "w2", "w3", "w4"]].notna().all().all()

a = p[p["group"] == "added"]
m = p[p["group"] == "member"]
pct = lambda s: 100 * (np.exp(s.median()) - 1)

print(f"S&P 500 additions 2018-2024: {len(ev)} events")
print(f"Sample with a complete five-quarter filing window: "
      f"{len(a)} additions, {len(m)} continuing-member controls\n")

qlab = lambda o: f"{o:+d}" if o else " 0"

print("Median number of institutions reporting a position, by quarter")
print("  quarter   additions   members")
for o in OFFSETS:
    print(f"    {qlab(o)}         {a[f'h{o}'].median():5.0f}     {m[f'h{o}'].median():5.0f}")

print("\nMedian quarter-on-quarter change in the number of reporting institutions")
print("  window        additions   members")
for w, lab in [("w1", " -2 -> -1"), ("w2", " -1 ->  0"),
               ("w3", "  0 -> +1"), ("w4", " +1 -> +2")]:
    tag = "   <- inclusion quarter" if w == "w3" else ""
    print(f" {lab}        {pct(a[w]):+6.2f}%   {pct(m[w]):+6.2f}%{tag}")

print(f"\nAdditions gaining holders in the inclusion quarter: {100 * (a['w3'] > 0).mean():.1f}%"
      f"   (members: {100 * (m['w3'] > 0).mean():.1f}%)")
print(f"Wilcoxon, additions: inclusion quarter vs prior quarter   "
      f"p = {stats.wilcoxon(a['w3'], a['w2']).pvalue:.2e}")
print(f"Mann-Whitney, inclusion quarter: additions vs members     "
      f"p = {stats.mannwhitneyu(a['w3'], m['w3']).pvalue:.2e}")

print("\nBlackRock + Vanguard + State Street share of reported institutional shares (median)")
print("  quarter   additions   members")
for o in OFFSETS:
    print(f"    {qlab(o)}          {100 * a[f'b{o}'].median():5.2f}%    {100 * m[f'b{o}'].median():5.2f}%")
print(f"  median change 0 -> +1:  additions {100 * (a['b1'] - a['b0']).median():+.2f}pp"
      f"   members {100 * (m['b1'] - m['b0']).median():+.2f}pp")

print("\nLargest and smallest holder gains in the inclusion quarter")
show = pd.concat([a.nlargest(4, "w3"), a.nsmallest(4, "w3")])
for _, r in show.iterrows():
    print(f"  {r['ticker']:<5} {r['name'][:28]:<28} {r['eff'].date()}  "
          f"{r['h0']:5.0f} -> {r['h1']:5.0f}  ({100 * (r['h1'] / r['h0'] - 1):+6.1f}%)")

# ------------------------------------------------------------------ chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for grp, lab, col in [(a, "S&P 500 additions", "#3b82f6"), (m, "continuing members", "#9ca3af")]:
    base = grp["h0"].median()
    ax[0].plot(OFFSETS, [100 * grp[f"h{o}"].median() / base for o in OFFSETS],
               marker="o", color=col, label=lab)
    ax[1].plot(OFFSETS, [100 * grp[f"b{o}"].median() for o in OFFSETS],
               marker="o", color=col, label=lab)
ax[0].axvline(0.5, color="#ef4444", linestyle="--", linewidth=1)
ax[1].axvline(0.5, color="#ef4444", linestyle="--", linewidth=1)
ax[0].set_ylabel("Median holder count (pre-inclusion quarter = 100)")
ax[1].set_ylabel("Median share held by the three index managers (%)")
for x in ax:
    x.set_xlabel("Quarters relative to the last filing before inclusion")
    x.set_xticks(OFFSETS)
    x.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")
fig.suptitle("Institutional ownership around S&P 500 additions, 2018-2024", color="#e0e0e0")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
Image.open(OUT_PNG).convert("RGB").quantize(
    colors=128, method=Image.MEDIANCUT).save(OUT_PNG, optimize=True)
