# Full write-up: https://xfinlink.com/blog/sp500-loss-making-companies-by-year-python
#
# How many S&P 500 companies reported a net loss each year, 2008-2025, counted on
# the roster in force on 31 December? A company is a loss-maker when its annual net
# income (attributable to the parent) for the fiscal year is below zero. Fiscal years
# ending June to December belong to that calendar year, January to May to the year
# before. Figures are those on file at the end of the following calendar year (as_of);
# where none was on file by then, the figure on file today is used. Share classes of
# one company count once. The same count on today's roster, traced back through each
# member's own filings, shows what a survivor list would report instead.
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

warnings.simplefilter("ignore")
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = "2026-11-24"  # the day today's roster was read
YEARS = range(2008, 2026)
chunks = lambda xs, n=100: [xs[i:i + n] for i in range(0, len(xs), n)]

# 1. S&P 500 rosters on each 31 December, and today's
rosters = {y: xfl.index("sp500", as_of=f"{y}-12-31") for y in YEARS}
rosters["today"] = xfl.index("sp500")

# 2. Share class -> company: each security's ticker resolved on the first roster date it appears
issuer = {}
for k, r in rosters.items():
    d = RUN_DATE if k == "today" else f"{k}-12-31"
    new = r.dropna(subset=["entity_id", "ticker"])
    new = new[~new["entity_id"].astype(int).isin(issuer)]
    for c in chunks(sorted(set(new["ticker"]))):
        try:
            got = xfl.fundamentals(c, period_type="annual", start=d[:4] + "-01-01", end=d, fields=["net_income"])
        except xfl.XfinlinkError:
            continue
        resolved = got.attrs["meta"].get("tickers_resolved") or {}
        for sec, t in zip(new["entity_id"].astype(int), new["ticker"]):
            if t in resolved and resolved[t] != sec:      # a ticker served as its own security has no company record
                issuer.setdefault(sec, int(resolved[t]))
left = {int(s): t for r in rosters.values() for s, t in zip(r["entity_id"], r["ticker"])
        if pd.notna(s) and pd.notna(t) and int(s) not in issuer}
for c in chunks(sorted(set(left.values())), 10):         # tickers that named several securities that day
    for v in xfl.resolve(c, include=["events"])["data"].values():
        for e in v.get("entities", []):
            if e.get("entity_id") in left and e.get("issuer_entity_id"):
                issuer[e["entity_id"]] = int(e["issuer_entity_id"])

def companies(r):
    """Roster lines grouped by company: issuer id where known, else the member's name."""
    r = r.dropna(subset=["entity_id"])
    out = {}
    for s, n in zip(r["entity_id"].astype(int), r["entity_name"]):
        out.setdefault(issuer.get(s, n), set()).add(n)
    return out

cos = {k: companies(r) for k, r in rosters.items()}
ids_of = lambda k: {c for c in cos[k] if isinstance(c, int)}
today = ids_of("today")

# 3. Annual net income: as on file at the end of the following year, and as on file today
pit = []
for y in YEARS:
    as_of = min(f"{y + 1}-12-31", RUN_DATE)
    for c in chunks(sorted(ids_of(y) | today)):
        f = xfl.fundamentals(entity_id=c, period_type="annual", as_of=as_of, start=f"{y}-06-01",
                             end=f"{y + 1}-05-31", fields=["net_income", "revenue"])
        pit.append(f.assign(src=0))
every = sorted(set().union(*[ids_of(k) for k in cos]))
now = [xfl.fundamentals(entity_id=c, period_type="annual", start="2008-06-01", end="2026-05-31",
                        fields=["net_income", "revenue"], max_rows=100_000).assign(src=1) for c in chunks(every)]
F = pd.concat(pit + now, ignore_index=True)
F["year"] = (F["period_end"] - pd.DateOffset(months=5)).dt.year   # June-December: that year; January-May: the year before

# One row per company-year, at the company's usual fiscal year-end month. A 12-month window
# ending in another month within 300 days of a neighbouring year-end is not a fiscal year.
F["m"] = F["period_end"].dt.month
usual = F.groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax())
usual.update(F[F["revenue"].notna()].groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax()))
F["off"] = ((F["m"] - F["entity_id"].map(usual)) % 12).map(lambda d: min(d, 12 - d))
ends = F.drop_duplicates(["entity_id", "period_end"]).sort_values(["entity_id", "period_end"])
g = ends.groupby("entity_id")["period_end"]
ends["near"] = np.minimum((ends["period_end"] - g.shift()).dt.days.fillna(999),
                          (g.shift(-1) - ends["period_end"]).dt.days.fillna(999))
F = F.merge(ends[["entity_id", "period_end", "near"]], on=["entity_id", "period_end"])
F = F[F["net_income"].notna() & ~((F["off"] >= 2) & (F["near"] < 300))]
NI = (F.sort_values(["src", "off", "period_end"], ascending=[True, True, False])
       .drop_duplicates(["entity_id", "year"]).set_index(["entity_id", "year"]))

# 4. Count loss-makers on each roster, on today's roster traced back, and by later fate
def tally(group, y):
    have = [c for c in group if (c, y) in NI.index]
    return len(have), sum(NI.loc[(c, y), "net_income"] < 0 for c in have)

rows = []
for y in YEARS:
    shared = {c for c, names in cos[y].items() if isinstance(c, int) and len(names) > 1}  # one record, two companies
    members = [c for c in cos[y] if c not in shared]
    stay = [c for c in members if c in today]           # still in the index today
    gone = [c for c in members if c not in today]       # removed since
    later = [c for c in today if c not in cos[y]]       # today's members not yet in the index
    n, k = tally(members, y)
    row = dict(year=y, companies=len(cos[y]), covered=n, losses=k, share=k / n)
    for name, grp in [("today", list(today)), ("stay", stay), ("gone", gone), ("later", later)]:
        gn, gk = tally(grp, y)
        row[name + "_n"], row[name + "_k"], row[name] = gn, gk, gk / gn
    row["fallback"] = sum(NI.loc[(c, y), "src"] == 1 for c in members if (c, y) in NI.index)
    rows.append(row)
t = pd.DataFrame(rows).set_index("year")
t["gap"] = t["today"] - t["share"]

# ---------------- report ----------------
print(f"{len(YEARS)} year-end rosters {YEARS[0]}-{YEARS[-1]} plus today's ({RUN_DATE}); "
      f"{len(every):,} companies; figures on file a year later, else today's ({t['fallback'].sum()} company-years)")
print(f"\n{'year':>4}{'members':>9}{'covered':>9}{'losses':>8}{'share':>8}{'today':>8}{'gap':>7}"
      f"{'still in':>10}{'removed':>9}{'not yet':>9}{'removed, losses':>17}")
for y, r in t.iterrows():
    print(f"{y:>4}{r.companies:9.0f}{r.covered:9.0f}{r.losses:8.0f}{r.share:8.1%}{r.today:8.1%}{r.gap * 100:+7.1f}"
          f"{r.stay:10.1%}{r.gone:9.1%}{r.later:9.1%}{r.gone_k:9.0f} of {r.losses:.0f}")
print(f"\nCoverage: {t['covered'].min():.0f} to {t['covered'].max():.0f} companies a year "
      f"({(t['covered'] / t['companies']).min():.1%} to {(t['covered'] / t['companies']).max():.1%} of members)")
hi = t["losses"].nlargest(3)
print("Most loss-makers: " + ", ".join(f"{y} {v:.0f} ({t.loc[y, 'share']:.1%})" for y, v in hi.items()))
print(f"Fewest: {t['losses'].idxmin()} {t['losses'].min():.0f} ({t.loc[t['losses'].idxmin(), 'share']:.1%}); "
      f"median {t['losses'].median():.0f} a year")
print(f"Today's roster traced back: loss share above the real roster in {(t['gap'] > 0).sum()} of {len(t)} years, "
      f"mean gap {t['gap'].mean() * 100:+.1f} points, widest {t['gap'].max() * 100:+.1f} in {t['gap'].idxmax()}")
print(f"Loss-makers later removed from the index: {t['gone_k'].sum():.0f} of {t['losses'].sum():.0f} company-years "
      f"({t['gone_k'].sum() / t['losses'].sum():.1%}); removed members are {t['gone_n'].sum() / t['covered'].sum():.1%} of member-years")
names = pd.concat([f.drop_duplicates("entity_id") for f in pit + now]).drop_duplicates("entity_id").set_index("entity_id")["entity_name"]
for y in [2008, 2009, 2015, 2020]:
    k = [c for c in cos[y] if (c, y) in NI.index and NI.loc[(c, y), "net_income"] < 0]
    big = sorted(k, key=lambda c: NI.loc[(c, y), "net_income"])[:3]
    print(f"{y} largest losses among covered members ($bn): " + ", ".join(f"{names[c].title()} {NI.loc[(c, y), 'net_income'] / 1e3:.1f}" for c in big))

# Chart: share of members reporting a net loss, real rosters vs today's roster traced back
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(t.index, t["share"] * 100, color="#3b82f6", width=0.7, label="Roster in force each 31 December")
ax.plot(t.index, t["today"] * 100, color="#d95926", lw=2, marker="o", ms=4, label="Today's roster, traced back")
for y in t.index:
    ax.annotate(f"{t.loc[y, 'losses']:.0f}", (y, t.loc[y, "share"] * 100), xytext=(0, -24), textcoords="offset points",
                ha="center", color="#0a0a0a", fontsize=8)
ax.set_xticks(list(t.index))
ax.set_xticklabels([str(y) for y in t.index], rotation=45)
ax.set_xlabel("Fiscal year (bar labels: number of loss-making members)")
ax.set_ylabel("Members reporting a net loss (%)")
ax.set_title("S&P 500 Companies Losing Money by Year, 2008-2025")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 21)
ax.legend(frameon=False, loc="upper center")
plt.tight_layout()
plt.savefig("sp500-loss-making-companies-by-year-python.png", dpi=150, facecolor="#0a0a0a")
