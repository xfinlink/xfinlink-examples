# Full write-up: https://xfinlink.com/blog/earnings-growth-persistence-rank-correlation-python
#
# Does fast earnings growth persist? Growth-cohort and Spearman rank-correlation
# tests on point-in-time S&P 500 membership, fiscal years 2001-2024.
# Built from SEC EDGAR public filings and market data.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM_YEARS = list(range(2004, 2020))   # formation years
W = 3                                  # formation window length, years
REV_FLOOR = 100.0                      # $m, base-year revenue
NI_FLOOR = 10.0                        # $m, net income at both window ends
MARGIN_FLOOR = 0.01                    # net income at least 1% of revenue
RECON_TOL = 0.10                       # annual vs sum-of-quarters tolerance
DROP_SECTORS = {"Financials", "Real Estate"}

# ---------------------------------------------------------------- universe
# Point-in-time membership at each formation year end, so cohorts are formed
# only from names that were actually in the index on that date.
snaps = {y: xfl.index("sp500", as_of=f"{y}-12-31") for y in FORM_YEARS}
uni = pd.concat([d.assign(form_year=y) for y, d in snaps.items()], ignore_index=True)
tickers = sorted(uni["ticker"].unique())


def pull(period_type):
    parts = [xfl.fundamentals(tickers[i:i + 80], start="2000-01-01", end="2025-12-31",
                              period_type=period_type, version="restated",
                              fields=["revenue", "net_income"], max_rows=100000)
             for i in range(0, len(tickers), 80)]
    return pd.concat(parts, ignore_index=True)


ann, qtr = pull("annual"), pull("quarterly")
raw_rows, raw_ents = len(ann), ann["entity_id"].nunique()
dup_pe = int(ann.duplicated(["entity_id", "period_end"]).sum())

# Tickers get re-used across companies over time, so match on entity_id and
# keep only the entities that actually appear in one of the index snapshots.
members = set(uni["entity_id"])
ann = ann[ann["entity_id"].isin(members)].copy()
recycled = raw_ents - ann["entity_id"].nunique()

ann = ann[~ann["gics_sector"].isin(DROP_SECTORS) & (ann["revenue"] > 0)].copy()
qtr = qtr[qtr["entity_id"].isin(members) & (qtr["revenue"] > 0)]
qtr = qtr.drop_duplicates(["entity_id", "period_end"]).sort_values("period_end")

# ------------------------------------------------- internal-consistency screen
# Annual revenue must agree with the four quarters that make up the same year.
# A mismatch marks a change in reporting basis (spin-off, discontinued
# operations, fiscal-year change), across which growth is not comparable.
qmap = {e: (g["period_end"].values, g["revenue"].values) for e, g in qtr.groupby("entity_id")}


def quarter_sum(eid, p_end):
    v = qmap.get(eid)
    if v is None:
        return np.nan
    ends, revs = v
    m = (ends > np.datetime64(p_end - pd.Timedelta(days=320))) & (ends <= np.datetime64(p_end))
    return revs[m].sum() if m.sum() == 4 else np.nan


ann["qsum"] = [quarter_sum(r.entity_id, r.period_end) for r in ann.itertuples()]
recon = ann["qsum"].notna()
dev = (ann["revenue"] / ann["qsum"] - 1).abs()
n_unrecon, n_fail = int((~recon).sum()), int((recon & (dev > RECON_TOL)).sum())
ann = ann[recon & (dev <= RECON_TOL)].copy()

# Fiscal years ending before June belong to the previous calendar year.
ann["year"] = ann["period_end"].dt.year - (ann["period_end"].dt.month < 6).astype(int)
ann = ann.sort_values(["entity_id", "period_end"]).drop_duplicates(["entity_id", "year"], keep="last")

key = list(zip(ann["entity_id"], ann["year"]))
PE = dict(zip(key, ann["period_end"]))
REV = dict(zip(key, ann["revenue"]))
NI = dict(zip(key, ann["net_income"]))
NAME = dict(zip(ann["entity_id"], ann["entity_name"]))
TKR = dict(zip(ann["entity_id"], ann["ticker"]))


def spaced(eid, y0, y1, k):
    """True when two annual period_ends really are k years apart."""
    a, b = PE.get((eid, y0)), PE.get((eid, y1))
    return a is not None and b is not None and abs((b - a).days - k * 365.25) <= 60


def usable_ni(eid, y):
    n, r = NI.get((eid, y)), REV.get((eid, y))
    return (n is not None and r is not None and not pd.isna(n)
            and n >= NI_FLOOR and n >= MARGIN_FLOOR * r)


# ---------------------------------------------------------------- cohort panel
rows = []
for t in FORM_YEARS:
    for eid in set(uni.loc[uni["form_year"] == t, "entity_id"]):
        r0, r1 = REV.get((eid, t - W)), REV.get((eid, t))
        rec = {"form_year": t, "entity_id": eid}
        base = r0 is not None and r1 is not None and r0 >= REV_FLOOR and spaced(eid, t - W, t, W)
        if base:
            rec["rev_form"] = (r1 / r0) ** (1 / W) - 1
        ni_ok = base and usable_ni(eid, t - W) and usable_ni(eid, t)
        if ni_ok:
            rec["ni_form"] = (NI[(eid, t)] / NI[(eid, t - W)]) ** (1 / W) - 1
        for k in range(1, 6):
            if not spaced(eid, t, t + k, k):
                continue
            rk, nk = REV.get((eid, t + k)), NI.get((eid, t + k))
            if base and rk is not None:
                rec[f"rev_r{k}"] = rk / r1
            if ni_ok and nk is not None and not pd.isna(nk):
                rec[f"ni_r{k}"] = nk / NI[(eid, t)]
        # next window, sharing year t as the base
        if base and rec.get(f"rev_r{W}") is not None:
            rec["rev_next"] = rec[f"rev_r{W}"]
        if ni_ok and rec.get(f"ni_r{W}") is not None:
            rec["ni_next"] = rec[f"ni_r{W}"]
        # next window one year later, sharing no year with the formation window
        if base and spaced(eid, t + 1, t + 1 + W, W):
            a, b = REV.get((eid, t + 1)), REV.get((eid, t + 1 + W))
            if a is not None and b is not None:
                rec["rev_skip"] = b / a
                if ni_ok and usable_ni(eid, t + 1) and NI.get((eid, t + 1 + W)) is not None \
                        and not pd.isna(NI[(eid, t + 1 + W)]):
                    rec["ni_skip"] = NI[(eid, t + 1 + W)] / NI[(eid, t + 1)]
        rows.append(rec)

d = pd.DataFrame(rows)

# 1% two-sided trim of the formation variable, inside each formation year
for col in ["rev_form", "ni_form"]:
    keep = pd.Series(False, index=d.index)
    for _, g in d.groupby("form_year"):
        s = g[col].dropna()
        keep.loc[s[(s >= s.quantile(0.01)) & (s <= s.quantile(0.99))].index] = True
    d.loc[~keep, col] = np.nan

d["ni_q"] = d.groupby("form_year")["ni_form"].transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
d["rev_q"] = d.groupby("form_year")["rev_form"].transform(lambda s: pd.qcut(s, 5, labels=False) + 1)

# ---------------------------------------------------------------------- output
bar = "=" * 76
print(bar)
print("SAMPLE  point-in-time S&P 500, formation years %d-%d" % (FORM_YEARS[0], FORM_YEARS[-1]))
print(bar)
print(f"  distinct tickers queried                 {len(tickers)}")
print(f"  annual rows returned / entities          {raw_rows} / {raw_ents}")
print(f"  duplicate (entity, period_end) rows      {dup_pe}")
print(f"  entities excluded by entity matching     {recycled}")
print(f"  rows without four comparable quarters    {n_unrecon}")
print(f"  rows outside the reporting-basis screen  {n_fail}")
print(f"  entities surviving all screens           {ann['entity_id'].nunique()}")
print(f"  firm-formation-years, revenue            {int(d['rev_form'].notna().sum())}")
print(f"  firm-formation-years, net income         {int(d['ni_form'].notna().sum())}")


def cohort(var, qcol):
    out = []
    for q in list(range(1, 6)) + ["all"]:
        g = d if q == "all" else d[d[qcol] == q]
        row = {"q": q, "n": int(g[f"{var}_form"].notna().sum()), "form": g[f"{var}_form"].median()}
        for k in range(1, 6):
            m = g[f"{var}_r{k}"].dropna().median()
            row[f"y{k}"] = m ** (1 / k) - 1
        out.append(row)
    return pd.DataFrame(out)


tables = {}
for var, qcol, lab in [("ni", "ni_q", "NET INCOME"), ("rev", "rev_q", "REVENUE")]:
    c = tables[var] = cohort(var, qcol)
    print()
    print(bar)
    print(f"{lab}  median annualised growth, by formation-window growth quintile")
    print(bar)
    print("  quintile      n   formation     +1yr    +2yr    +3yr    +4yr    +5yr")
    for r in c.itertuples():
        lbl = "all" if r.q == "all" else f"Q{r.q}"
        print(f"  {lbl:8s} {r.n:5d}    {r.form * 100:7.2f}%  "
              + " ".join(f"{getattr(r, f'y{k}') * 100:6.2f}%" for k in range(1, 6)))
    hi, lo = c.iloc[4], c.iloc[0]
    print(f"  Q5 - Q1            {(hi['form'] - lo['form']) * 100:7.2f}pp  "
          + " ".join(f"{(hi[f'y{k}'] - lo[f'y{k}']) * 100:6.2f}pp" for k in range(1, 6)))

print()
print(bar)
print("SPEARMAN RANK CORRELATION between consecutive %d-year growth windows" % W)
print(bar)
print("  form_year     n   revenue  earnings | no shared base:  revenue  earnings")
sp = {c: [] for c in ["rev_next", "ni_next", "rev_skip", "ni_skip"]}
for t in FORM_YEARS:
    g = d[d["form_year"] == t]
    vals = {}
    for var in ["rev", "ni"]:
        for suf in ["next", "skip"]:
            p = g.dropna(subset=[f"{var}_form", f"{var}_{suf}"])
            vals[f"{var}_{suf}"] = stats.spearmanr(p[f"{var}_form"], p[f"{var}_{suf}"]).statistic
            sp[f"{var}_{suf}"].append(vals[f"{var}_{suf}"])
    n = int(g.dropna(subset=["ni_form", "ni_next"]).shape[0])
    print(f"  {t}      {n:4d}    {vals['rev_next']:6.3f}    {vals['ni_next']:6.3f} |"
          f"                  {vals['rev_skip']:6.3f}    {vals['ni_skip']:6.3f}")
print("  mean              " + f"    {np.mean(sp['rev_next']):6.3f}    {np.mean(sp['ni_next']):6.3f} |"
      f"                  {np.mean(sp['rev_skip']):6.3f}    {np.mean(sp['ni_skip']):6.3f}")
print("  years > 0         " + f"    {sum(np.array(sp['rev_next']) > 0):2d}/16     "
      f"{sum(np.array(sp['ni_next']) > 0):2d}/16 |                  "
      f"{sum(np.array(sp['rev_skip']) > 0):2d}/16     {sum(np.array(sp['ni_skip']) > 0):2d}/16")

print()
print("WHERE THE TOP FORMATION QUINTILE LANDS IN THE NEXT %d-YEAR WINDOW" % W)
for var, qcol, lab in [("ni", "ni_q", "earnings"), ("rev", "rev_q", "revenue ")]:
    g = d.dropna(subset=[qcol, f"{var}_next"]).copy()
    g["nq"] = g.groupby("form_year")[f"{var}_next"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1)
    t5 = g.loc[g[qcol] == 5, "nq"].value_counts(normalize=True).sort_index()
    print(f"  {lab}  n={int((g[qcol] == 5).sum()):4d}   "
          + "   ".join(f"Q{int(i)} {v * 100:4.1f}%" for i, v in t5.items())
          + "   (20.0% each if past growth carried no information)")

# ----------------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
colors = {1: "#f59e0b", 2: "#6b7280", 3: "#6b7280", 4: "#6b7280", 5: "#3b82f6"}
x = range(6)
labels = ["formation\n(3 yr)", "+1 yr", "+2 yr", "+3 yr", "+4 yr", "+5 yr"]
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, (var, title) in zip(axes, [("ni", "Net income"), ("rev", "Revenue")]):
    c = tables[var]
    for r in c.itertuples():
        y = [r.form * 100] + [getattr(r, f"y{k}") * 100 for k in range(1, 6)]
        if r.q == "all":
            ax.plot(x, y, color="#e0e0e0", lw=1.2, ls="--", label="All firms", zorder=2)
        else:
            ax.plot(x, y, color=colors[r.q], lw=2.4 if r.q in (1, 5) else 1.2,
                    marker="o", ms=4, alpha=1.0 if r.q in (1, 5) else 0.55,
                    label=f"Q{r.q}" + (" (fastest)" if r.q == 5 else " (slowest)" if r.q == 1 else ""),
                    zorder=3 if r.q in (1, 5) else 1)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.margins(y=0.16)
    ax.set_ylabel(f"{title} growth, % a year\n(cohort median)")
    ax.set_title(f"{title}: growth after ranking on the previous three years",
                 color="#e0e0e0", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, ncol=6, fontsize=8.5, loc="upper right")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(labels)
fig.suptitle("Does fast earnings growth persist? S&P 500 members, formation years 2004-2019",
             color="#e0e0e0", fontsize=12.5)
plt.tight_layout()
plt.savefig("earnings-growth-persistence-rank-correlation-python.png", dpi=150,
            facecolor="#0a0a0a")
