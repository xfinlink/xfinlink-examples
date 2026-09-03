# Full write-up: https://xfinlink.com/blog/does-high-payout-slow-earnings-growth-python
"""Do companies that pay out more of their earnings grow earnings more slowly?

Sorts point-in-time S&P 500 members into dividend payout buckets at two formation
dates (fiscal 2014 and fiscal 2019) and measures the following five fiscal years
of total net income growth. Arnott and Asness (2003) found that higher aggregate
payout preceded faster market-wide earnings growth; this is the cross-sectional
version of the same test on individual companies.

Universe and fundamentals are addressed by entity id, so a symbol later reassigned
to another company cannot enter the sample.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

CHART = "does-high-payout-slow-earnings-growth-python.png"
COHORTS = [(2014, "2015-06-30"), (2019, "2020-06-30")]  # base fiscal year, roster date
HORIZON = 5                       # fiscal years measured after formation
EXCLUDE = {"Financials", "Real Estate"}
TOL = 0.10                        # per-share reconciliation tolerance
MIN_BASE = 50.0                   # $50m floor on base-year net income
FIELDS = ["revenue", "net_income", "eps_diluted", "dividends_per_share",
          "dividends_paid", "weighted_avg_shares_diluted", "total_assets",
          "operating_cash_flow"]
EDGES = [-1e-9, 0.0, 0.20, 0.40, 0.60, 1.00, 1e9]
LABELS = ["No dividend", "0-20%", "20-40%", "40-60%", "60-100%", "Above 100%"]


def chunked(seq, n=60):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))


# ---------------------------------------------------- 1. point-in-time rosters
rosters = {fy: retry(xfl.index, "sp500", as_of=d) for fy, d in COHORTS}
universe = sorted({int(e) for r in rosters.values() for e in r["entity_id"]})

# ---------------------------------------------------- 2. annual fundamentals
fun = pd.concat([retry(xfl.fundamentals, entity_id=c, period_type="annual",
                       start="2013-06-01", end="2026-06-30",
                       fields=FIELDS, max_rows=200000)
                 for c in chunked(universe)], ignore_index=True)
fun["period_end"] = pd.to_datetime(fun["period_end"])

# Fiscal year comes from the period end date, not from the reported label: a year
# ending in January through May belongs to the calendar year before it.
fun["fy"] = np.where(fun["period_end"].dt.month <= 5,
                     fun["period_end"].dt.year - 1, fun["period_end"].dt.year)

# One row per company-year: keep the filing that carries the most primary fields,
# breaking ties on the later filing date.
fun["filled"] = fun[FIELDS].notna().sum(axis=1)
fun = fun.sort_values(["entity_id", "fy", "filled", "filing_date"])
fun = fun.groupby(["entity_id", "fy"], as_index=False).tail(1)

# Reconcile the per-share figures against the dollar figures. Diluted EPS times the
# diluted share count must reproduce net income; rows that miss by more than TOL are
# not internally consistent and take no part in the analysis.
ann = fun.dropna(subset=["revenue", "net_income", "eps_diluted",
                         "weighted_avg_shares_diluted"]).copy()
ann = ann[ann["weighted_avg_shares_diluted"] > 0]
implied = ann["eps_diluted"] * ann["weighted_avg_shares_diluted"] / ann["net_income"]
checked = ann[(implied - 1.0).abs() <= TOL].copy()

print("Payout ratio and subsequent earnings growth, S&P 500")
print(f"  company-years pulled                       {len(fun):>6,}")
print(f"  with revenue, earnings and a share count   {len(ann):>6,}")
print(f"  per-share figures reconcile within {TOL:.0%}      {len(checked):>6,}")

# ---------------------------------------------------- 3. formation and outcome
rows = []
for base_fy, _ in COHORTS:
    members = {int(e) for e in rosters[base_fy]["entity_id"]}
    base = checked[checked["entity_id"].isin(members) & (checked["fy"] == base_fy)]
    base = base[~base["gics_sector"].isin(EXCLUDE)].copy()
    end = checked.loc[checked["fy"] == base_fy + HORIZON,
                      ["entity_id", "net_income", "revenue"]]
    end = end.rename(columns={"net_income": "ni_end", "revenue": "rev_end"})
    merged = base.merge(end, on="entity_id", how="left")
    merged["cohort"] = f"FY{base_fy}-FY{base_fy + HORIZON}"
    rows.append(merged)
panel = pd.concat(rows, ignore_index=True)

# A company that reports cash dividends but no per-share figure cannot be placed in a
# payout bucket, so it leaves the sample; everything else with no per-share dividend
# is a non-payer.
mixed = panel["dividends_per_share"].isna() & (panel["dividends_paid"].fillna(0) > 0)
panel = panel[~mixed].copy()
panel["dividends_per_share"] = panel["dividends_per_share"].fillna(0.0)

loss = panel[(panel["eps_diluted"] <= 0) | (panel["net_income"] <= 0)]
small = panel[(panel["net_income"] > 0) & (panel["net_income"] < MIN_BASE)]
form = panel[panel["net_income"] >= MIN_BASE].copy()
form = form[form["eps_diluted"] > 0]
form["payout"] = form["dividends_per_share"] / form["eps_diluted"]
form["bucket"] = pd.cut(form["payout"], bins=EDGES, labels=LABELS)

print(f"  index members outside {' and '.join(sorted(EXCLUDE))}         "
      f"{len(panel) + int(mixed.sum()):>6,}")
print(f"  no per-share dividend figure to bucket     {int(mixed.sum()):>6,}")
print(f"  loss-making or zero earnings at formation  {len(loss):>6,}")
print(f"  base-year net income under ${MIN_BASE:.0f}m          {len(small):>6,}")
print(f"  companies entering the payout buckets      {len(form):>6,}")

# ---------------------------------------------------- 4. five-year outcomes
grown = form.dropna(subset=["ni_end"]).copy()
grown["growth"] = grown["ni_end"] / grown["net_income"] - 1.0
grown["rev_growth"] = grown["rev_end"] / grown["revenue"] - 1.0

table = grown.groupby("bucket", observed=True).agg(
    companies=("growth", "size"),
    med_payout=("payout", "median"),
    med_growth=("growth", "median"),
    mean_growth=("growth", "mean"),
    med_rev=("rev_growth", "median"),
)
table["annualised"] = (1.0 + table["med_growth"]) ** (1.0 / HORIZON) - 1.0
table["no_fy5"] = form["bucket"].value_counts() - table["companies"]

print(f"\nFive fiscal years of net income growth after formation "
      f"({len(grown)} companies)\n")
print(f"{'Payout at formation':<20}{'n':>5}{'Median':>9}{'Median':>10}"
      f"{'Annual':>9}{'Mean':>10}{'Median':>10}{'No FY+5':>9}")
print(f"{'':<20}{'':>5}{'payout':>9}{'earnings':>10}{'ised':>9}"
      f"{'earnings':>10}{'revenue':>10}{'filing':>9}")
print("-" * 82)
for lab in LABELS:
    r = table.loc[lab]
    print(f"{lab:<20}{int(r['companies']):>5}{r['med_payout']:>8.0%}"
          f"{r['med_growth']:>10.1%}{r['annualised']:>9.1%}"
          f"{r['mean_growth']:>10.1%}{r['med_rev']:>10.1%}"
          f"{int(r['no_fy5']):>9}")

payers = grown[grown["payout"] > 0]
normal = payers[payers["payout"] <= 1.0]
rho_all = stats.spearmanr(payers["payout"], payers["growth"])
rho_norm = stats.spearmanr(normal["payout"], normal["growth"])
print(f"\nRank correlation, payout against subsequent earnings growth")
print(f"  all dividend payers          n={len(payers):>3}  "
      f"rho={rho_all.statistic:+.3f}  p={rho_all.pvalue:.3f}")
print(f"  payout of 100% or less       n={len(normal):>3}  "
      f"rho={rho_norm.statistic:+.3f}  p={rho_norm.pvalue:.3f}")

print("\nMedian earnings growth by cohort")
by_cohort = grown.pivot_table(index="bucket", columns="cohort",
                              values="growth", aggfunc="median", observed=True)
print(f"{'Payout at formation':<20}" + "".join(f"{c:>14}" for c in by_cohort.columns))
for lab in LABELS:
    print(f"{lab:<20}" + "".join(f"{by_cohort.loc[lab, c]:>13.1%}"
                                 for c in by_cohort.columns))

# ---------------------------------------------------- 5. chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(LABELS))
ax.bar(x, [table.loc[l, "med_growth"] * 100 for l in LABELS],
       color="#3b82f6", width=0.62, zorder=2)
marks = ["o", "s"]
for i, c in enumerate(by_cohort.columns):
    ax.plot(x, [by_cohort.loc[l, c] * 100 for l in LABELS], marks[i],
            color="#e0e0e0", markersize=5, linestyle="none", label=c, zorder=3)
for i, lab in enumerate(LABELS):
    v = table.loc[lab, "med_growth"] * 100
    top = max([v] + [by_cohort.loc[lab, c] * 100 for c in by_cohort.columns])
    ax.text(i, top + 4.0, f"{v:.0f}%", ha="center", color="#e0e0e0", fontsize=10)
    ax.text(i, -8, f"n={int(table.loc[lab, 'companies'])}", ha="center",
            color="#8a8a8a", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("Median net income growth over five fiscal years (%)")
ax.set_xlabel("Dividend payout ratio at formation")
ax.set_title("Higher payout does not mean slower earnings growth\n"
             "S&P 500 members, fiscal 2014 and fiscal 2019 formation dates",
             color="#e0e0e0", fontsize=12)
ax.axhline(0, color="#3a3a3a", linewidth=0.8)
ax.set_ylim(-14, 135)
ax.legend(frameon=False, labelcolor="#e0e0e0", loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nChart written to {CHART}")
