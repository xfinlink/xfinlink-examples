# Full write-up: https://xfinlink.com/blog/high-margin-screen-sector-bet-python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

Y0, Y1 = 2014, 2023
EXCLUDE = {"Financials", "Real Estate"}
FIELDS = ["revenue", "gross_profit", "operating_income", "net_income", "total_assets"]
METRICS = ["Gross margin", "Operating margin", "Net margin", "Return on assets"]
OUT_PNG = "high-margin-screen-sector-bet-python.png"

# Every company that sat in the index at any year end in the window, so the
# cross-section is not restricted to the names that survived in it.
ids = set()
for y in range(Y0 - 1, Y1 + 1):
    ids |= set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])
ids = sorted(ids)
batches = [ids[i:i + 100] for i in range(0, len(ids), 100)]


def pull(b):
    return xfl.fundamentals(entity_id=b, period_type="annual", fields=FIELDS,
                            start=f"{Y0}-01-01", end=f"{Y1 + 1}-06-30", max_rows=40000)


with ThreadPoolExecutor(max_workers=6) as ex:
    fund = pd.concat(ex.map(pull, batches), ignore_index=True)

d = fund[fund["fiscal_year"].between(Y0, Y1)].drop_duplicates(
    ["entity_id", "fiscal_year"], keep="last")
d = d[~d["gics_sector"].isin(EXCLUDE)].dropna(subset=FIELDS + ["gics_sector"])
d = d[(d["revenue"] > 0) & (d["total_assets"] > 0)].copy()

d["Gross margin"] = d["gross_profit"] / d["revenue"]
d["Operating margin"] = d["operating_income"] / d["revenue"]
d["Net margin"] = d["net_income"] / d["revenue"]
d["Return on assets"] = d["operating_income"] / d["total_assets"]
for m in METRICS:                       # trim the 1% tails inside each year
    d[m] = d.groupby("fiscal_year")[m].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))

print(f"{d['entity_id'].nunique()} companies, {len(d):,} company-years, "
      f"fiscal {Y0}-{Y1}, {d['gics_sector'].nunique()} sectors")


def eta_sq(sector, v):
    """Share of the cross-sectional variance of v that sits between sectors."""
    grand = v.mean()
    between = sum(len(x) * (x.mean() - grand) ** 2 for _, x in v.groupby(sector))
    return between / ((v - grand) ** 2).sum()


rows = []
for y, g in d.groupby("fiscal_year"):
    r = {"fiscal_year": y}
    for m in METRICS:
        r[m + " (value)"] = eta_sq(g["gics_sector"], g[m])
        r[m] = eta_sq(g["gics_sector"], g[m].rank(pct=True))
    rows.append(r)
expl = pd.DataFrame(rows).set_index("fiscal_year")

print("\nShare of cross-sectional variance explained by sector, fiscal "
      f"{Y0}-{Y1}")
print(f"{'':<18}{'ranks: mean':>12}{'min':>8}{'max':>8}{'values: mean':>14}")
for m in METRICS:
    print(f"{m:<18}{expl[m].mean():>11.1%}{expl[m].min():>8.1%}"
          f"{expl[m].max():>8.1%}{expl[m + ' (value)'].mean():>14.1%}")

base = d["gics_sector"].value_counts(normalize=True)
print(f"\nSector mix of the top decile, pooled {Y0}-{Y1} (sample share in brackets)")
tilt = {}
for m in METRICS:
    top = d[d.groupby("fiscal_year")[m].rank(pct=True, ascending=False) <= 0.10]
    share = top["gics_sector"].value_counts(normalize=True)
    tilt[m] = share
    lead = share.head(2)
    print(f"{m:<18}" + "  ".join(
        f"{s} {v:.0%} [{base[s]:.0%}]" for s, v in lead.items())
        + f"   top two {lead.sum():.0%} vs {base[lead.index].sum():.0%}")

print("\nNames kept when the same screen is ranked inside each sector")
for m in METRICS:
    keep = []
    for y, g in d.groupby("fiscal_year"):
        raw = set(g.index[g[m].rank(pct=True, ascending=False) <= 0.10])
        neutral = set(g.index[g.groupby("gics_sector")[m].rank(
            pct=True, ascending=False) <= 0.10])
        keep.append(len(raw & neutral) / len(raw))
    print(f"{m:<18}{np.mean(keep):>7.1%}  (year range {min(keep):.0%}-{max(keep):.0%})")

# ---- chart ----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
colours = ["#3b82f6", "#f59e0b", "#10b981", "#a78bfa"]
for m, c in zip(METRICS, colours):
    ax1.plot(expl.index, expl[m] * 100, marker="o", ms=4, color=c, label=m)
ax1.set_ylabel("Percent of spread explained by sector")
ax1.set_xlabel("Fiscal year")
ax1.set_title("How much of the profitability spread is sector membership?",
              color="#e0e0e0")
ax1.set_ylim(0, 30)
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           ncol=4, fontsize=9)

order = base.index.tolist()
x = np.arange(len(order))
ax2.bar(x - 0.2, [base[s] * 100 for s in order], 0.4, color="#4b5563",
        label="Share of the sample")
ax2.bar(x + 0.2, [tilt["Gross margin"].get(s, 0) * 100 for s in order], 0.4,
        color="#3b82f6", label="Share of the top decile by gross margin")
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace(" ", "\n") for s in order], fontsize=8)
ax2.set_ylabel("Percent of companies")
ax2.set_title("The top decile is tilted even though sector explains little",
              color="#e0e0e0")
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           fontsize=9)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
