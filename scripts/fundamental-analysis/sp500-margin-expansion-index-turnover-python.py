# Full write-up: https://xfinlink.com/blog/sp500-margin-expansion-index-turnover-python
"""How much of the rise in the S&P 500 aggregate net profit margin comes from the
companies in the index earning more, and how much from the index changing which
companies it holds? Point-in-time year-end rosters 2014 to 2025, annual filings."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2014, 2026))
OUT_PNG = "sp500-margin-expansion-index-turnover-python.png"

# 1. Point-in-time roster at each year end, carried by entity_id so a reassigned
#    symbol cannot substitute one company for another inside the panel.
rosters = {y: set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"]) for y in YEARS}
entity_ids = sorted(set().union(*rosters.values()))

frames = []
for i in range(0, len(entity_ids), 100):
    frames.append(xfl.fundamentals(
        entity_id=entity_ids[i:i + 100], period_type="annual",
        start="2013-06-01", end="2026-06-30",
        fields=["revenue", "net_income"]))
raw = pd.concat(frames, ignore_index=True)

# 2. Assign each annual report to a calendar year: a fiscal year ending in June or
#    later belongs to that calendar year, one ending January to May to the year before.
df = raw.copy()
df["year"] = np.where(df["period_end"].dt.month >= 6,
                      df["period_end"].dt.year, df["period_end"].dt.year - 1)
df = df[df["year"].between(2014, 2025)]
df = df.sort_values(["entity_id", "year", "period_end"])
df = df.drop_duplicates(["entity_id", "year"], keep="last")
df = df[df["revenue"].notna() & (df["revenue"] > 0) & df["net_income"].notna()]

sector = df.groupby("entity_id")["gics_sector"].last()
rev = df.pivot(index="entity_id", columns="year", values="revenue")
inc = df.pivot(index="entity_id", columns="year", values="net_income")


def decompose(keep):
    """Aggregate margin by year, split into continuing members and roster change."""
    r, n = rev.loc[keep], inc.loc[keep]
    memb = {y: sorted(i for i in rosters[y] if i in r.index and not np.isnan(r.at[i, y]))
            for y in YEARS}
    lvl = {y: 100 * n.loc[memb[y], y].sum() / r.loc[memb[y], y].sum() for y in YEARS}
    rows = []
    for y in YEARS[1:]:
        stay = [i for i in memb[y] if i in memb[y - 1]]
        within = 100 * (n.loc[stay, y].sum() / r.loc[stay, y].sum()
                        - n.loc[stay, y - 1].sum() / r.loc[stay, y - 1].sum())
        rows.append({"year": y, "n": len(memb[y]), "stay": len(stay),
                     "rev": r.loc[memb[y], y].sum(), "inc": n.loc[memb[y], y].sum(),
                     "margin": lvl[y], "total": lvl[y] - lvl[y - 1], "within": within,
                     "roster": lvl[y] - lvl[y - 1] - within})
    out = pd.DataFrame(rows)
    first = {"year": 2014, "n": len(memb[2014]), "stay": np.nan,
             "rev": r.loc[memb[2014], 2014].sum(), "inc": n.loc[memb[2014], 2014].sum(),
             "margin": lvl[2014], "total": np.nan, "within": np.nan, "roster": np.nan}
    return pd.concat([pd.DataFrame([first]), out], ignore_index=True), memb


full, members = decompose(rev.index)

print(f"S&P 500 year-end rosters {YEARS[0]}-{YEARS[-1]}: {len(entity_ids)} distinct entities, "
      f"annual rows retrieved: {len(raw):,}")
print(f"Company-years with a usable revenue and net income line: {len(df):,}\n")

print("Year   Members   Revenue $bn   Net income $bn   Net margin %   Same-member path %")
path = [full.loc[0, "margin"]]
for _, r in full.iloc[1:].iterrows():
    path.append(path[-1] + r["within"])
for (_, r), p in zip(full.iterrows(), path):
    print(f"{r.year:.0f}       {r.n:3.0f}      {r.rev / 1000:9,.0f}   "
          f"{r.inc / 1000:12,.0f}   {r.margin:11.2f}   {p:17.2f}")

tot = full.loc[full.year == 2025, "margin"].iloc[0] - full.loc[full.year == 2014, "margin"].iloc[0]
wit = full["within"].sum()
print(f"\nYear   Continuing members   Change in margin (pp)   Continuing (pp)   Roster change (pp)")
for _, r in full.iloc[1:].iterrows():
    print(f"{r.year:.0f}              {r.stay:3.0f}                   {r.total:9.2f}         "
          f"{r.within:9.2f}            {r.roster:9.2f}")
print(f"Total                                   {tot:9.2f}         {wit:9.2f}            {tot - wit:9.2f}")
print(f"\nRoster change accounts for {100 * (tot - wit) / tot:.1f}% of the margin expansion; "
      f"its yearly contribution is positive in {(full['roster'] > 0).sum()} of {len(full) - 1} years")

# 3. Mechanism: the profitability of the companies entering against those leaving.
jn = jr = ln = lr = 0.0
jc = lc = 0
for y in YEARS[1:]:
    join = [i for i in members[y] if i not in rosters[y - 1]]
    leave = [i for i in members[y - 1] if i not in rosters[y]]
    jn += inc.loc[join, y].sum(); jr += rev.loc[join, y].sum(); jc += len(join)
    ln += inc.loc[leave, y - 1].sum(); lr += rev.loc[leave, y - 1].sum(); lc += len(leave)
print(f"\nEntering the index ({jc} entries): net margin {100 * jn / jr:5.2f}%, "
      f"mean revenue ${jr / jc:,.0f}m")
print(f"Leaving the index  ({lc} exits):   net margin {100 * ln / lr:5.2f}%, "
      f"mean revenue ${lr / lc:,.0f}m")

# 4. Same question from the growth side: chain the index aggregate against a chain
#    that only ever compares a company with itself.
chain = {"revenue": [1.0, 1.0], "net income": [1.0, 1.0]}
for y in YEARS[1:]:
    stay = [i for i in members[y] if i in members[y - 1]]
    for k, tab in (("revenue", rev), ("net income", inc)):
        chain[k][0] *= tab.loc[members[y], y].sum() / tab.loc[members[y - 1], y - 1].sum()
        chain[k][1] *= tab.loc[stay, y].sum() / tab.loc[stay, y - 1].sum()
print("\nCumulative 2014 to 2025          Index aggregate   Same-member chain")
for k, (a, b) in chain.items():
    print(f"  {k:<28}    {100 * (a - 1):11.1f}%     {100 * (b - 1):11.1f}%")

# 5. Robustness: financial and property companies report revenue on a different basis.
keep = [i for i in rev.index if sector.get(i) not in ("Financials", "Real Estate")]
ex, _ = decompose(keep)
etot = ex.loc[ex.year == 2025, "margin"].iloc[0] - ex.loc[ex.year == 2014, "margin"].iloc[0]
ewit = ex["within"].sum()
print(f"\nExcluding Financials and Real Estate ({len(keep)} entities): margin "
      f"{ex.loc[0, 'margin']:.2f}% to {ex.loc[len(ex) - 1, 'margin']:.2f}%, "
      f"total {etot:+.2f}pp = continuing {ewit:+.2f}pp + roster change {etot - ewit:+.2f}pp "
      f"({100 * (etot - ewit) / etot:.1f}%)")

# 6. Chart: the reported margin path against the path the same companies produced.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [3, 2]})

ax1.plot(full["year"], full["margin"], color="#3b82f6", linewidth=2.2, marker="o",
         markersize=5, label="S&P 500 aggregate net margin")
ax1.plot(full["year"], path, color="#9ca3af", linewidth=2.0, linestyle="--",
         label="Path from continuing members only")
ax1.annotate(f"{tot - wit:.2f}pp", xy=(2025, (full["margin"].iloc[-1] + path[-1]) / 2),
             xytext=(2025.15, (full["margin"].iloc[-1] + path[-1]) / 2),
             color="#e0e0e0", fontsize=11, va="center")
ax1.set_ylabel("Net profit margin (%)")
ax1.set_xlim(2013.7, 2025.9)
ax1.legend(frameon=False, loc="upper left")
ax1.set_title("S&P 500 margin expansion: earned by its members, or bought by changing them",
              color="#e0e0e0", fontsize=13, pad=12)

ax2.bar(full["year"].iloc[1:], full["roster"].iloc[1:], color="#3b82f6", width=0.6)
ax2.axhline(0, color="#e0e0e0", linewidth=1)
ax2.set_ylabel("Roster change (pp)")
ax2.set_xlabel("Year")
ax2.set_xlim(2013.7, 2025.9)

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
plt.tight_layout()
fig.subplots_adjust(left=0.085, right=0.93)
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nChart saved to {OUT_PNG}")
