# Full write-up: https://xfinlink.com/blog/does-high-roic-persist-fade-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF = "2015-12-31"
BASE, LAST = 2015, 2024

# Membership as it stood at the end of 2015. Ranking on today's roster would
# quietly drop every company that was acquired or failed in between, and those
# are disproportionately the ones whose returns faded.
members = xfl.index("sp500", as_of=AS_OF)
ids = sorted(members["entity_id"].dropna().astype(int).unique())
print(f"{len(ids)} companies in the index at {AS_OF}")

CHUNK = 150
m = pd.concat([
    xfl.metrics(entity_id=ids[i:i + CHUNK], period_type="annual",
                start=f"{BASE}-01-01", end=f"{LAST}-12-31", fields=["roic"],
                max_rows=50000)
    for i in range(0, len(ids), CHUNK)], ignore_index=True)
m["year"] = pd.to_datetime(m["period_end"]).dt.year
m = m[(m["year"] >= BASE) & (m["year"] <= LAST)]
panel = m.pivot_table(index="entity_id", columns="year", values="roic")
print(f"{len(panel)} companies with at least one annual figure, "
      f"{BASE} to {LAST}")

base = panel[BASE].dropna()
print(f"{len(base)} ranked on {BASE} return on invested capital")
print()

q = pd.qcut(base, 5, labels=[1, 2, 3, 4, 5])
print(f"{'quintile':9s} {'n':>4s} {'2015':>7s}" +
      "".join(f"{y:>7d}" for y in range(BASE + 1, LAST + 1)))
paths = {}
for lab in [5, 4, 3, 2, 1]:
    ent = q[q == lab].index
    row = [panel.loc[panel.index.intersection(ent), y].median() * 100
           for y in range(BASE, LAST + 1)]
    paths[lab] = row
    print(f"Q{lab:<8d} {len(ent):4d} " +
          "".join(f"{v:6.1f}%" for v in row))
print()
print("median return on invested capital, %, by 2015 quintile")
print()

top, bot = paths[5], paths[1]
print(f"top-minus-bottom spread: {top[0] - bot[0]:.1f}pp in {BASE}, "
      f"{top[-1] - bot[-1]:.1f}pp in {LAST} "
      f"({(top[-1] - bot[-1]) / (top[0] - bot[0]) * 100:.0f}% of the original)")
print(f"top quintile median fell {top[0] - top[-1]:.1f}pp; "
      f"bottom quintile median rose {bot[-1] - bot[0]:.1f}pp")
print()

# Survival of the ranking itself, base year against final year.
end = panel[LAST].dropna()
both = base.index.intersection(end.index)
q_end = pd.qcut(end[both], 5, labels=[1, 2, 3, 4, 5])
q_base = pd.qcut(base[both], 5, labels=[1, 2, 3, 4, 5])
print(f"{len(both)} companies ranked in both {BASE} and {LAST} "
      f"({len(both) / len(base) * 100:.0f}% of the original cross-section)")
trans = pd.crosstab(q_base, q_end, normalize="index") * 100
print()
print("where each 2015 quintile sat in 2024 (row %, quintile 5 = highest)")
print("2015      " + "".join(f"{f'Q{c}':>8s}" for c in trans.columns))
for r in trans.index:
    print(f"Q{r:<8d} " + "".join(f"{trans.loc[r, c]:7.1f}%"
                                 for c in trans.columns))
print()
stay = trans.loc[5, 5]
print(f"{stay:.1f}% of the 2015 top quintile was still top quintile in {LAST}")
print(f"rank correlation {BASE} vs {LAST}: "
      f"{base[both].corr(end[both], method='spearman'):.3f}")

# Chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f"})
fig, ax = plt.subplots(figsize=(10, 6))
years = list(range(BASE, LAST + 1))
shades = {5: "#3b82f6", 4: "#60a5fa", 3: "#9ca3af", 2: "#f59e0b", 1: "#ef4444"}
for lab in [5, 4, 3, 2, 1]:
    ax.plot(years, paths[lab], color=shades[lab], lw=2,
            marker="o", ms=3.5, label=f"Q{lab}")
    dy = {1: 5, 2: -11}.get(lab, -3)
    ax.annotate(f"Q{lab}", (years[-1], paths[lab][-1]),
                textcoords="offset points", xytext=(8, dy),
                color=shades[lab], fontsize=9.5)
ax.set_xlabel("Year")
ax.set_ylabel("Median return on invested capital (%)")
ax.set_title("High returns on capital fade, and the spread closes by two thirds\n"
             "S&P 500 members as of December 2015, sorted into quintiles that year",
             color="#e0e0e0", fontsize=12)
ax.set_xlim(BASE - 0.2, LAST + 0.7)
ax.grid(axis="y", color="#1f1f1f", lw=0.8)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("does-high-roic-persist-fade-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart written")
