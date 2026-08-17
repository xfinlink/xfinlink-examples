# Full write-up: https://xfinlink.com/blog/rd-intensity-forward-revenue-growth-python
"""Does heavy R&D spending show up as faster revenue growth three years later?

R&D intensity (research spend / revenue) is measured in year t, revenue growth is
measured from t to t+3, so the predictor is always fixed before the outcome.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

HORIZON = 3

members = xfl.index("sp500")
ids = members["entity_id"].dropna().astype(int).tolist()
frames = []
for i in range(0, len(ids), 100):
    frames.append(xfl.fundamentals(
        entity_id=ids[i:i + 100], period_type="annual", start="2016-01-01", end="2026-06-30",
        fields=["revenue", "research_and_development", "gics_sector"]))
f = pd.concat(frames, ignore_index=True)
f["period_end"] = pd.to_datetime(f["period_end"])

# Map each fiscal year to the calendar year it mostly covers, so companies with
# January and June year-ends line up with December filers.
f["cy"] = f["period_end"].dt.year - (f["period_end"].dt.month <= 6).astype(int)
f = f[f["revenue"] > 0].sort_values(["entity_id", "cy"]).drop_duplicates(["entity_id", "cy"], keep="last")

rev = f.pivot(index="cy", columns="entity_id", values="revenue")
rd = f.pivot(index="cy", columns="entity_id", values="research_and_development")
sector = f.groupby("entity_id")["gics_sector"].last()

obs = []
for t in range(2017, 2026 - HORIZON):
    if t not in rev.index or (t + HORIZON) not in rev.index:
        continue
    d = pd.DataFrame({
        "intensity": rd.loc[t] / rev.loc[t],
        "growth": (rev.loc[t + HORIZON] / rev.loc[t]) ** (1 / HORIZON) - 1,
    }).dropna()
    obs.append(d[(d["intensity"] > 0) & (d["intensity"] < 1)].assign(t=t))
o = pd.concat(obs).reset_index().rename(columns={"index": "entity_id"})
o["gics_sector"] = o["entity_id"].map(sector)

# Quintiles are formed inside each starting year, so no year's growth dominates.
o["q"] = o.groupby("t")["intensity"].transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
tab = o.groupby("q").agg(n=("growth", "size"), rd_intensity=("intensity", "median"),
                         median_cagr=("growth", "median"), mean_cagr=("growth", "mean"))
print(f"pooled observations: {len(o)}, companies: {o['entity_id'].nunique()}")
print("\nR&D intensity quintile -> revenue CAGR over the next 3 years (percent)")
print(pd.concat([tab[["n"]], (tab[["rd_intensity", "median_cagr", "mean_cagr"]] * 100).round(2)],
                axis=1).to_string())

rho, p = stats.spearmanr(o["intensity"], o["growth"])
print(f"\npooled Spearman rho = {rho:.3f} (p = {p:.1e}, n = {len(o)})")

print("\nBy starting year")
for t, g in o.groupby("t"):
    q1, q5 = g[g["q"] == 1]["growth"].median(), g[g["q"] == 5]["growth"].median()
    r, pp = stats.spearmanr(g["intensity"], g["growth"])
    print(f"  {t}->{t + HORIZON}  n={len(g):3d}  Q1 {q1 * 100:5.2f}%  Q5 {q5 * 100:5.2f}%"
          f"  spread {(q5 - q1) * 100:+5.2f}pp  rho {r:+.3f}")

print("\nWithin sector (40 or more observations)")
for s, g in o.groupby("gics_sector"):
    if len(g) < 40:
        continue
    r, pp = stats.spearmanr(g["intensity"], g["growth"])
    print(f"  {s:<24} n={len(g):3d}  median intensity {g['intensity'].median() * 100:5.2f}%"
          f"  rho {r:+.3f} (p={pp:.3f})")

# ---- chart ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
qs = tab.index.tolist()
ax.bar(qs, tab["median_cagr"] * 100, width=0.62, color="#3b82f6", zorder=2)
for t, g in o.groupby("t"):
    ym = g.groupby("q")["growth"].median() * 100
    ax.plot(ym.index, ym.values, color="#9aa4b2", lw=0.8, alpha=0.55, marker="o",
            markersize=3, zorder=3)
ax.set_xticks(qs)
ax.set_xticklabels([f"Q{q}\n{v * 100:.1f}%" for q, v in zip(qs, tab["rd_intensity"])])
ax.set_xlabel("R&D intensity quintile, with the quintile's median R&D spend as a share of revenue")
ax.set_ylabel("Revenue growth over the next 3 years (percent a year)")
ax.set_title("Heavier research spending precedes faster revenue growth")
ax.axhline(0, color="#3a3a3a", lw=0.8)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
ax.grid(axis="y", color="#1f1f1f", lw=0.7, zorder=0)
plt.tight_layout()
plt.savefig("rd-intensity-forward-revenue-growth-python.png", dpi=150)
