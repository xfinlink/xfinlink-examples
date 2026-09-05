# Full write-up: https://xfinlink.com/blog/does-large-goodwill-predict-writedowns-python
"""Does a large goodwill balance predict a future writedown?

Sorts point-in-time S&P 500 members on goodwill as a share of total assets and
measures how often each bucket books a material goodwill writedown over the
following three fiscal years.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(60)

YEARS = list(range(2012, 2025))
FIELDS = ["goodwill", "impairment_charges", "total_assets", "total_equity"]
PNG = "does-large-goodwill-predict-writedowns-python.png"

# --- universe: union of point-in-time rosters, carried by entity id ----------
ids = set()
for y in YEARS:
    ids |= set(xfl.index("sp500", as_of=f"{y}-12-31")["entity_id"])

excluded = set()
for sector in ["Financials", "Real Estate"]:
    offset = 0
    while True:
        page = xfl.search(gics_sector=sector, limit=500, offset=offset)
        if page.empty:
            break
        excluded |= set(page["entity_id"])
        if len(page) < 500:
            break
        offset += 500

universe = sorted(int(i) for i in (ids - excluded))


def pull(batch, tries=3):
    """Fetch one batch, retrying before splitting it."""
    for attempt in range(tries):
        try:
            return xfl.fundamentals(entity_id=batch, period_type="annual",
                                    fields=FIELDS, start="2011-01-01",
                                    end="2025-12-31", max_rows=200000)
        except xfl.XfinlinkError:
            if attempt < tries - 1:
                time.sleep(3)
            elif len(batch) > 5:
                half = len(batch) // 2
                return pd.concat([pull(batch[:half]), pull(batch[half:])],
                                 ignore_index=True)
            else:
                return pd.DataFrame()


p = pd.concat([pull(universe[i:i + 25]) for i in range(0, len(universe), 25)],
              ignore_index=True)

# --- one row per company per fiscal year ------------------------------------
p["period_end"] = pd.to_datetime(p["period_end"])
p["fy"] = np.where(p["period_end"].dt.month >= 6,
                   p["period_end"].dt.year, p["period_end"].dt.year - 1)
p = (p.sort_values("period_end")
       .drop_duplicates(["entity_id", "fy"], keep="last")
       .sort_values(["entity_id", "fy"]))

raw_rows = len(p)
p["goodwill"] = p["goodwill"].fillna(0.0)
p = p[(p["total_assets"] >= 100) & (p["goodwill"] >= 0)
      & (p["goodwill"] <= p["total_assets"])].copy()

# --- writedown events -------------------------------------------------------
p["charge"] = p["impairment_charges"].abs()
grp = p.groupby("entity_id")
p["gw_prev"] = grp["goodwill"].shift(1)
p["ta_prev"] = grp["total_assets"].shift(1)
linked = (grp["fy"].shift(1).eq(p["fy"] - 1) & p["gw_prev"].notna()
          & p["ta_prev"].ge(100))
p["gw_fall"] = np.where(linked, (p["gw_prev"] - p["goodwill"]) / p["ta_prev"], np.nan)
p["charge_ta"] = np.where(linked, p["charge"] / p["ta_prev"], np.nan)
p.loc[p["charge_ta"] > 1, "charge_ta"] = np.nan
p["writedown"] = (p["gw_fall"] >= 0.01) & (p["charge_ta"] >= 0.005)
p["strict"] = (p["gw_fall"] >= 0.02) & (p["charge_ta"] >= 0.01)

# --- cohort: signal at fiscal year t, outcome over t+1 to t+3 ---------------
fwd = p[["entity_id", "fy", "writedown", "strict", "charge_ta"]]
d = p[p["fy"].between(2012, 2021)][
    ["entity_id", "fy", "ticker", "entity_name", "gics_sector",
     "goodwill", "total_assets", "total_equity"]].copy()
d["gw_int"] = d["goodwill"] / d["total_assets"]

for k in (1, 2, 3):
    nxt = fwd.assign(fy=fwd["fy"] - k).rename(
        columns={"writedown": f"wd{k}", "strict": f"st{k}", "charge_ta": f"sz{k}"})
    d = d.merge(nxt, on=["entity_id", "fy"], how="left", indicator=f"seen{k}")
    d[f"sz{k}"] = d[f"sz{k}"].where(d[f"wd{k}"] == True)

d = d[(d["seen1"] == "both") & (d["seen2"] == "both") & (d["seen3"] == "both")]
d["wd3y"] = d[["wd1", "wd2", "wd3"]].fillna(False).any(axis=1)
d["st3y"] = d[["st1", "st2", "st3"]].fillna(False).any(axis=1)
d["size3y"] = d[["sz1", "sz2", "sz3"]].max(axis=1)

d["q"] = d.groupby("fy")["gw_int"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int))

tab = d.groupby("q").agg(n=("wd3y", "size"), med_int=("gw_int", "median"),
                         rate1=("wd1", "mean"), rate3=("wd3y", "mean"),
                         strict3=("st3y", "mean"), med_size=("size3y", "median"))

model = sm.Logit(d["wd3y"].astype(int),
                 sm.add_constant(pd.DataFrame({
                     "gw_int": d["gw_int"],
                     "log_assets": np.log(d["total_assets"])}))).fit(disp=0)

# --- output -----------------------------------------------------------------
print(f"Point-in-time S&P 500 rosters {YEARS[0]}-{YEARS[-1]}, "
      "Financials and Real Estate excluded")
print(f"  company-years pulled: {raw_rows:,}   after plausibility screen: {len(p):,}")
print(f"  cohort company-years with a full 3-year forward window: {len(d):,} "
      f"from {d['entity_id'].nunique()} companies")
print(f"  material writedown events in the panel: {int(p['writedown'].sum())}")
print(f"  base rate, any writedown within 3 years: {100 * d['wd3y'].mean():.1f}%\n")

print("Goodwill / total assets quintile at fiscal year end")
print("  q   n     median goodwill/assets   writedown 1y   writedown 3y   "
      "strict 3y   median charge (% of assets)")
for q, r in tab.iterrows():
    print(f"  Q{q}  {int(r['n']):<5} {r['med_int']:>12.3f} "
          f"{100 * r['rate1']:>16.1f}% {100 * r['rate3']:>13.1f}% "
          f"{100 * r['strict3']:>11.1f}% {100 * r['med_size']:>14.1f}%")

low = d["gw_int"] < 0.01
print(f"\n  goodwill under 1% of assets: {int(low.sum()):,} company-years, "
      f"3-year writedown rate {100 * d.loc[low, 'wd3y'].mean():.1f}%")
print(f"  goodwill at or above 1%:     {int((~low).sum()):,} company-years, "
      f"3-year writedown rate {100 * d.loc[~low, 'wd3y'].mean():.1f}%")

print("\nLogit, writedown within 3 years")
print(model.summary2().tables[1].round(4).to_string())
print(f"  pseudo R-squared {model.prsquared:.4f}, n {int(model.nobs):,}")

at = model.predict(sm.add_constant(pd.DataFrame({
    "gw_int": [0.05, 0.40],
    "log_assets": np.log(d["total_assets"].median())}), has_constant="add"))
print(f"  fitted probability at median company size: "
      f"{100 * at.iloc[0]:.1f}% when goodwill is 5% of assets, "
      f"{100 * at.iloc[1]:.1f}% when it is 40%")

se = np.sqrt(tab["rate3"] * (1 - tab["rate3"]) / tab["n"])
gap = tab.loc[4, "rate3"] - tab.loc[5, "rate3"]
print(f"  Q4 minus Q5 three-year rate: {100 * gap:+.1f} points, "
      f"z = {gap / np.sqrt(se[4] ** 2 + se[5] ** 2):.2f}")

print("\nLargest events by dollar charge")
big = p[p["writedown"]].nlargest(8, "charge")
for _, r in big.iterrows():
    print(f"  {r['ticker']:<6} FY{int(r['fy'])}  goodwill {r['gw_prev']:>9,.0f} -> "
          f"{r['goodwill']:>9,.0f}   charge {r['charge']:>9,.0f}m "
          f"({100 * r['charge_ta']:.1f}% of assets)")

# --- chart ------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = np.arange(5)
ax1.bar(x - 0.2, 100 * tab["rate3"], 0.4, color="#3b82f6", label="within 3 years")
ax1.bar(x + 0.2, 100 * tab["rate1"], 0.4, color="#6b7280", label="within 1 year")
for i, v in enumerate(100 * tab["rate3"]):
    ax1.text(x[i] - 0.2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels([f"Q{q}\n{r.med_int:.0%} of assets"
                     for q, r in tab.iterrows()])
ax1.set_ylabel("Companies booking a writedown (%)")
ax1.set_title("Goodwill intensity and the chance of a later writedown, "
              "S&P 500 members 2012-2021", color="#e0e0e0")
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")
ax1.spines[["top", "right"]].set_visible(False)

d["bin"] = pd.qcut(d["gw_int"].rank(method="first"), 12, labels=False)
obs = d.groupby("bin").agg(gw=("gw_int", "median"), rate=("wd3y", "mean"))
grid = np.linspace(0, d["gw_int"].quantile(0.99), 100)
fit = model.predict(sm.add_constant(pd.DataFrame({
    "gw_int": grid, "log_assets": np.log(d["total_assets"].median())}),
    has_constant="add"))
ax2.scatter(100 * obs["gw"], 100 * obs["rate"], color="#3b82f6", s=40,
            label="observed, twelve equal groups")
ax2.plot(100 * grid, 100 * fit, color="#e0e0e0", lw=1.5, ls="--",
         label="logit fit at median company size")
ax2.set_xlabel("Goodwill as a share of total assets (%)")
ax2.set_ylabel("Writedown within 3 years (%)")
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {PNG}")
