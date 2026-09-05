# Full write-up: https://xfinlink.com/blog/revenue-concentration-earnings-volatility-python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

Y0, Y1 = 2015, 2024        # fiscal years used for the earnings history
MIN_YEARS = 8              # annual observations required inside that window
OUT_PNG = "revenue-concentration-earnings-volatility-python.png"

# Members at the start of the window as well as today, so the cross-section is
# not restricted to the companies that happened to stay in the index.
ids = sorted(set(xfl.index("sp500", as_of=f"{Y0}-12-31")["entity_id"])
             | set(xfl.index("sp500")["entity_id"]))
print(f"{len(ids)} companies: S&P 500 membership at {Y0}-12-31 plus the current roster")

batches = [ids[i:i + 100] for i in range(0, len(ids), 100)]


def pull(b):
    return xfl.fundamentals(entity_id=b, period_type="annual", start=f"{Y0}-01-01",
                            include_segments=True, max_rows=40000)


with ThreadPoolExecutor(max_workers=3) as ex:
    fund = pd.concat(ex.map(pull, batches), ignore_index=True)
print(f"{len(fund):,} annual filings, {fund['entity_id'].nunique()} companies")

# Earnings volatility: the standard deviation of operating income scaled by total
# assets, across the window. Assets are the comparable base here; revenue is not,
# because a bank, a REIT and a retailer book wildly different revenue against the
# same balance sheet. The result is a pure dispersion number, not a growth rate,
# so it does not blow up when one year's base happens to be small.
fund["roa"] = fund["operating_income"] / fund["total_assets"]
hist = (fund[(fund["fiscal_year"] >= Y0) & (fund["fiscal_year"] <= Y1)]
        .dropna(subset=["roa"])
        .drop_duplicates(["entity_id", "fiscal_year"], keep="last"))
vol = hist.groupby("entity_id").agg(
    years=("roa", "size"), earn_sd=("roa", "std"), earn_mean=("roa", "mean"),
    assets=("total_assets", "mean"), ticker=("ticker", "last"),
    name=("entity_name", "last"), sector=("gics_sector", "last"))
vol = vol[vol["years"] >= MIN_YEARS]
print(f"{len(vol)} companies with at least {MIN_YEARS} annual observations "
      f"in {Y0}-{Y1}")

# Concentration: a Herfindahl index over the revenue shares of the reportable
# operating segments in the most recent annual filing. 1.0 is a single-segment
# company; 0.25 is four equal segments. Business segments are used because they
# are the units management reports results on, and they cover the most companies.
latest = fund.sort_values("fiscal_year").groupby("entity_id").tail(1)


def hhi(segs):
    if not isinstance(segs, list) or not segs:
        return np.nan
    v = np.array([s["value"] for s in segs], dtype=float)
    v = v[v > 0]
    return np.nan if v.size == 0 else float(((v / v.sum()) ** 2).sum())


latest = latest.assign(
    n_segments=latest["segments_business"].map(
        lambda s: len(s) if isinstance(s, list) else 0),
    hhi=latest["segments_business"].map(hhi),
    coverage=pd.to_numeric(latest["segment_coverage"].map(
        lambda c: c.get("business_pct") if isinstance(c, dict) else None),
        errors="coerce"))

# A single reported segment is a Herfindahl of 1.0 by construction. Filings whose
# segment revenues do not reconcile to total revenue are left out.
one = (latest["n_segments"] == 1) & latest["coverage"].between(80, 110)
latest.loc[one, "hhi"] = 1.0
keep = latest[(latest["n_segments"] >= 1) & latest["coverage"].between(80, 110)
              & latest["hhi"].notna()]
print(f"{len(keep)} companies with reconciling segment revenue in the latest "
      f"annual filing ({one.sum()} of them single-segment)")

df = vol.join(keep.set_index("entity_id")[["hhi", "n_segments", "fiscal_year"]],
              how="inner").dropna(subset=["earn_sd", "hhi", "sector"])
df["log_assets"] = np.log(df["assets"])
print(f"{len(df)} companies in the final cross-section, "
      f"{df['sector'].nunique()} sectors")
print()

print(f"{'':22s} {'mean':>8s} {'median':>8s} {'sd':>8s} {'min':>8s} {'max':>8s}")
for col, label in [("hhi", "segment Herfindahl"), ("earn_sd", "earnings volatility"),
                   ("earn_mean", "mean operating ROA")]:
    s = df[col]
    print(f"{label:22s} {s.mean():8.3f} {s.median():8.3f} {s.std():8.3f} "
          f"{s.min():8.3f} {s.max():8.3f}")
print()

# Quartiles of concentration, least to most concentrated.
df["q"] = pd.qcut(df["hhi"], 4, labels=False) + 1
print(f"{'quartile':9s} {'n':>4s} {'mean HHI':>9s} {'segments':>9s} "
      f"{'earn vol':>10s} {'median vol':>11s}")
for q, g in df.groupby("q"):
    print(f"{q:<9d} {len(g):4d} {g['hhi'].mean():9.3f} "
          f"{g['n_segments'].mean():9.1f} {g['earn_sd'].mean():10.4f} "
          f"{g['earn_sd'].median():11.4f}")
print()

# Does concentration survive controls for size and sector?
specs = {
    "concentration only": ["hhi"],
    "+ log assets": ["hhi", "log_assets"],
    "+ log assets + sector": ["hhi", "log_assets"],
}
print(f"{'specification':26s} {'HHI coef':>9s} {'t-stat':>8s} {'p':>7s} {'R2':>7s}")
for label, cols in specs.items():
    X = df[cols].copy()
    if "sector" in label:
        X = X.join(pd.get_dummies(df["sector"], drop_first=True, dtype=float))
    fit = sm.OLS(df["earn_sd"], sm.add_constant(X)).fit()
    print(f"{label:26s} {fit.params['hhi']:9.4f} {fit.tvalues['hhi']:8.2f} "
          f"{fit.pvalues['hhi']:7.3f} {fit.rsquared:7.3f}")
print()

corr_p = df["hhi"].corr(df["earn_sd"])
corr_s = df["hhi"].corr(df["earn_sd"], method="spearman")
print(f"Pearson correlation {corr_p:.3f}, Spearman {corr_s:.3f}")
print()

print(f"{'sector':26s} {'n':>4s} {'mean HHI':>9s} {'earn vol':>10s}")
for sec, g in df.groupby("sector"):
    print(f"{sec:26s} {len(g):4d} {g['hhi'].mean():9.3f} {g['earn_sd'].mean():10.4f}")

# Chart --------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f"})
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df["hhi"], df["earn_sd"], s=22, alpha=0.55, color="#3b82f6",
           edgecolors="none", zorder=2)
xs = np.linspace(df["hhi"].min(), 1.0, 50)
b, a = np.polyfit(df["hhi"], df["earn_sd"], 1)
ax.plot(xs, a + b * xs, color="#f59e0b", lw=2.2, zorder=4,
        label=f"fitted: {a:.3f} + {b:.3f} x concentration")
qm = df.groupby("q").agg(x=("hhi", "mean"), y=("earn_sd", "mean"))
ax.plot(qm["x"], qm["y"], "o-", color="#e0e0e0", ms=9, lw=1.4, zorder=5,
        label="quartile means")
ax.set_xlabel("Revenue concentration across reporting segments "
              "(Herfindahl, 1.0 = single segment)")
ax.set_ylabel("Earnings volatility, 2015-2024 (sd of operating income / assets)")
ax.set_title("Concentrated revenue barely moves earnings volatility\n"
             f"{len(df)} S&P 500 companies, segments from the latest annual "
             "filing", color="#e0e0e0", fontsize=12)
ax.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0",
          fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {OUT_PNG}")
