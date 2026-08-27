# Full write-up: https://xfinlink.com/blog/sector-revenue-beta-vs-stock-beta-python
"""Which sectors are really cyclical: revenue betas vs stock betas.

Builds a quarterly revenue-cycle series for each GICS sector from the
point-in-time S&P 500 rosters (2018Q3-2026Q1), regresses each sector on the
rest of the corporate sector to get a revenue beta, and compares that with the
sector ETF's stock beta against SPY over the same window.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

QUARTERS = pd.period_range("2018Q3", "2026Q1", freq="Q")
ETF = {"Energy": "XLE", "Consumer Discretionary": "XLY", "Materials": "XLB",
       "Industrials": "XLI", "Information Technology": "XLK", "Utilities": "XLU",
       "Communication Services": "XLC", "Health Care": "XLV",
       "Consumer Staples": "XLP"}
COVID = {pd.Period(q) for q in ["2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"]}

# --- point-in-time membership, one roster per quarter end -------------------
roster = {q: set(xfl.index("sp500", as_of=q.end_time.date().isoformat())["entity_id"])
          for q in QUARTERS}
ids = sorted(set().union(*roster.values()))

# --- quarterly revenue, addressed by entity id so ticker changes do not break
# the panel -----------------------------------------------------------------
rev = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="quarterly",
                                  fields=["revenue"], start="2017-06-01",
                                  end="2026-08-27", max_rows=200000)
                 for i in range(0, len(ids), 100)], ignore_index=True)

rev = rev[rev["gics_sector"].notna() & (rev["revenue"] > 0)]
rev = rev[~rev["gics_sector"].isin(["Financials", "Real Estate"])]

# Fiscal periods drift: Apple's Q3 ends 1 July, its Q4 ends 30 September, and
# both land in calendar Q3. Shift by 45 days to place each period in the
# calendar quarter it actually covers.
rev["cq"] = (rev["period_end"] - pd.Timedelta(days=45)).dt.to_period("Q")
rev["as_filed"] = (rev["source"] == "filing").astype(int)
rev = (rev.sort_values(["entity_id", "cq", "as_filed", "period_end"])
          .drop_duplicates(["entity_id", "cq"], keep="last"))

prior = rev[["entity_id", "cq", "revenue"]].rename(columns={"revenue": "rev_prior"})
prior["cq"] = prior["cq"] + 4
panel = rev.merge(prior, on=["entity_id", "cq"], how="inner")
panel["growth"] = panel["revenue"] / panel["rev_prior"] - 1

# --- sector line = median member growth; cycle line = every other sector ----
rows = []
for q in QUARTERS:
    sub = panel[(panel["cq"] == q) & (panel["entity_id"].isin(roster[q]))]
    for sector, g in sub.groupby("gics_sector"):
        rows.append({"cq": q, "sector": sector, "n": len(g),
                     "sector_growth": g["growth"].median(),
                     "cycle_growth": sub.loc[sub["gics_sector"] != sector, "growth"].median()})
    rows.append({"cq": q, "sector": "ALL", "n": len(sub),
                 "sector_growth": sub["growth"].median(), "cycle_growth": np.nan})
cycle = pd.DataFrame(rows)

# --- stock betas over the same window --------------------------------------
funds = list(ETF.values()) + ["SPY"]
px = pd.concat([xfl.prices(funds[i:i + 3], start="2018-07-01", end="2026-03-31",
                           fields=["return_daily"], max_rows=200000)
                for i in range(0, len(funds), 3)], ignore_index=True)
ret = px.pivot(index="date", columns="ticker", values="return_daily").dropna()

out = []
for sector, g in cycle[cycle["sector"] != "ALL"].groupby("sector"):
    fit = sm.OLS(g["sector_growth"].values,
                 sm.add_constant(g["cycle_growth"].values)).fit(
                     cov_type="HAC", cov_kwds={"maxlags": 4})
    g_ex = g[~g["cq"].isin(COVID)]
    ex = sm.OLS(g_ex["sector_growth"].values,
                sm.add_constant(g_ex["cycle_growth"].values)).fit().params[1]
    stock = sm.OLS(ret[ETF[sector]].values,
                   sm.add_constant(ret["SPY"].values)).fit().params[1]
    out.append({"sector": sector, "rev_beta": fit.params[1], "t": fit.tvalues[1],
                "r2": fit.rsquared, "rev_beta_ex_covid": ex, "stock_beta": stock,
                "worst": g["sector_growth"].min(), "n_min": int(g["n"].min())})
res = pd.DataFrame(out).sort_values("rev_beta", ascending=False).reset_index(drop=True)

# --- report -----------------------------------------------------------------
allq = cycle[cycle["sector"] == "ALL"]
print(f"Firms per quarter: {allq['n'].min()}-{allq['n'].max()}  "
      f"quarters: {len(QUARTERS)}  trading days: {len(ret)}")
print(f"Corporate revenue cycle: trough {allq['sector_growth'].min():+.1%} "
      f"({allq.loc[allq['sector_growth'].idxmin(), 'cq']}), "
      f"peak {allq['sector_growth'].max():+.1%} "
      f"({allq.loc[allq['sector_growth'].idxmax(), 'cq']}), "
      f"latest {allq['sector_growth'].iloc[-1]:+.1%}")
print()
print(f"{'Sector':<24}{'Rev beta':>9}{'t':>7}{'R2':>7}{'ex-COVID':>10}"
      f"{'Stock beta':>12}{'Worst qtr':>11}{'Firms':>7}")
for r in res.itertuples():
    print(f"{r.sector:<24}{r.rev_beta:>9.2f}{r.t:>7.1f}{r.r2:>7.2f}"
          f"{r.rev_beta_ex_covid:>10.2f}{r.stock_beta:>12.2f}"
          f"{r.worst:>10.1%}{r.n_min:>7}")
print()
print(f"Revenue beta range: {res['rev_beta'].min():.2f} to {res['rev_beta'].max():.2f} "
      f"({res['rev_beta'].max() / res['rev_beta'].min():.0f}x)")
print(f"Stock beta range:   {res['stock_beta'].min():.2f} to {res['stock_beta'].max():.2f} "
      f"({res['stock_beta'].max() / res['stock_beta'].min():.1f}x)")
print("Rank correlation between the two: "
      f"{res['rev_beta'].corr(res['stock_beta'], method='spearman'):.2f}")

# --- chart ------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
y = np.arange(len(res))[::-1]
xmax = res["rev_beta"].max() * 1.28
for ax, col, colour, label in [(axes[0], "rev_beta", "#3b82f6", "Revenue beta"),
                               (axes[1], "stock_beta", "#f59e0b", "Stock beta")]:
    ax.barh(y, res[col], color=colour, height=0.62)
    for yi, v in zip(y, res[col]):
        ax.text(v + xmax * 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color="#e0e0e0")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(label)
    ax.tick_params(left=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(res["sector"])
fig.suptitle("Sector revenue betas vs stock betas, 2018-2026", y=0.97)
plt.tight_layout()
plt.savefig("sector-revenue-beta-vs-stock-beta-python.png", dpi=150,
            facecolor="#0a0a0a")
