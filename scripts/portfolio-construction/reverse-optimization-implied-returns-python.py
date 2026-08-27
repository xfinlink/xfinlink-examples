# Full write-up: https://xfinlink.com/blog/reverse-optimization-implied-returns-python
"""Reverse optimization: the expected returns implied by the S&P 500's own weights.

Forward mean-variance optimization needs an expected return for every asset and
nobody has one. Reverse optimization runs the argument backwards: take the
capitalisation-weighted index as the optimal portfolio and back out the excess
return vector that makes it optimal, mu = lambda * Sigma * w.

For each year end from 2016 to 2024 the point-in-time S&P 500 roster is priced,
its covariance matrix estimated from three years of weekly returns, and its
implied return vector computed. The ranking is then compared with what each
member actually delivered over the following twelve months: total return in
excess of cash, realised volatility and worst drawdown.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ERP = 0.05                  # annual equity risk premium the index is assumed to price
FORMATIONS = [f"{y}-12-31" for y in range(2016, 2025)]
PX_START, PX_END = "2014-01-01", "2025-12-31"
LOOKBACK = 3                # years of weekly returns behind each formation date
QLABELS = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
OUT_PNG = "reverse-optimization-implied-returns-python.png"

# ---------------------------------------------------------------- universe --
# Members are addressed by entity id, never by ticker. A symbol on a 2016
# roster can belong to a different company today, and a join on the string
# would price the wrong business.
rosters = {d: xfl.index("sp500", as_of=d) for d in FORMATIONS}
ids = sorted({int(i) for r in rosters.values() for i in r["entity_id"]})
names = (pd.concat(rosters.values())[["entity_id", "ticker", "entity_name"]]
         .drop_duplicates("entity_id"))

BLOCKS = [("2014-01-01", "2016-12-31"), ("2017-01-01", "2019-12-31"),
          ("2020-01-01", "2022-12-31"), ("2023-01-01", "2025-12-31")]
jobs = [(ids[i:i + 25], b) for b in BLOCKS for i in range(0, len(ids), 25)]


def weekly(job):
    chunk, (start, end) = job
    for attempt in range(3):
        try:
            return xfl.prices(entity_id=chunk, start=start, end=end, interval="1w",
                              fields=["date", "ticker", "return_daily", "close",
                                      "shares_outstanding"], max_rows=100000)
        except xfl.XfinlinkError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


with ThreadPoolExecutor(max_workers=5) as pool:
    px = pd.concat(pool.map(weekly, jobs), ignore_index=True)
px = px.drop_duplicates(subset=["entity_id", "date"]).sort_values(["entity_id", "date"])

cash_px = xfl.prices("BIL", start=PX_START, end=PX_END, interval="1w",
                     fields=["date", "return_daily"])
bench = xfl.prices("SPY", start=PX_START, end=PX_END, interval="1w",
                   fields=["date", "return_daily"])
cash = cash_px.set_index("date")["return_daily"]
bench = bench.set_index("date")["return_daily"]

# An equal-weighted average is sensitive to a single extreme value, so weekly
# returns are screened for internal consistency against the price path: a
# reported total return above 50% must come with a price move of at least 25%.
px["pchg"] = px.groupby("entity_id")["close"].pct_change()
odd = (px["return_daily"].abs() > 0.5) & (px["pchg"].abs() < 0.25)
n_obs, n_odd = int(px["return_daily"].notna().sum()), int(odd.sum())
px.loc[odd, "return_daily"] = np.nan

ret = px.pivot_table(index="date", columns="entity_id", values="return_daily")
mcap = (px.pivot_table(index="date", columns="entity_id", values="close")
        * px.pivot_table(index="date", columns="entity_id", values="shares_outstanding"))
# A newly listed member's first partial week sits on its own date. Keep the
# weeks the universe actually trades together.
grid = ret.index[ret.notna().sum(axis=1) >= 0.5 * ret.shape[1]]
ret, mcap = ret.reindex(grid), mcap.reindex(grid)

# ------------------------------------------------------- reverse optimizer --
rows, yearly = [], []
for f in FORMATIONS:
    fd = pd.Timestamp(f)
    members = [i for i in rosters[f]["entity_id"].astype(int) if i in ret.columns]
    look = ret.loc[(ret.index > fd - pd.DateOffset(years=LOOKBACK)) & (ret.index <= fd), members]
    fwd = ret.loc[(ret.index > fd) & (ret.index <= fd + pd.DateOffset(years=1)), members]

    keep = [c for c in look.columns[look.notna().all()] if fwd[c].notna().sum() >= len(fwd) - 2]
    cap = mcap.loc[mcap.index <= fd, keep].ffill().iloc[-1]
    keep = [c for c in keep if np.isfinite(cap.get(c, np.nan)) and cap[c] > 0]

    w = (cap[keep] / cap[keep].sum()).to_numpy()          # capitalisation weights
    R = look[keep].to_numpy()
    cov = np.cov(R, rowvar=False) * 52.0                  # annualised covariance
    var_p = float(w @ cov @ w)
    implied = (ERP / var_p) * (cov @ w)                   # mu = lambda * Sigma * w

    rp = R @ w                                            # the index's own return series
    beta = np.array([np.cov(R[:, j], rp)[0, 1] / np.var(rp, ddof=1)
                     for j in range(R.shape[1])])

    F = fwd[keep].fillna(0.0)
    cum = (1.0 + F).cumprod()
    rf = float((1 + cash[(cash.index > fd) & (cash.index <= fd + pd.DateOffset(years=1))]).prod() - 1)
    total = ((1.0 + F).prod() - 1.0).to_numpy()

    d = pd.DataFrame({
        "entity_id": keep, "year": fd.year + 1, "implied": implied, "beta": beta,
        "fwd_ret": total, "fwd_exc": (1 + total) / (1 + rf) - 1,
        "fwd_vol": (F.std(ddof=1) * np.sqrt(52)).to_numpy(),
        "fwd_dd": (cum / cum.cummax() - 1.0).min().to_numpy(),
    })
    d["quintile"] = pd.qcut(d["implied"], 5, labels=QLABELS)
    rows.append(d)

    yearly.append({
        "year": fd.year + 1, "roster": len(members), "n": len(keep),
        "sigma_p": np.sqrt(var_p), "lam": ERP / var_p,
        "lo": implied.min(), "hi": implied.max(),
        "maxdev": float(np.max(np.abs(implied - ERP * beta))),
        "corr_spy": float(pd.Series(rp, index=look.index).corr(bench.reindex(look.index))),
        "cash": rf, "mkt": float((1 + F.to_numpy() @ w).prod() - 1),
        "rho_ret": d[["implied", "fwd_ret"]].corr(method="spearman").iloc[0, 1],
        "rho_vol": d[["implied", "fwd_vol"]].corr(method="spearman").iloc[0, 1],
    })

panel = pd.concat(rows, ignore_index=True)
yr = pd.DataFrame(yearly)

byq = panel.groupby(["year", "quintile"], observed=True).agg(
    implied=("implied", "mean"), beta=("beta", "mean"), ret=("fwd_ret", "mean"),
    exc=("fwd_exc", "mean"), med=("fwd_exc", "median"), vol=("fwd_vol", "mean"),
    dd=("fwd_dd", "mean"), n=("beta", "size"))
agg = byq.groupby("quintile", observed=True).mean()
agg["exc_vol"] = agg["exc"] / agg["vol"]
growth = (1 + byq["ret"].unstack()).prod()

# ------------------------------------------------------------------ output --
SEP = "-" * 78
print(SEP)
print("REVERSE OPTIMIZATION OF THE S&P 500, FORMATION YEAR ENDS 2016-2024")
print("implied excess return mu = lambda * Sigma * w, lambda set so the index")
print(f"prices a {ERP:.0%} annual risk premium")
print(SEP)
print(f"universe: point-in-time S&P 500 members at each year end, addressed by")
print(f"          entity id; {LOOKBACK} years of weekly returns behind each formation")
print(f"          date and 12 months in front of it, {int(yr['n'].mean())} members per year on average")
print(f"weekly observations {n_obs:,}; set aside by the consistency screen: {n_odd}")
print()

print("PER FORMATION")
print(f"{'year':<6}{'names':>6}{'index vol':>11}{'lambda':>8}{'implied lo':>12}"
      f"{'implied hi':>12}{'max |mu-ERPb|':>15}{'corr w/ SPY':>13}")
for _, r in yr.iterrows():
    print(f"{int(r['year']):<6}{int(r['n']):>6}{r['sigma_p']*100:>10.1f}%{r['lam']:>8.2f}"
          f"{r['lo']*100:>11.2f}%{r['hi']*100:>11.2f}%{r['maxdev']:>15.1e}{r['corr_spy']:>13.3f}")
print()

print("IMPLIED-RETURN QUINTILES, AVERAGE OF THE NINE CROSS-SECTIONS")
print(f"{'quintile':<12}{'n':>5}{'implied':>9}{'beta':>7}{'excess ret':>12}"
      f"{'median':>9}{'volatility':>12}{'drawdown':>10}{'ret/vol':>9}")
for q, r in agg.iterrows():
    print(f"{str(q):<12}{r['n']:>5.0f}{r['implied']*100:>8.2f}%{r['beta']:>7.2f}"
          f"{r['exc']*100:>11.2f}%{r['med']*100:>8.2f}%{r['vol']*100:>11.1f}%"
          f"{r['dd']*100:>9.1f}%{r['exc_vol']:>9.2f}")
print()
print("growth of $1 held in each quintile, equal weighted, rebalanced every year:")
print("  " + "  ".join(f"{q.split()[0]} {growth[q]:.2f}x" for q in QLABELS))
print()

print("RANK CORRELATION OF THE IMPLIED RETURN WITH WHAT FOLLOWED")
print(f"{'year':<6}{'vs 12m return':>15}{'vs 12m volatility':>20}{'index return':>15}")
for _, r in yr.iterrows():
    print(f"{int(r['year']):<6}{r['rho_ret']:>+15.3f}{r['rho_vol']:>+20.3f}{r['mkt']*100:>14.1f}%")
print(f"{'mean':<6}{yr['rho_ret'].mean():>+15.3f}{yr['rho_vol'].mean():>+20.3f}"
      f"{yr['mkt'].mean()*100:>14.1f}%")
print()

last = panel[panel["year"] == 2025].merge(names, on="entity_id", how="left")
print("FORMED 31 DEC 2024: THE ENDS OF THE IMPLIED-RETURN RANKING")
print(f"{'ticker':<8}{'company':<32}{'beta':>6}{'implied':>9}{'2025 return':>13}")
for _, r in pd.concat([last.nsmallest(4, "implied"), last.nlargest(4, "implied")]).iterrows():
    print(f"{r['ticker']:<8}{str(r['entity_name']).title()[:31]:<32}{r['beta']:>6.2f}"
          f"{r['implied']*100:>8.2f}%{r['fwd_ret']*100:>12.1f}%")

# ------------------------------------------------------------------- chart --
BG, FG, ACCENT, MUTED, WARM = "#0a0a0a", "#e0e0e0", "#3b82f6", "#6b7280", "#f59e0b"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

x, bw = np.arange(len(agg)), 0.27
ax1.bar(x - bw, agg["implied"] * 100, bw, color=ACCENT, label="Implied by the index")
ax1.bar(x, agg["exc"] * 100, bw, color=MUTED, label="Realised, average")
ax1.bar(x + bw, agg["med"] * 100, bw, color=WARM, label="Realised, median")
ax1.set_xticks(x)
ax1.set_xticklabels([q.replace(" ", "\n") for q in QLABELS], fontsize=8)
ax1.set_xlabel("Implied-return quintile")
ax1.set_ylabel("Annual return above cash (%)")
ax1.set_title("What was implied, what arrived", fontsize=10, color=FG)
ax1.legend(frameon=False, fontsize=8, labelcolor=FG)

xy, bw2 = np.arange(len(yr)), 0.38
ax2.bar(xy - bw2 / 2, yr["rho_vol"], bw2, color=ACCENT, label="Next 12m volatility")
ax2.bar(xy + bw2 / 2, yr["rho_ret"], bw2, color=MUTED, label="Next 12m return")
ax2.axhline(0, color=FG, linewidth=0.8)
ax2.set_xticks(xy)
ax2.set_xticklabels([str(int(y)) for y in yr["year"]], fontsize=8, rotation=45)
ax2.set_xlabel("Twelve months to December")
ax2.set_ylabel("Rank correlation with the implied return")
ax2.set_title("A risk ranking, not a return ranking", fontsize=10, color=FG)
ax2.legend(frameon=False, fontsize=8, labelcolor=FG)

for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("Expected returns implied by the S&P 500's own weights, 2017-2025",
             color=FG, fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nchart saved to {OUT_PNG}")
