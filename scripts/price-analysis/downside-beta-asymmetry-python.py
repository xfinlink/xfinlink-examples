# Full write-up: https://xfinlink.com/blog/downside-beta-asymmetry-python
"""Do stocks fall harder than they rise? Downside vs upside beta, and whether
the gap between them survives out of sample.

Conditional betas follow Ang, Chen and Xing (2006): the market is split at its
own sample mean, and beta is re-estimated inside each half.
"""
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "GOOG", "META", "DIS", "VZ",
           "AMZN", "HD", "MCD", "NKE", "PG", "KO", "PEP", "COST", "WMT", "JNJ",
           "UNH", "LLY", "ABBV", "JPM", "GS", "BAC", "XOM", "CVX", "CAT", "HON",
           "UNP", "NEE", "DUK", "LIN", "AMT", "PLD"]
START, END = "2021-07-26", "2026-07-24"

# ── Data ──────────────────────────────────────────────────────────────
parts = []
for i in range(0, len(TICKERS) + 1, 6):
    batch = (TICKERS + ["SPY"])[i:i + 6]
    for attempt in range(4):
        try:
            parts.append(xfl.prices(batch, start=START, end=END,
                                    fields=["close", "return_daily"],
                                    max_rows=200000))
            break
        except xfl.XfinlinkError:
            time.sleep(3)
px = pd.concat(parts, ignore_index=True)

# One entity can trade under more than one ticker over time (Facebook became
# Meta in June 2022). Key on entity_id so each series stays continuous.
px["label"] = px.groupby("entity_id")["ticker"].transform("last")
rets = px.pivot_table(index="date", columns="label", values="return_daily")
stocks = [c for c in rets.columns if c != "SPY"]


def conditional_betas(panel):
    """Return full, downside and upside beta for every stock in `panel`."""
    mkt = panel["SPY"].dropna()
    mu = mkt.mean()
    out = {}
    for s in stocks:
        d = pd.concat([panel[s], mkt], axis=1, keys=["s", "m"]).dropna()
        down = (d["m"] < mu).astype(float)
        up = 1.0 - down
        # Separate intercept and slope per regime: identical coefficients to
        # two subsample regressions, but one covariance matrix for the contrast.
        X = np.column_stack([down, up, d["m"] * down, d["m"] * up])
        fit = sm.OLS(d["s"].values, X).fit(cov_type="HC1")
        contrast = fit.t_test(np.array([[0, 0, 1, -1]]))
        full = sm.OLS(d["s"].values, sm.add_constant(d["m"].values)).fit().params[1]
        out[s] = dict(beta=full, beta_dn=fit.params[2], beta_up=fit.params[3],
                      gap=fit.params[2] - fit.params[3],
                      tstat=float(np.ravel(contrast.tvalue)[0]),
                      pval=float(np.ravel(contrast.pvalue)[0]),
                      n=len(d))
    return pd.DataFrame(out).T.astype(float)


full = conditional_betas(rets).sort_values("gap", ascending=False)

mkt = rets["SPY"].dropna()
n_pos = int((full["gap"] > 0).sum())
n_sig = int((full["pval"] < 0.05).sum())

# ── Does the asymmetry persist? Split the sample in half ──────────────
mid = rets.index[len(rets) // 2]
h1 = conditional_betas(rets[rets.index < mid])
h2 = conditional_betas(rets[rets.index >= mid])
persistence = {c: spearmanr(h1[c], h2[c]) for c in ["beta", "beta_dn", "beta_up", "gap"]}

# ── Output ────────────────────────────────────────────────────────────
print(f"Universe: {len(stocks)} large caps + SPY   {START} to {END}")
print(f"Trading days: {len(mkt)}   market mean daily return: "
      f"{mkt.mean() * 1e4:.2f} bp   down days: {int((mkt < mkt.mean()).sum())}   "
      f"up days: {int((mkt >= mkt.mean()).sum())}\n")

print("FULL SAMPLE - conditional betas vs SPY (sorted by downside minus upside)")
print(f"{'':6s} {'beta':>7s} {'beta-':>7s} {'beta+':>7s} {'gap':>7s} {'t':>7s} {'p':>7s}")
for tkr, row in full.iterrows():
    print(f"{tkr:6s} {row['beta']:7.3f} {row['beta_dn']:7.3f} {row['beta_up']:7.3f} "
          f"{row['gap']:+7.3f} {row['tstat']:+7.2f} {row['pval']:7.3f}")

print(f"\nMean gap across {len(full)} names: {full['gap'].mean():+.4f} "
      f"(cross-sectional SD {full['gap'].std():.3f})")
print(f"Names with gap > 0: {n_pos} of {len(full)}")
print(f"Gaps significant at p<0.05: {n_sig} "
      f"(expected by chance at 5%: {0.05 * len(full):.1f})")

print(f"\nOUT OF SAMPLE - rank correlation between first half (to {mid.date()}) "
      "and second half")
for name, label in [("beta", "market beta"), ("beta_dn", "downside beta"),
                    ("beta_up", "upside beta"), ("gap", "downside minus upside")]:
    r = persistence[name]
    print(f"  {label:22s} Spearman {r.statistic:+.3f}   p = {r.pvalue:.4f}")
print(f"\nCross-sectional SD of the gap: first half {h1['gap'].std():.3f}, "
      f"second half {h2['gap'].std():.3f}")
same_sign = int((((h1["gap"] > 0) == (h2["gap"] > 0))).sum())
print(f"Names keeping the sign of their gap in both halves: {same_sign} of {len(h1)} "
      f"(coin flip: {len(h1) / 2:.1f})")
top, bot = h1["gap"].nlargest(7).index, h1["gap"].nsmallest(7).index
print(f"Top 7 by first-half gap:    first half {h1['gap'][top].mean():+.3f} "
      f"-> second half {h2['gap'][top].mean():+.3f}")
print(f"Bottom 7 by first-half gap: first half {h1['gap'][bot].mean():+.3f} "
      f"-> second half {h2['gap'][bot].mean():+.3f}")

# ── Chart ─────────────────────────────────────────────────────────────
BG, FG, ACCENT = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
panels = [("beta", "Market beta", axes[0]),
          ("gap", "Downside beta minus upside beta", axes[1])]
for col, title, ax in panels:
    ax.set_facecolor(BG)
    ax.scatter(h1[col], h2[col], s=42, color=ACCENT, alpha=0.85,
               edgecolors="none", zorder=3)
    lo = min(h1[col].min(), h2[col].min())
    hi = max(h1[col].max(), h2[col].max())
    pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555",
            linewidth=1, linestyle="--", zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    rho = persistence[col].statistic
    ax.set_title(f"{title}\nrank correlation {rho:+.2f}", color=FG, fontsize=11)
    ax.set_xlabel("Estimated on 2021-2023", color=FG, fontsize=9)
    ax.set_ylabel("Estimated on 2024-2026", color=FG, fontsize=9)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
fig.suptitle("Market beta persists. The downside-upside gap does not.",
             color=FG, fontsize=13)
plt.tight_layout()
plt.savefig("downside-beta-asymmetry-python.png", dpi=150, facecolor=BG)
