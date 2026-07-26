# Full write-up: https://xfinlink.com/blog/insider-buying-clusters-forward-returns-python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "insider-buying-clusters-forward-returns-python"
IMG = f"{SLUG}.png"
START_YEAR, END_YEAR = 2015, 2025
MIN_TRADE = 10_000          # ignore token purchases
HORIZON = 3                 # months of forward return
EXCLUDE_ROLES = ("10% Owner", "Other")
OFFICER_KEYS = ("CEO", "CFO", "COO", "CTO", "President", "Officer", "General Counsel")

# ---------------------------------------------------------------- universe
# Membership is keyed on entity_id, not ticker: symbols get reassigned and
# recycled, entity ids do not. One extra roster year because a December
# transaction can be filed the following January.
rosters = {y: xfl.index("sp500", as_of=f"{y}-01-01", limit=600)
           for y in range(START_YEAR, END_YEAR + 2)}
snaps = {y: set(r["entity_id"]) for y, r in rosters.items()}
universe = sorted(set().union(*[set(rosters[y]["ticker"].dropna())
                                for y in range(START_YEAR, END_YEAR + 1)]))
print(f"Point-in-time S&P 500 rosters {START_YEAR}-{END_YEAR}: {len(universe)} distinct tickers")

# ---------------------------------------------------------------- insider buys
# 50 tickers per call. Above 100 the client splits the request internally and
# returns partial data with a warning if a sub-request fails; staying under the
# split threshold turns any failure into a hard error instead of a silent gap.
def chunked(fn, tickers, size=50, **kw):
    return pd.concat([fn(tickers[i:i + size], **kw)
                      for i in range(0, len(tickers), size)], ignore_index=True)


buys = chunked(xfl.insiders, universe, transaction_type="open_market_buy",
               start=f"{START_YEAR}-01-01", end=f"{END_YEAR}-12-31", max_rows=300_000)
print(f"Open-market purchases pulled: {len(buys):,} rows, {buys['ticker'].nunique()} tickers")

buys = buys[~buys["insider_role"].isin(EXCLUDE_ROLES)]
buys = buys[buys["transaction_value"] >= MIN_TRADE]
buys["month"] = buys["filing_date"].dt.tz_localize(None).dt.to_period("M")
# word boundaries matter: a plain substring test scores "Director" as an officer
# because "direCTOr" contains "CTO"
buys["is_officer"] = buys["insider_role"].str.contains(
    r"\b(?:" + "|".join(OFFICER_KEYS) + r")\b", case=False, regex=True)
print(f"After dropping 10% owners and trades under ${MIN_TRADE:,}: {len(buys):,} rows")

# ---------------------------------------------------------------- firm-month signal
sig = buys.groupby(["entity_id", "month"]).agg(
    ticker=("ticker", "last"),
    n_insiders=("insider_name", "nunique"),
    dollars=("transaction_value", "sum"),
    officer_share=("is_officer", "mean"),
).reset_index()

# keep only firm-months where the company sat in the index at that year's opening roster
sig = sig[[e in snaps[m.year] for e, m in zip(sig["entity_id"], sig["month"])]]
print(f"Firm-months with qualifying insider buying: {len(sig):,}")

# ---------------------------------------------------------------- prices
tick = sorted(sig["ticker"].unique()) + ["SPY"]
px = chunked(xfl.prices, tick, start=f"{START_YEAR}-01-01", end="2026-07-01",
             interval="1mo", fields=["close", "return_daily", "shares_outstanding"],
             max_rows=300_000)
px["month"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "month"]).sort_values(["entity_id", "month"])
covered = sig["entity_id"].isin(px["entity_id"]).mean()
print(f"Monthly price coverage of signal companies: {covered * 100:.1f}%")

# forward HORIZON-month total return, starting the month AFTER the signal month
def fwd(s):
    r = np.log1p(s)
    return np.expm1(r.shift(-1).rolling(HORIZON).sum().shift(-(HORIZON - 1)))

px["fwd"] = px.groupby("entity_id")["return_daily"].transform(fwd)
spy = px[px["ticker"] == "SPY"].set_index("month")["fwd"]

sig = sig.merge(px[["entity_id", "month", "fwd", "close", "shares_outstanding"]],
                on=["entity_id", "month"], how="left")
sig["mkt_cap"] = sig["close"] * sig["shares_outstanding"]
sig["abn"] = sig["fwd"] - sig["month"].map(spy)
sig["intensity_bp"] = 10_000 * sig["dollars"] / sig["mkt_cap"]
sig = sig.dropna(subset=["abn", "intensity_bp"])
print(f"Firm-months with a complete {HORIZON}-month forward window: {len(sig):,}")
print(f"Formation months: {sig['month'].min()} to {sig['month'].max()}")

# Winsorise the excess return at the 1st and 99th percentiles. Three-month
# single-stock excess returns have very long tails, so a handful of names would
# otherwise set the group averages on their own.
lo, hi = sig["abn"].quantile([0.01, 0.99])
raw_mean = sig["abn"].mean()
sig["abn"] = sig["abn"].clip(lo, hi)
print(f"Excess return winsorised at {lo*100:.1f}% / {hi*100:.1f}%; "
      f"sample mean moves from {raw_mean*100:.2f}% to {sig['abn'].mean()*100:.2f}%")

# ---------------------------------------------------------------- buckets
sig["bucket"] = np.where(sig["n_insiders"] >= 3, "3+ insiders",
                  np.where(sig["n_insiders"] == 2, "2 insiders", "1 insider"))
sig["size_q"] = "Q" + (sig.groupby("month")["intensity_bp"]
                       .rank(pct=True).mul(4).apply(np.ceil).clip(1, 4).astype(int).astype(str))


def nw_t(monthly_means):
    """Newey-West t-stat on the time series of cross-sectional monthly means."""
    y = monthly_means.dropna()
    if len(y) < 12:
        return np.nan
    fit = sm.OLS(y.values, np.ones(len(y))).fit(cov_type="HAC",
                                                cov_kwds={"maxlags": HORIZON})
    return fit.tvalues[0]


def table(frame, key):
    """One row per group. mean_abn is the average across formation months, which
    is the same series the Newey-West t-stat is computed on, so the two agree."""
    rows = []
    for name, g in frame.groupby(key, observed=True):
        monthly = g.groupby("month")["abn"].mean()
        rows.append({
            key: name,
            "n": len(g),
            "months": monthly.notna().sum(),
            "mean_abn": monthly.mean(),
            "med_abn": g["abn"].median(),
            "hit": (g["abn"] > 0).mean(),
            "t_nw": nw_t(monthly),
        })
    return pd.DataFrame(rows)


def show(df, key, label):
    print(f"\n{label}")
    print(f"{key:<22}{'n':>6}{'months':>8}{'mean abn':>11}"
          f"{'median':>9}{'hit rate':>10}{'NW t':>8}")
    for _, r in df.iterrows():
        print(f"{str(r[key]):<22}{r['n']:>6,}{r['months']:>8}{r['mean_abn']*100:>10.2f}%"
              f"{r['med_abn']*100:>8.2f}%{r['hit']*100:>9.1f}%{r['t_nw']:>8.2f}")


by_count = table(sig, "bucket").sort_values("bucket")
by_size = table(sig, "size_q").sort_values("size_q")

print(f"\n=== Forward {HORIZON}-month return vs SPY, S&P 500 insider open-market buys ===")
base = sig.groupby("month")["abn"].mean()
print(f"Baseline: all {len(sig):,} firm-months across {base.notna().sum()} formation "
      f"months  mean abnormal {base.mean()*100:.2f}%  NW t {nw_t(base):.2f}")
show(by_count, "bucket", "By number of distinct insiders buying in the filing month")
show(by_size, "size_q", "By purchase intensity (dollars bought / market cap), ranked within month")

# cluster split by who was buying
cl = sig[sig["bucket"] == "3+ insiders"].copy()
cl["who"] = np.where(cl["officer_share"] > 0, "At least one officer", "Directors only")
show(table(cl, "who").sort_values("who"), "who", "Clusters (3+) by who was buying")

# horse race: does the count survive once intensity is controlled for?
X = sm.add_constant(pd.DataFrame({
    "log_n": np.log(sig["n_insiders"]),
    "log_intensity": np.log(sig["intensity_bp"]),
}))
fit = sm.OLS(sig["abn"].values, X).fit(cov_type="cluster",
                                       cov_kwds={"groups": sig["month"].astype(str)})
print("\nPooled regression, abnormal return on both measures (SE clustered by month)")
for nm in ["log_n", "log_intensity"]:
    print(f"  {nm:<15} coef {fit.params[nm]:+.4f}   t {fit.tvalues[nm]:+.2f}   "
          f"p {fit.pvalues[nm]:.3f}")
print(f"  n = {int(fit.nobs):,}   R-squared = {fit.rsquared:.4f}")

# ---------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3f3f3f",
})
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
panels = [
    (axes[0], by_count, "bucket", "Insiders buying in the month",
     "How many insiders bought"),
    (axes[1], by_size, "size_q", "Purchase size quartile (Q4 = largest)",
     "How much they bought"),
]
lim = max(abs(by_count["mean_abn"]).max(), abs(by_size["mean_abn"]).max()) * 100 * 1.9
for ax, tb, key, xlab, title in panels:
    vals = tb["mean_abn"] * 100
    ax.bar(tb[key].astype(str), vals, color="#3b82f6", width=0.6)
    for x, (v, n) in enumerate(zip(vals, tb["n"])):
        ax.annotate(f"n={n:,}", (x, v), xytext=(0, 6 if v >= 0 else -14),
                    textcoords="offset points", ha="center",
                    fontsize=8, color="#9ca3af")
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-lim, lim)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(f"Average {HORIZON}-month return minus SPY (%)")
fig.suptitle("Insider buying clusters and forward 3-month excess return, S&P 500 2015-2025",
             fontsize=12, color="#e0e0e0")
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print(f"\nChart saved to {IMG}")
