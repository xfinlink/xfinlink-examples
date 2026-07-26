# Full write-up: https://xfinlink.com/blog/gross-profitability-forward-returns-python
"""Does gross profitability predict forward returns inside the S&P 500?

Novy-Marx (2013) argues that gross profit scaled by assets is the cleanest
accounting measure of economic profitability, and that it predicts the cross
section of returns about as well as book-to-market does. This tests the claim
on a point-in-time S&P 500 universe across eleven annual formations, with net
income scaled by assets as the comparison.

Two rules keep the test honest. The characteristic must have been knowable at
the formation date: the fiscal year ended at least 180 days earlier and the
filing carrying the figure was already submitted. And gross profit has to
survive cross-validation, since revenue minus cost of revenue and the reported
gross profit line disagree on roughly 8% of filings.
"""

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORMATIONS = [pd.Timestamp(f"{y}-06-30") for y in range(2015, 2026)]
EXCLUDE = {"Financials", "Real Estate"}   # gross profit is not defined for these
MIN_LAG = 180       # days from fiscal year end to formation
MAX_LAG = 1095      # discard a characteristic older than three years
TOLERANCE = 0.05    # allowed disagreement between the two gross profit builds
FIELDS = ["period_end", "filing_date", "fiscal_year", "gics_sector", "revenue",
          "cost_of_revenue", "gross_profit", "total_assets", "net_income"]
IMG = "gross-profitability-forward-returns-python.png"


# -- Retrieval --------------------------------------------------------------
# xfl.prices() splits >100 tickers into parallel batches and returns partial
# results with a warning if any batch fails. Chunk smaller, promote the warning
# to an error, and check that every requested ticker actually came back.
def fetch(fn, tickers, chunk, **kw):
    frames, missing = [], list(tickers)
    for _ in range(4):
        failed = []
        for i in range(0, len(missing), chunk):
            batch = missing[i:i + chunk]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    d = fn(batch, **kw)
                frames.append(d)
                returned = set(d["ticker"])
                failed += [t for t in batch if t not in returned]
            except Exception:
                failed += batch
        missing = failed
        if not missing:
            break
        chunk = max(5, chunk // 2)
        time.sleep(3)
    return pd.concat(frames, ignore_index=True), missing


universe = {f.year: xfl.index("sp500", as_of=f.strftime("%Y-%m-%d"))
            for f in FORMATIONS}
tickers = sorted({t for u in universe.values() for t in u["ticker"]})

px, px_missing = fetch(xfl.prices, tickers, 25, start="2015-06-01",
                       end="2026-07-01", interval="1mo", max_rows=300000)
fun, fun_missing = fetch(xfl.fundamentals, tickers, 50, period_type="annual",
                         start="2011-01-01", fields=FIELDS, max_rows=60000)
fun = fun.dropna(subset=["period_end", "filing_date"])


# -- Panel construction -----------------------------------------------------
def latest_known(pool, formation, respect_filing_date=True):
    """The most recent annual report available to an investor at `formation`."""
    ok = pool["period_end"] <= formation - pd.Timedelta(days=MIN_LAG)
    if respect_filing_date:
        ok &= pool["filing_date"] <= formation
    return pool[ok].sort_values("period_end").groupby("ticker").tail(1)


def build(respect_filing_date=True):
    rows, coverage = [], []
    for formation in FORMATIONS:
        names = set(universe[formation.year]["ticker"])
        pool = fun[fun["ticker"].isin(names)]
        chars = latest_known(pool, formation, respect_filing_date)

        # twelve complete monthly bars, and a series that already existed at
        # formation so the first bar is not a part-month stub for a new listing
        window = px[(px["date"] > formation) &
                    (px["date"] <= formation + pd.DateOffset(years=1))]
        window = window[window["ticker"].isin(px[px["date"] <= formation]["ticker"])]
        grouped = window.groupby("ticker")["return_daily"]
        fwd = grouped.apply(lambda s: np.prod(1 + s.values) - 1)
        fwd = fwd[grouped.size() == 12]

        rows.append(chars.assign(formation=formation,
                                 fwd_ret=chars["ticker"].map(fwd)))
        coverage.append({"formation": formation.year, "in_index": len(names),
                         "with_filings": pool["ticker"].nunique(),
                         "with_characteristic": len(rows[-1])})

    panel = pd.concat(rows, ignore_index=True)
    panel["lag_days"] = (panel["formation"] - panel["period_end"]).dt.days
    panel = panel[~panel["gics_sector"].isin(EXCLUDE)]
    panel = panel[panel["lag_days"] <= MAX_LAG]
    panel = panel.dropna(subset=["revenue", "cost_of_revenue", "gross_profit",
                                 "net_income", "total_assets", "fwd_ret"])
    panel = panel[(panel["total_assets"] > 0) & (panel["revenue"] > 0)]

    # Two independent builds of the same quantity. Where they disagree one of
    # the underlying line items is mis-tagged, and neither build can be trusted.
    derived = panel["revenue"] - panel["cost_of_revenue"]
    panel["gap"] = (derived - panel["gross_profit"]).abs() / panel["revenue"]
    panel["gpa"] = derived / panel["total_assets"]
    panel["gpa_field"] = panel["gross_profit"] / panel["total_assets"]
    panel["roa"] = panel["net_income"] / panel["total_assets"]
    return panel, pd.DataFrame(coverage)


full, coverage = build(respect_filing_date=True)
panel = full[full["gap"] <= TOLERANCE].copy()
relaxed, _ = build(respect_filing_date=False)
relaxed = relaxed[relaxed["gap"] <= TOLERANCE].copy()


def quintile(s):
    return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def add_sorts(df):
    df = df.copy()
    for col in ["gpa", "gpa_field", "roa"]:
        df[f"q_{col}"] = df.groupby("formation")[col].transform(quintile)
    # sector-neutral: rank within sector first, then sort on the rank
    df["sn_gpa"] = df.groupby(["formation", "gics_sector"])["gpa"].transform(
        lambda s: s.rank(pct=True) - 0.5)
    df["q_sn_gpa"] = df.groupby("formation")["sn_gpa"].transform(quintile)
    return df


full, panel, relaxed = add_sorts(full), add_sorts(panel), add_sorts(relaxed)


def spread(df, qcol, stat="mean"):
    """Quintile returns per cohort, then a t-test on the eleven annual spreads."""
    per = df.groupby(["formation", qcol])["fwd_ret"].agg(stat).unstack()
    diff = per[5] - per[1]
    t, p = stats.ttest_1samp(diff.values, 0)
    return per, diff, t, p


def ic(df, col):
    series = df.groupby("formation").apply(
        lambda d: stats.spearmanr(d[col], d["fwd_ret"]).statistic,
        include_groups=False)
    t, p = stats.ttest_1samp(series.values, 0)
    return series, t, p


# -- Output -----------------------------------------------------------------
pd.set_option("display.width", 200)
print("=" * 80)
print("GROSS PROFITABILITY AND FORWARD RETURNS | S&P 500 point-in-time universe")
print("=" * 80)
print(f"Formations         : {len(FORMATIONS)} annual, 30 June {FORMATIONS[0].year}"
      f" to 30 June {FORMATIONS[-1].year}")
print(f"Name-years         : {len(panel):,}   companies: {panel['ticker'].nunique()}")
print(f"Characteristic lag : median {int(panel['lag_days'].median())} days from"
      f" fiscal year end to formation")
print(f"Dropped, the two gross profit builds disagree by >{TOLERANCE:.0%} of revenue:"
      f" {len(full) - len(panel)} of {len(full)}")
print(f"Tickers returning no filings: {len(fun_missing)} of {len(tickers)}"
      f" (delisted names)   no prices: {len(px_missing)}")
print("\nUniverse coverage by cohort")
print(coverage.to_string(index=False))

header = (f"{'sort':<28}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5':>7}"
          f"{'Q5-Q1':>9}{'t':>7}{'p':>7}")
variants = [("Gross profit / assets", panel, "q_gpa"),
            ("  sector-neutral", panel, "q_sn_gpa"),
            ("Net income / assets", panel, "q_roa"),
            ("GP/A, no cross-validation", full, "q_gpa_field"),
            ("GP/A, look-ahead allowed", relaxed, "q_gpa")]

print("\n" + "-" * 80)
print("AVERAGE forward 12-month total return by quintile (%)")
print("-" * 80)
print(header)
for label, df, qcol in variants:
    per, diff, t, p = spread(df, qcol)
    cells = "".join(f"{v * 100:7.1f}" for v in per.mean())
    print(f"{label:<28}{cells}{diff.mean() * 100:9.1f}{t:7.2f}{p:7.3f}")

print("\nMEDIAN forward 12-month total return by quintile (%)")
print(header)
for label, df, qcol in variants[:3]:
    per, diff, t, p = spread(df, qcol, stat="median")
    cells = "".join(f"{v * 100:7.1f}" for v in per.mean())
    print(f"{label:<28}{cells}{diff.mean() * 100:9.1f}{t:7.2f}{p:7.3f}")

print("\n" + "-" * 80)
print("Rank information coefficient (Spearman, characteristic vs forward return)")
print("-" * 80)
for label, col in [("Gross profit / assets", "gpa"), ("  sector-neutral", "sn_gpa"),
                   ("Net income / assets", "roa")]:
    series, t, p = ic(panel, col)
    print(f"{label:<28} mean IC {series.mean():+.3f}   t {t:+5.2f}   p {p:.3f}   "
          f"positive in {int((series > 0).sum())}/{len(series)} cohorts")

print("\n" + "-" * 80)
print("Gross profitability Q5-Q1 spread by cohort (%)")
print("-" * 80)
per_gpa, diff_gpa, _, _ = spread(panel, "q_gpa")
cohort = pd.DataFrame({
    "n": panel.groupby("formation").size(),
    "universe_mean": panel.groupby("formation")["fwd_ret"].mean() * 100,
    "Q1": per_gpa[1] * 100, "Q5": per_gpa[5] * 100, "Q5-Q1": diff_gpa * 100})
cohort.index = cohort.index.year
print(cohort.round(1).to_string())

worst = diff_gpa.idxmin()
q1 = panel[(panel["formation"] == worst) & (panel["q_gpa"] == 1)]
print(f"\nWidest negative cohort {worst.year}: Q1 mean {q1['fwd_ret'].mean() * 100:.1f}%"
      f" vs median {q1['fwd_ret'].median() * 100:.1f}% across {len(q1)} names")
top = q1.nlargest(5, "fwd_ret")[["ticker", "gics_sector", "fiscal_year", "gpa", "fwd_ret"]]
top["gpa"] = top["gpa"].round(3)
top["fwd_ret"] = (top["fwd_ret"] * 100).round(0)
print(top.to_string(index=False))

print(f"\nSpearman correlation between the two characteristics: "
      f"{stats.spearmanr(panel['gpa'], panel['roa']).statistic:.3f}")


# -- Chart ------------------------------------------------------------------
BG, FG, ACCENT, MUTED = "#0a0a0a", "#e0e0e0", "#3b82f6", "#9ca3af"
gpa_per, _, _, _ = spread(panel, "q_gpa")
roa_per, _, _, _ = spread(panel, "q_roa")
gpa_mean, roa_mean = gpa_per.mean() * 100, roa_per.mean() * 100
gpa_se = gpa_per.std(ddof=1) / np.sqrt(len(gpa_per)) * 100
roa_se = roa_per.std(ddof=1) / np.sqrt(len(roa_per)) * 100

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
x, w = np.arange(5), 0.38
ax.bar(x - w / 2, gpa_mean.values, w, yerr=gpa_se.values, capsize=3,
       color=ACCENT, ecolor=MUTED, label="Gross profit / assets")
ax.bar(x + w / 2, roa_mean.values, w, yerr=roa_se.values, capsize=3,
       color=MUTED, ecolor="#6b7280", label="Net income / assets")
ax.set_xticks(x)
ax.set_xticklabels(["Q1\nleast profitable", "Q2", "Q3", "Q4", "Q5\nmost profitable"],
                   color=FG, fontsize=9)
ax.set_ylabel("Average forward 12-month total return (%)", color=FG, fontsize=10)
ax.set_title("Profitability quintiles and the returns that followed, S&P 500, 2015-2025",
             color=FG, fontsize=12, pad=12)
ax.tick_params(colors=FG)
for s in ax.spines.values():
    s.set_color("#2a2a2a")
ax.grid(axis="y", color="#1f1f1f", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(facecolor=BG, edgecolor="#2a2a2a", labelcolor=FG, fontsize=9)
plt.tight_layout()
plt.savefig(IMG, dpi=150, facecolor=BG)
print(f"\nChart written to {IMG} (bars are cohort averages, whiskers one standard error)")
