# Full write-up: https://xfinlink.com/blog/revenue-breadth-diffusion-index-market-python
"""
Does revenue breadth predict the stock market?

Builds a quarterly diffusion index from point-in-time S&P 500 membership: the
share of companies whose fiscal quarter grew revenue against the same quarter a
year earlier. Measures the index against SPY quarterly total returns at leads
and lags of up to three quarters, in the full 2004-2026 sample and with the two
crisis windows removed.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

SLUG = "revenue-breadth-diffusion-index-market-python"
START, END = "2004Q1", "2026Q1"
CRISIS = [("2008Q3", "2009Q4"), ("2020Q1", "2020Q4")]


def fetch(fn, **kw):
    for attempt in range(3):
        try:
            return fn(**kw)
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(8)
    return None


# 1. Survivorship-bias-free universe: the index as it actually stood at each
#    year end, carried by entity id so a recycled ticker cannot swap companies.
ids = set()
for year in range(2004, 2026):
    ids |= set(fetch(xfl.index, index_name="sp500",
                     as_of=f"{year}-12-31")["entity_id"].dropna().astype(int))
ids |= set(fetch(xfl.index, index_name="sp500")["entity_id"].dropna().astype(int))
ids = sorted(ids)

# 2. Quarterly revenue for every one of them, back far enough to give the
#    earliest sample quarter a year-earlier comparison.
f = pd.concat(
    [fetch(xfl.fundamentals, entity_id=ids[i:i + 60], period_type="quarterly",
           start="2003-01-01", end="2026-09-10", fields=["revenue"], max_rows=200000)
     for i in range(0, len(ids), 60)],
    ignore_index=True)

# 3. Year-over-year growth: this fiscal quarter against the fourth one back.
#    The 300-430 day gap check keeps a missing quarter from creating a false
#    comparison across an eight-quarter jump.
f = f.dropna(subset=["revenue", "period_end"]).drop_duplicates(["entity_id", "period_end"])
f = f.sort_values(["entity_id", "period_end"])
grp = f.groupby("entity_id")
f["prev"] = grp["revenue"].shift(4)
gap = (f["period_end"] - grp["period_end"].shift(4)).dt.days
d = f[(f["prev"] > 0) & gap.between(300, 430)].copy()
d["grew"] = d["revenue"] > d["prev"]
d["q"] = d["period_end"].dt.to_period("Q")

# 4. The diffusion index: share of companies growing, by calendar quarter of
#    period end. A quarter's reports reach the public record during the quarter
#    that follows it, so q+1 is the first quarter anyone could act on.
D = d.groupby("q").agg(n=("grew", "size"), diffusion=("grew", "mean")).loc[START:END]

# 5. Market: SPY total return compounded within each calendar quarter.
spy = fetch(xfl.prices, ticker="SPY", start="2003-06-01", end="2026-09-10",
            fields=["return_daily"])
spy["q"] = spy["date"].dt.to_period("Q")
mkt = spy.groupby("q")["return_daily"].apply(lambda x: (1 + x).prod() - 1)

t = D.join(mkt.rename("mkt"), how="inner")
t["dD"] = t["diffusion"].diff()

calm = t.copy()
for lo, hi in CRISIS:
    calm = calm[(calm.index < pd.Period(lo)) | (calm.index > pd.Period(hi))]

LAGS = range(-3, 4)


def leadlag(frame, col):
    """corr(series at q, market return at q+k), with its t-statistic."""
    out = []
    for k in LAGS:
        a, b = frame[col], frame["mkt"].shift(-k)
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append((k, r, r * np.sqrt((m.sum() - 2) / (1 - r ** 2)), int(m.sum())))
    return out


full_change, full_level = leadlag(t, "dD"), leadlag(t, "diffusion")
calm_change, calm_level = leadlag(calm, "dD"), leadlag(calm, "diffusion")

fwd = t.assign(nxt=t["mkt"].shift(-1)).dropna(subset=["dD", "nxt"])
reg = stats.linregress(fwd["dD"], fwd["nxt"])

# ---------------------------------------------------------------- output ----
lo_q, hi_q = t["diffusion"].idxmin(), t["diffusion"].idxmax()
print("Revenue breadth against the market, point-in-time S&P 500")
print(f"Panel:   {len(ids)} entities from the 2004-2025 year-end rosters plus the current")
print(f"         roster; {d['entity_id'].nunique()} carry a usable quarterly revenue series,")
print(f"         {len(d):,} company-quarters")
print("Index:   share of companies whose fiscal quarter grew revenue against the")
print("         same quarter a year earlier, grouped by calendar quarter of period end")
print(f"Sample:  {len(t)} quarters, {t.index.min()} to {t.index.max()}, "
      f"{t['n'].min()} to {t['n'].max()} companies per quarter")
print("Market:  SPY total return compounded within each calendar quarter")
print()
print(f"Index    mean {t['diffusion'].mean():.2%}   sd {t['diffusion'].std():.2%}   "
      f"low {t['diffusion'].min():.2%} ({lo_q})   high {t['diffusion'].max():.2%} ({hi_q})")
print()
print("Correlation with the market return k quarters away")
print(f"  {'k':<18}" + "".join(f"{k:+8d}" for k in LAGS))
for label, rows in [("change in index", full_change), ("index level", full_level)]:
    print(f"  {label:<18}" + "".join(f"{r:+8.3f}" for _, r, _, _ in rows))
    print(f"  {'t':<18}" + "".join(f"{s:+8.2f}" for _, _, s, _ in rows))
print()
print(f"Same, excluding 2008Q3-2009Q4 and 2020Q1-2020Q4 ({len(calm)} quarters)")
for label, rows in [("change in index", calm_change), ("index level", calm_level)]:
    print(f"  {label:<18}" + "".join(f"{r:+8.3f}" for _, r, _, _ in rows))
    print(f"  {'t':<18}" + "".join(f"{s:+8.2f}" for _, _, s, _ in rows))
print()
print("The two turning points, quarter by quarter")
print(f"  {'quarter':<9}{'index':>9}{'change':>10}{'market':>10}")
for q in list(pd.period_range("2008Q3", "2009Q3", freq="Q")) + \
         list(pd.period_range("2020Q1", "2020Q3", freq="Q")):
    r = t.loc[q]
    print(f"  {str(q):<9}{r['diffusion']:>8.2%}{r['dD'] * 100:>+9.2f}pt{r['mkt']:>+10.2%}")
print()
print("Next quarter's market return regressed on this quarter's change in the index")
print(f"  beta {reg.slope:+.3f}   t {reg.slope / reg.stderr:+.2f}   "
      f"R2 {reg.rvalue ** 2:.4f}   n {len(fwd)}")

# ----------------------------------------------------------------- chart ----
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = t.index.to_timestamp()
ax1.bar(x, t["mkt"] * 100, width=70, color="#475569", label="SPY quarterly total return (%)")
ax1.axhline(0, color="#333333", lw=0.8)
ax1b = ax1.twinx()
ax1b.plot(x, t["diffusion"] * 100, color="#3b82f6", lw=1.8,
          label="Companies growing revenue year over year (%)")
ax1b.set_ylabel("Share growing (%)", color="#3b82f6")
ax1b.tick_params(axis="y", colors="#3b82f6")
ax1b.set_ylim(18, 93)
ax1.set_ylabel("Quarterly return (%)")
ax1.set_ylim(-26, 24)
ax1.set_title("Revenue breadth in the S&P 500 and the market, 2004-2026")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower right", facecolor="#0a0a0a",
           edgecolor="#333333", fontsize=8)

w = 0.38
ks = np.array(list(LAGS))
ax2.bar(ks - w / 2, [r for _, r, _, _ in full_change], w, color="#3b82f6",
        label="Full sample, 89 quarters")
ax2.bar(ks + w / 2, [r for _, r, _, _ in calm_change], w, color="#475569",
        label="Excluding the 2008-09 and 2020 windows")
ax2.axhline(0, color="#666666", lw=0.8)
ax2.set_xticks(ks)
ax2.set_xlabel("Market return measured k quarters after the index moves")
ax2.set_ylabel("Correlation")
ax2.set_title("Change in revenue breadth against market returns at each lead and lag")
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", fontsize=8)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
