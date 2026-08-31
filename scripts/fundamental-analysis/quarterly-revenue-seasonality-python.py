# Full write-up: https://xfinlink.com/blog/quarterly-revenue-seasonality-python
"""How seasonal is a company's revenue, and what does that do to a quarter-over-quarter read?

Measures the share of each fiscal year's revenue that lands in each fiscal quarter for a
sector-diverse set of S&P 500 companies, then quantifies the error made by reading raw
quarter-over-quarter growth instead of year-over-year growth.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["HD", "LOW", "TGT", "BBY", "ROST", "NKE", "WMT", "COST", "KO", "PG",
           "CL", "KMB", "MSFT", "ADBE", "ORCL", "CRM", "INTU", "AAPL", "HON",
           "CAT", "UNP", "EMR", "JNJ", "ABT", "MRK", "UNH", "XOM", "CVX", "COP"]
START, END = "2015-07-01", "2025-12-31"

qtr = xfl.fundamentals(TICKERS, period_type="quarterly", start=START, end=END, fields=["revenue"])
ann = xfl.fundamentals(TICKERS, period_type="annual", start=START, end=END, fields=["revenue"])


def one_row_per_period(g):
    """Collapse records describing the same reporting period into a single row."""
    g = g.sort_values(["period_end", "source"])
    keep = []
    for _, r in g.iterrows():
        if not keep or (r["period_end"] - keep[-1]["period_end"]).days >= 45:
            keep.append(r)
        elif r["source"] == "filing" and keep[-1]["source"] != "filing":
            keep[-1] = r
    return pd.DataFrame(keep)


qtr = pd.concat([one_row_per_period(g) for _, g in qtr.groupby("ticker")], ignore_index=True)

# Fiscal quarter POSITION, derived from the data. A quarter whose period end matches an
# annual period end is that company's fiscal Q4; positions count backwards from there.
# Calendar month is never used: fiscal year ends range from January to September here.
frames = {}
for t, g in qtr.groupby("ticker"):
    g = g.sort_values("period_end").reset_index(drop=True)
    gaps = g["period_end"].diff().dt.days.dropna()
    ends = ann.loc[ann["ticker"] == t, "period_end"]
    q4 = [i for i, d in enumerate(g["period_end"]) if (ends - d).abs().dt.days.min() <= 7]
    if len(g) < 38 or not gaps.between(75, 130).all() or g["revenue"].le(0).any():
        continue
    if len(q4) < 5 or len({i % 4 for i in q4}) != 1:
        continue
    g["fq"] = (g.index - q4[0] + 3) % 4 + 1
    frames[t] = g
q = pd.concat(frames.values(), ignore_index=True)

# Complete fiscal years only, and only those whose four quarters reconcile to the
# company's reported annual revenue to within 1%.
q["block"] = q.groupby("ticker")["fq"].transform(lambda s: (s == 1).cumsum())
q["n_in_block"] = q.groupby(["ticker", "block"])["fq"].transform("size")
q["fy_rev"] = q.groupby(["ticker", "block"])["revenue"].transform("sum")
q["fy_end"] = q.groupby(["ticker", "block"])["period_end"].transform("max")
fy = q[q["n_in_block"] == 4].merge(
    ann[["ticker", "period_end", "revenue"]].rename(columns={"period_end": "fy_end", "revenue": "annual"}),
    on=["ticker", "fy_end"], how="inner")
fy["recon"] = (fy["fy_rev"] - fy["annual"]).abs() / fy["annual"]
fy = fy[fy["recon"] <= 0.01]
fy = fy[fy.groupby("ticker")["block"].transform("nunique") >= 8]
fy["share"] = fy["revenue"] / fy["fy_rev"]

prof = fy.pivot_table(index="ticker", columns="fq", values="share", aggfunc="mean")
prof.columns = [f"Q{c}" for c in prof.columns]
prof["spread"] = prof.max(axis=1) - prof.min(axis=1)
prof["peak"] = prof[["Q1", "Q2", "Q3", "Q4"]].idxmax(axis=1)
prof["years"] = fy.groupby("ticker")["block"].nunique()
prof["sector"] = fy.groupby("ticker")["gics_sector"].first()
prof = prof.sort_values("spread", ascending=False)

# The cost of reading the wrong growth number: the quarter right after each company's
# peak quarter, compared sequentially and against the same quarter a year earlier.
q = q[q["ticker"].isin(prof.index)].copy()
q["qoq"] = q.groupby("ticker")["revenue"].pct_change(1)
q["yoy"] = q.groupby("ticker")["revenue"].pct_change(4)
rows = []
for t in prof.index:
    nxt = int(prof.loc[t, "peak"][1]) % 4 + 1
    s = q[(q["ticker"] == t) & (q["fq"] == nxt)].dropna(subset=["qoq", "yoy"])
    rows.append({"ticker": t, "post_peak": f"Q{nxt}", "n": len(s),
                 "qoq": s["qoq"].median(), "yoy": s["yoy"].median(),
                 "false_drops": int(((s["qoq"] < 0) & (s["yoy"] > 0)).sum())})
mis = pd.DataFrame(rows).set_index("ticker")
mis["gap_pp"] = 100 * (mis["yoy"] - mis["qoq"])

print(f"Companies screened: {len(TICKERS)}   in final sample: {len(prof)}   "
      f"fiscal years used: {fy['block'].groupby(fy['ticker']).nunique().sum()}")
print("\nShare of fiscal-year revenue by fiscal quarter position")
print(f"{'':6}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'spread':>9}{'peak':>6}{'yrs':>5}  sector")
for t, r in prof.iterrows():
    print(f"{t:6}{r.Q1:8.1%}{r.Q2:8.1%}{r.Q3:8.1%}{r.Q4:8.1%}{r.spread:9.1%}{r.peak:>6}"
          f"{r.years:5.0f}  {r.sector}")

print("\nQuarter after the peak: sequential read vs year-over-year read")
print(f"{'':6}{'qtr':>5}{'QoQ':>9}{'YoY':>9}{'gap(pp)':>10}{'false drops':>13}")
for t, r in mis.loc[prof.index].iterrows():
    print(f"{t:6}{r.post_peak:>5}{r.qoq:9.1%}{r.yoy:9.1%}{r.gap_pp:10.1f}{r.false_drops:>8}/{r.n}")

# ---- chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [1, 1.15]})

show = list(prof.index[:4]) + list(prof.index[-3:])
palette = ["#3b82f6", "#f59e0b", "#22d3ee", "#a78bfa", "#6b7280", "#6b7280", "#6b7280"]
peak_month = fy.loc[fy["fq"] == fy["ticker"].map(lambda t: int(prof.loc[t, "peak"][1]))] \
               .groupby("ticker")["period_end"].agg(lambda s: s.max().strftime("%b"))
for name, colour in zip(show, palette):
    r = prof.loc[name, ["Q1", "Q2", "Q3", "Q4"]].astype(float).values
    flat = name in prof.index[-3:]
    tag = name if flat else f"{name}  peak {prof.loc[name, 'peak']} = {peak_month[name]}"
    ax1.plot([1, 2, 3, 4], r * 100, marker="o", ms=4, color=colour,
             lw=1.3 if flat else 2.1, ls="--" if flat else "-", label=tag)
ax1.axhline(25, color="#3a3a3a", lw=0.8)
ax1.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
ax1.set_xlabel("Fiscal quarter of the company's own year")
ax1.set_ylabel("Share of fiscal-year revenue (%)")
ax1.set_title("Where the year's revenue lands", color="#e0e0e0", fontsize=10)
ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

order = prof.sort_values("spread")
bars = ax2.barh(range(len(order)), order["spread"] * 100,
                color=["#3b82f6" if v > 0.04 else "#374151" for v in order["spread"]])
ax2.set_yticks(range(len(order)), order.index, fontsize=8)
ax2.set_xlabel("Seasonality: highest quarter share minus lowest (pp)")
ax2.set_title("How uneven the year is", color="#e0e0e0", fontsize=10)
for i, v in enumerate(order["spread"] * 100):
    ax2.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7.5, color="#9ca3af")
ax2.set_xlim(0, order["spread"].max() * 100 * 1.15)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Quarterly revenue seasonality, S&P 500 sample, fiscal 2016-2025",
             color="#e0e0e0", fontsize=12)
plt.tight_layout()
plt.savefig("quarterly-revenue-seasonality-python.png", dpi=150, facecolor="#0a0a0a")
