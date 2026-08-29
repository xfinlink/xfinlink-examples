# Full write-up: https://xfinlink.com/blog/do-insiders-predict-their-own-stock-returns-python
"""
Do company insiders predict their own stock's returns?

Officers and directors report open-market trades in their own company on SEC
Form 4. This script classifies every S&P 500 member in every quarter from 2014
to mid-2025 by the direction of those trades, then measures what the stock did
over the following one to twelve months. Membership is point-in-time, so
companies that later left the index stay in for the quarters they were in it.

Transaction counts drive the classification. Share counts, prices and dollar
values play no part.
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

SLUG = "do-insiders-predict-their-own-stock-returns-python"
HORIZONS = [1, 3, 6, 12]
GROUPS = ["Insider buying only", "Buying and selling",
          "Insider selling only", "No open-market trades"]
COLOURS = {"Insider buying only": "#3b82f6", "Buying and selling": "#93c5fd",
           "Insider selling only": "#f97316", "No open-market trades": "#6b7280"}
INS_FIELDS = ["entity_id", "transaction_date", "transaction_code", "insider_name"]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch(fn, **kw):
    for attempt in range(4):
        try:
            return fn(**kw)
        except Exception as exc:                       # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(6)
    return pd.DataFrame()


# 1. The index as it stood at the end of each quarter.
quarter_ends = pd.date_range("2014-03-31", "2025-06-30", freq="QE")
rosters = {q: set(xfl.index("sp500", as_of=q.strftime("%Y-%m-%d"))["entity_id"])
           for q in quarter_ends}
universe = sorted(set().union(*rosters.values()))

# 2. Every Form 4 line for those companies, one quarter at a time.
parts = []
for chunk in chunked(universe, 50):
    for q in quarter_ends:
        got = fetch(xfl.insiders, entity_id=chunk, form_type="4",
                    fields=INS_FIELDS,
                    start=q.to_period("Q").start_time.strftime("%Y-%m-%d"),
                    end=q.strftime("%Y-%m-%d"))
        if len(got):
            parts.append(got)
ins = pd.concat(parts, ignore_index=True).drop_duplicates()

# Codes are read as filed and normalised for case before anything is selected.
# P is an open-market purchase, S an open-market sale. Grants, option
# exercises, tax withholding and gifts are compensation mechanics rather than
# a decision to trade, so they play no part.
ins["code"] = ins["transaction_code"].astype(str).str.strip().str.upper()
ins["tdate"] = pd.to_datetime(ins["transaction_date"], utc=True).dt.tz_localize(None)
trades = ins[ins["code"].isin(["P", "S"])].copy()
trades["q"] = trades["tdate"].dt.to_period("Q")

# 3. Monthly total returns for the same companies.
parts = []
for chunk in chunked(universe, 50):
    got = fetch(xfl.prices, entity_id=chunk, start="2014-01-01", end="2026-08-31",
                interval="1mo", fields=["return_daily"], max_rows=200000)
    if len(got):
        parts.append(got[["entity_id", "date", "return_daily"]])
px = pd.concat(parts, ignore_index=True).dropna(subset=["return_daily"])
px["m"] = px["date"].dt.tz_localize(None).dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
gross = 1.0 + px.pivot(index="m", columns="entity_id", values="return_daily").sort_index()


def window_return(months, ids):
    cols = [c for c in ids if c in gross.columns]
    if not set(months).issubset(set(gross.index)):
        return pd.Series(dtype=float)
    r = (gross.loc[months, cols].prod(axis=0, skipna=False) - 1.0).dropna()
    lo, hi = r.quantile([0.01, 0.99])
    return r.clip(lo, hi)


# 4. Classify each company-quarter, then look forward and backward.
forward, backward = [], []
for qend, members in sorted(rosters.items()):
    q = pd.Period(qend, freq="Q")
    end_month = q.end_time.to_period("M")

    sub = trades[(trades["q"] == q) & (trades["entity_id"].isin(members))]
    d = pd.DataFrame(index=pd.Index(sorted(members), name="entity_id"))
    for code, col in (("P", "n_buy"), ("S", "n_sell")):
        n = sub[sub["code"] == code].groupby("entity_id").size()
        d[col] = n.reindex(d.index).fillna(0).astype(int)
    d["group"] = np.select(
        [(d.n_buy > 0) & (d.n_sell == 0), (d.n_buy > 0) & (d.n_sell > 0),
         (d.n_buy == 0) & (d.n_sell > 0)], GROUPS[:3], default=GROUPS[3])

    prior = window_return(pd.period_range(end_month - 11, periods=12, freq="M"), d.index)
    if len(prior):
        t = d.loc[prior.index].copy()
        t["abn"] = prior.values - prior.mean()
        backward.append(t)

    # A full month passes between the quarter closing and the holding period
    # starting, so every Form 4 from the quarter is public before any return
    # is counted.
    for h in HORIZONS:
        fwd = window_return(pd.period_range(end_month + 2, periods=h, freq="M"), d.index)
        if len(fwd) < 100:
            continue
        t = d.loc[fwd.index].copy()
        t["ret"] = fwd.values
        t["abn"] = fwd.values - fwd.mean()
        t["q"], t["h"] = str(q), h
        forward.append(t.reset_index())

panel = pd.concat(forward, ignore_index=True)
before = pd.concat(backward)

print(f"companies in the point-in-time universe: {len(universe)}")
print(f"open-market purchase transactions {int((trades.code == 'P').sum()):,}  "
      f"sale transactions {int((trades.code == 'S').sum()):,}")
print(f"formation quarters {panel['q'].nunique()}  "
      f"{panel['q'].min()} to {panel['q'].max()}")

base = panel[panel["h"] == 12]
print(f"\ncompany-quarters by insider activity, 12-month window "
      f"({len(base):,} in total)")
prior_mean = before.groupby("group")["abn"].mean() * 100
for g in GROUPS:
    n = int((base["group"] == g).sum())
    print(f"  {g:<22} {n:>6,} ({n / len(base):5.1%})   "
          f"prior 12m vs index {prior_mean[g]:+6.2f}%")

raw = panel.pivot_table(index="group", columns="h", values="ret", aggfunc="mean") * 100
abn = panel.pivot_table(index="group", columns="h", values="abn", aggfunc="mean") * 100
print("\nforward return after the signal, mean across company-quarters (%)")
print(f"{'':<23}" + "".join(f"{h:>7}m" for h in HORIZONS) + "   " +
      "".join(f"{h:>7}m" for h in HORIZONS))
print(f"{'':<23}{'raw':>32}{'vs index':>32}")
for g in GROUPS:
    print(f"  {g:<21}" + "".join(f"{raw.loc[g, h]:>7.2f} " for h in HORIZONS) + "  " +
          "".join(f"{abn.loc[g, h]:>7.2f} " for h in HORIZONS))

print("\nspread between groups, tested on the 46 quarterly cross-sectional means")
pairs = [("Insider buying only", "Insider selling only"),
         ("Insider buying only", "No open-market trades"),
         ("Insider selling only", "No open-market trades")]
for h in HORIZONS:
    per_q = panel[panel["h"] == h].pivot_table(
        index="q", columns="group", values="abn", aggfunc="mean")
    for a, b in pairs:
        s = (per_q[a] - per_q[b]).dropna()
        t_stat, p_val = stats.ttest_1samp(s, 0.0)
        print(f"  {h:>2}m  {a} minus {b:<22} {s.mean() * 100:+6.2f}pp  "
              f"t={t_stat:+5.2f}  p={p_val:.3f}")

per_q = panel[panel["h"] == 12].pivot_table(
    index="q", columns="group", values="abn", aggfunc="mean")
spread = per_q["Insider buying only"] - per_q["Insider selling only"]
print(f"\n12-month window: buying-only beat selling-only in "
      f"{int((spread > 0).sum())} of {len(spread)} quarters")
for label, mask in (("2014Q1-2019Q4", spread.index < "2020Q1"),
                    ("2020Q1-2025Q2", spread.index >= "2020Q1")):
    s = spread[mask]
    t_stat, p_val = stats.ttest_1samp(s, 0.0)
    print(f"  {label}: {s.mean() * 100:+6.2f}pp  t={t_stat:+5.2f}  "
          f"p={p_val:.3f}  n={len(s)}")

# 5. Chart: what the groups looked like before the signal, and after it.
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11})
short = ["Buying\nonly", "Buying and\nselling", "Selling\nonly", "No open-market\ntrades"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
panels = [("The 12 months before the signal", [prior_mean[g] for g in GROUPS]),
          ("The 12 months after the signal", [abn.loc[g, 12] for g in GROUPS])]
for ax, (title, values) in zip(axes, panels):
    ax.bar(short, values, color=[COLOURS[g] for g in GROUPS], width=0.62)
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-19.5, 8.0)
    ax.spines[["top", "right"]].set_visible(False)
    for x, v in enumerate(values):
        ax.text(x, v + (0.7 if v >= 0 else -1.9), f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, color="#e0e0e0")
axes[0].set_ylabel("Return relative to the index (%)")
fig.suptitle("S&P 500 stocks grouped by insider trade direction, 2014-2025",
             color="#e0e0e0", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150)
print(f"\nchart written to {SLUG}.png")
