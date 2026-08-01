# Full write-up: https://xfinlink.com/blog/low-volatility-anomaly-sp500-quintiles-python
"""Do low-volatility stocks deliver better risk-adjusted returns than high-volatility
stocks? Volatility quintiles built on point-in-time S&P 500 membership, 2016-2026.

Each 31 December the index membership as it stood on that date is pulled, every member
is ranked on the annualised standard deviation of its trailing 36 monthly total returns,
and the ranking is cut into five equal-weighted portfolios held through the following
calendar year. Nothing about the future enters the ranking or the universe.
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "low-volatility-anomaly-sp500-quintiles-python"
YEARS = list(range(2016, 2027))          # holding years; 2026 runs to July
FORMATION_MONTHS = 36
START, END = "2013-01-01", "2026-07-31"
LAST = "2026-07"
CHUNK = 25

# ── Point-in-time universe ────────────────────────────────────────────
# as_of returns membership as it stood on that date, so companies later removed
# from the index stay in the sample and companies later added stay out of it.
members = {}
for y in YEARS:
    ix = xfl.index("sp500", as_of=f"{y-1}-12-31")
    members[y] = sorted(int(e) for e in ix["entity_id"].dropna())

universe = sorted({e for ids in members.values() for e in ids})

# ── Monthly total returns ─────────────────────────────────────────────
# entity_id rather than ticker: an entity keeps one id through ticker changes,
# so a series stays continuous when a company renames or reticker.
parts = []
for i in range(0, len(universe), CHUNK):
    for attempt in range(5):
        try:
            parts.append(xfl.prices(entity_id=universe[i:i + CHUNK], start=START, end=END,
                                    interval="1mo", fields=["close", "return_daily"],
                                    max_rows=500000))
            break
        except xfl.XfinlinkError as exc:
            last = exc
            time.sleep(5)
    else:
        raise last  # never let a failed chunk leave a silent hole in the panel
px = pd.concat(parts, ignore_index=True)
raw_entities = px["entity_id"].nunique()

# ── Integrity screens ─────────────────────────────────────────────────
px = px.drop_duplicates(["entity_id", "date"]).dropna(subset=["return_daily"])
px = px[px["close"] > 0]
px["month"] = px["date"].dt.to_period("M")
sectors = px.drop_duplicates("entity_id").set_index("entity_id")["gics_sector"]
rets = px.pivot_table(index="month", columns="entity_id", values="return_daily", aggfunc="first")

# A monthly total return above +200% or below -90% is outside anything a large-cap
# equity produces, so the whole series is set aside rather than winsorised.
dropped = [c for c in rets.columns if (rets[c] > 2.0).any() or (rets[c] < -0.90).any()]
rets = rets.drop(columns=dropped)

# ── Quintile portfolios ───────────────────────────────────────────────
legs, everything, diag, formvol = {}, [], [], {}
for y in YEARS:
    form = pd.period_range(f"{y-FORMATION_MONTHS//12}-01", f"{y-1}-12", freq="M")
    hold = pd.period_range(f"{y}-01", LAST if y == YEARS[-1] else f"{y}-12", freq="M")
    window = rets.loc[rets.index.isin(form), [e for e in members[y] if e in rets.columns]]
    # a complete formation window, or the name does not get ranked this year
    ranked = window.columns[window.notna().sum() == FORMATION_MONTHS]
    vol = window[ranked].std() * np.sqrt(12)
    quintile = pd.qcut(vol.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    held = rets.loc[rets.index.isin(hold)]
    for k in range(1, 6):
        legs.setdefault(k, []).append(held[vol.index[quintile == k]].mean(axis=1))
        formvol.setdefault(k, []).append(vol[quintile == k].mean())
    everything.append(held[ranked].mean(axis=1))
    diag.append((y, len(members[y]), len(ranked), vol[quintile == 1].mean(), vol[quintile == 5].mean()))
    if y == YEARS[-1]:
        last_vol, last_q = vol, quintile

P = pd.DataFrame({f"Q{k}": pd.concat(v).sort_index() for k, v in legs.items()})
P["ALL"] = pd.concat(everything).sort_index()
mean_form = {k: float(np.mean(v)) for k, v in formvol.items()}


def summary(s):
    """CAGR, annualised volatility, Sharpe and worst peak-to-trough on month-end values."""
    cagr = (1 + s).prod() ** (12 / len(s)) - 1
    wealth = (1 + s).cumprod()
    return cagr, s.std() * np.sqrt(12), s.mean() / s.std() * np.sqrt(12), (wealth / wealth.cummax() - 1).min()


annual = (1 + P).groupby(P.index.year).prod() - 1
spread = P["Q1"] - P["Q5"]
gear = P["Q5"].std() / P["Q1"].std()
geared = gear * P["Q1"]

# ── Output ────────────────────────────────────────────────────────────
print(f"Point-in-time S&P 500 volatility quintiles, {P.index.min()} to {P.index.max()} "
      f"({len(P)} months)")
print(f"Formation: annualised standard deviation of the trailing {FORMATION_MONTHS} monthly total "
      "returns, measured each 31 December")
print("Portfolios: equal weighted, reformed every January, Sharpe taken without a risk-free deduction")
print(f"\n{len(YEARS)} membership snapshots cover {raw_entities} distinct companies; "
      f"{len(dropped)} set aside by the return-bound screen\n")

print(f"{'Formation':<11}{'Members':>9}{'Ranked':>8}{'Q1 vol':>9}{'Q5 vol':>9}")
for y, n, k, v1, v5 in diag:
    print(f"{y:<11}{n:>9}{k:>8}{v1:>9.1%}{v5:>9.1%}")

print(f"\n{'':<12}{'Formation vol':>14}{'CAGR':>9}{'Ann vol':>9}{'Sharpe':>8}"
      f"{'Worst mo':>10}{'Max DD':>9}")
labels = {1: "Q1 lowest", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 highest"}
for k in range(1, 6):
    c = f"Q{k}"
    a, v, sh, dd = summary(P[c])
    print(f"{labels[k]:<12}{mean_form[k]:>14.1%}{a:>9.2%}{v:>9.2%}{sh:>8.2f}"
          f"{P[c].min():>10.1%}{dd:>9.1%}")
a, v, sh, dd = summary(P["ALL"])
print(f"{'All ranked':<12}{'':>14}{a:>9.2%}{v:>9.2%}{sh:>8.2f}{P['ALL'].min():>10.1%}{dd:>9.1%}")

print("\nCalendar year total return (%)")
print(f"{'':<6}" + "".join(f"{c:>8}" for c in P.columns))
for y, row in annual.iterrows():
    print(f"{y:<6}" + "".join(f"{row[c]*100:>8.1f}" for c in P.columns))

print(f"\nQ1 minus Q5: {((1+spread).prod()**(12/len(spread))-1):.2%} a year, "
      f"t = {spread.mean()/spread.std()*np.sqrt(len(spread)):.2f}; "
      f"Q1 ahead in {int((annual['Q1']>annual['Q5']).sum())} of {len(annual)} calendar years")
for lo, hi, lab in [("2016-01", "2020-12", "2016-2020"), ("2021-01", LAST, "2021-2026")]:
    s1, s5 = summary(P.loc[lo:hi, "Q1"]), summary(P.loc[lo:hi, "Q5"])
    print(f"  {lab}  Q1 return {s1[0]:6.2%} Sharpe {s1[2]:5.2f}   "
          f"Q5 return {s5[0]:6.2%} Sharpe {s5[2]:5.2f}")
ga, gv, gsh, gdd = summary(geared)
print(f"Q1 geared {gear:.3f}x to Q5 volatility: {ga:.2%} a year before financing cost, "
      f"max drawdown {gdd:.1%}")
# highest annual financing rate on the borrowed leg that still leaves the geared
# low-volatility book ahead of Q5
lo, hi = 0.0, 0.30
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if summary(geared - (gear - 1) * mid / 12)[0] > summary(P["Q5"])[0] else (lo, mid)
print(f"Breakeven financing rate on the borrowed {gear-1:.3f} of capital: {lo:.2%} a year")

top = pd.concat([last_vol.rename("vol"), sectors], axis=1).dropna(subset=["vol"])
print(f"\n{YEARS[-1]} sector mix, lowest quintile: "
      + ", ".join(f"{s} {n}" for s, n in top[last_q.values == 1]["gics_sector"].value_counts().head(4).items()))
print(f"{YEARS[-1]} sector mix, highest quintile: "
      + ", ".join(f"{s} {n}" for s, n in top[last_q.values == 5]["gics_sector"].value_counts().head(4).items()))

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "axes.edgecolor": "#333333",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0", "figure.dpi": 150, "font.size": 10,
})
colours = ["#3b82f6", "#7dd3fc", "#94a3b8", "#fbbf24", "#f97316"]
widths = [2.4, 1.2, 1.2, 1.2, 2.4]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.35, 1]})

x = P.index.to_timestamp()
for k in range(1, 6):
    ax1.plot(x, (1 + P[f"Q{k}"]).cumprod(), color=colours[k - 1], linewidth=widths[k - 1],
             label=f"Q{k} {'lowest volatility' if k == 1 else 'highest volatility' if k == 5 else ''}".strip())
ax1.set_ylabel("Growth of $1 invested")
ax1.set_title("Do low-volatility stocks deliver better risk-adjusted returns?\n"
              "S&P 500 volatility quintiles, point-in-time membership, 2016-2026", pad=12)
ax1.legend(frameon=False, loc="upper left", fontsize=9)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

pos = np.arange(5)
ret = [summary(P[f"Q{k}"])[0] * 100 for k in range(1, 6)]
vol = [summary(P[f"Q{k}"])[1] * 100 for k in range(1, 6)]
sha = [summary(P[f"Q{k}"])[2] for k in range(1, 6)]
ax2.bar(pos - 0.2, ret, 0.4, color="#3b82f6", label="Return a year")
ax2.bar(pos + 0.2, vol, 0.4, color="#f97316", label="Volatility a year")
for i, s in enumerate(sha):
    ax2.text(i, max(ret[i], vol[i]) + 1.2, f"Sharpe {s:.2f}", ha="center", fontsize=9, color="#e0e0e0")
ax2.set_xticks(pos)
ax2.set_xticklabels(["Q1\nlowest volatility", "Q2", "Q3", "Q4", "Q5\nhighest volatility"])
ax2.set_ylabel("Percent a year")
ax2.set_ylim(0, max(vol) + 6)
ax2.legend(frameon=False, loc="upper left", fontsize=9)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
