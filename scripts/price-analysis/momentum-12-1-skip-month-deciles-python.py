# Full write-up: https://xfinlink.com/blog/momentum-12-1-skip-month-deciles-python
"""Does skipping the most recent month improve cross-sectional price momentum?
Decile sorts on point-in-time S&P 500 membership, January 2006 to July 2026.

At every month end the index membership as it stood on that date is pulled, every member
with a complete twelve-month history is ranked twice - once on the cumulative total return
over months t-11 to t-1 (the standard 12-1 construction, most recent month skipped) and
once over months t-11 to t (the same window with that month left in) - and each ranking is
cut into ten equal-weighted deciles held for the following month. The two rankings differ
by one month of data and nothing else, so the gap between their decile spreads is what the
skip is worth. The skipped month is also ranked on its own, which tests the short-horizon
reversal the convention exists to remove.
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "momentum-12-1-skip-month-deciles-python"
FIRST_FORM, LAST_FORM = "2005-12-31", "2026-06-30"   # formation month ends
START, END = "2004-12-01", "2026-08-05"              # price window, with the run-up
CHUNK = 20

# ── Point-in-time universe ────────────────────────────────────────────
# as_of returns membership as it stood on that date, so a company removed in 2011 is
# ranked in every month up to its removal and in none after it. Ranking today's members
# across the past twenty years would rank the survivors and call the result momentum.
members = {}
for d in pd.date_range(FIRST_FORM, LAST_FORM, freq="ME").strftime("%Y-%m-%d"):
    ix = xfl.index("sp500", as_of=d)
    members[pd.Period(d, "M")] = sorted(int(e) for e in ix["entity_id"].dropna())

universe = sorted({e for ids in members.values() for e in ids})

# ── Monthly total returns ─────────────────────────────────────────────
# entity_id rather than ticker: an entity keeps one id through ticker changes, so a series
# stays continuous when a company renames, and a ticker later reissued to a different
# company cannot splice that company's prices onto this one's history.
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
            time.sleep(10 * (attempt + 1))
    else:
        raise last  # never let a failed chunk leave a silent hole in the panel
px = pd.concat(parts, ignore_index=True)
raw_entities = px["entity_id"].nunique()

# ── Integrity screens ─────────────────────────────────────────────────
px = px.drop_duplicates(["entity_id", "date"]).dropna(subset=["return_daily"])
px = px[px["close"] > 0]
px["month"] = px["date"].dt.to_period("M")
rets = px.pivot_table(index="month", columns="entity_id", values="return_daily", aggfunc="first")

# A monthly total return above +200% or below -90% is far outside what a large-cap equity
# produces, so the whole series is set aside rather than winsorised. The screen is blunt on
# purpose: it removes a handful of genuine crisis moves along with anything unusable.
dropped = [c for c in rets.columns if (rets[c] > 2.0).any() or (rets[c] < -0.90).any()]
rets = rets.drop(columns=dropped)

# ── Decile portfolios ─────────────────────────────────────────────────
SIGNALS = ("skip", "noskip", "reversal")
legs = {s: {k: {} for k in range(1, 11)} for s in SIGNALS}
diag = []

for T in sorted(members):
    H = T + 1                                     # the month the portfolios are held
    if H not in rets.index:
        continue
    window = rets.loc[(rets.index >= T - 11) & (rets.index <= T)]
    if len(window) != 12:
        continue
    cols = [e for e in members[T] if e in rets.columns]
    window = window[cols]
    # a complete twelve-month formation window and a return for the holding month,
    # or the name is not ranked this month
    usable = window.notna().all() & rets.loc[H, cols].notna()
    window = window.loc[:, usable[usable].index]
    held = rets.loc[H, window.columns]

    signal = {
        "skip": (1 + window.iloc[:-1]).prod() - 1,   # months t-11 to t-1
        "noskip": (1 + window).prod() - 1,           # months t-11 to t
        "reversal": window.iloc[-1],                 # month t on its own
    }
    decile = {}
    for name, s in signal.items():
        decile[name] = pd.qcut(s.rank(method="first"), 10, labels=list(range(1, 11))).astype(int)
        for k in range(1, 11):
            legs[name][k][H] = held[s.index[decile[name] == k]].mean()

    a, b = decile["skip"], decile["noskip"]
    diag.append((T, len(members[T]), window.shape[1],
                 float((signal["skip"].rank(pct=True) - signal["noskip"].rank(pct=True)).abs().mean()),
                 float((a == b).mean()),
                 float(((a == 10) & (b == 10)).sum() / (a == 10).sum()),
                 float(((a == 1) & (b == 1)).sum() / (a == 1).sum())))

P = {name: pd.DataFrame({f"D{k}": pd.Series(v[k]).sort_index() for k in range(1, 11)})
     for name, v in legs.items()}
for name in SIGNALS:
    P[name]["LS"] = P[name]["D10"] - P[name]["D1"]

D = pd.DataFrame(diag, columns=["form", "members", "ranked", "rankmove",
                                "same", "top_same", "bottom_same"])
months = P["skip"].index


def summary(s):
    """CAGR, annualised volatility, Sharpe without a risk-free deduction, t-statistic of
    the monthly mean, and worst peak-to-trough on month-end values."""
    wealth = (1 + s).cumprod()
    return ((1 + s).prod() ** (12 / len(s)) - 1, s.std() * np.sqrt(12),
            s.mean() / s.std() * np.sqrt(12), s.mean() / (s.std() / np.sqrt(len(s))),
            (wealth / wealth.cummax() - 1).min())


# ── Output ────────────────────────────────────────────────────────────
print(f"Cross-sectional momentum deciles on point-in-time S&P 500 membership, "
      f"{months.min()} to {months.max()} ({len(months)} holding months)")
print("Formation: cumulative total return over months t-11 to t-1 (12-1, most recent month "
      "skipped)\n           and over months t-11 to t (12-0, most recent month kept)")
print("Portfolios: ten equal-weighted deciles, re-formed at each month end, held one month")
print(f"\n{len(members)} membership snapshots cover {len(universe)} distinct companies; "
      f"{raw_entities} carry a monthly price history and {len(dropped)} of those are set "
      f"aside by the return-bound screen")
print(f"Names ranked each month: {int(D['ranked'].min())} to {int(D['ranked'].max())}, "
      f"median {int(D['ranked'].median())}")

print(f"\n{'':<14}{'12-1 skip month':>26}{'12-0 month kept':>26}")
print(f"{'Decile':<14}{'CAGR':>9}{'Ann vol':>9}{'Sharpe':>8}{'CAGR':>9}{'Ann vol':>9}{'Sharpe':>8}")
labels = {1: "D1 losers", 10: "D10 winners"}
for k in range(1, 11):
    a, b = summary(P["skip"][f"D{k}"]), summary(P["noskip"][f"D{k}"])
    print(f"{labels.get(k, f'D{k}'):<14}{a[0]:>9.2%}{a[1]:>9.2%}{a[2]:>8.2f}"
          f"{b[0]:>9.2%}{b[1]:>9.2%}{b[2]:>8.2f}")

print("\nWinners minus losers, equal weighted, rebalanced monthly")
sa, sb = summary(P["skip"]["LS"]), summary(P["noskip"]["LS"])
print(f"{'':<28}{'12-1 skip':>12}{'12-0 kept':>12}")
for label, i, fmt in [("Return a year", 0, "{:.2%}"), ("Annualised volatility", 1, "{:.2%}"),
                      ("Sharpe", 2, "{:.2f}"), ("t-statistic of monthly mean", 3, "{:.2f}"),
                      ("Worst drawdown", 4, "{:.1%}")]:
    print(f"{label:<28}{fmt.format(sa[i]):>12}{fmt.format(sb[i]):>12}")
print(f"{'Worst month':<28}{P['skip']['LS'].min():>12.1%}{P['noskip']['LS'].min():>12.1%}")
print(f"{'Best month':<28}{P['skip']['LS'].max():>12.1%}{P['noskip']['LS'].max():>12.1%}")
print(f"{'Months positive':<28}{(P['skip']['LS'] > 0).mean():>12.1%}"
      f"{(P['noskip']['LS'] > 0).mean():>12.1%}")
ex09 = months.year != 2009
ea, eb = summary(P["skip"]["LS"][ex09]), summary(P["noskip"]["LS"][ex09])
print(f"{'Return a year, 2009 removed':<28}{ea[0]:>12.2%}{eb[0]:>12.2%}")
print(f"{'t-statistic, 2009 removed':<28}{ea[3]:>12.2f}{eb[3]:>12.2f}")

print(f"\nHow much the skip changes the ranking")
print(f"  mean absolute move in percentile rank      {D['rankmove'].mean():.1%}")
print(f"  names landing in the same decile           {D['same'].mean():.1%}")
print(f"  top decile shared by both constructions    {D['top_same'].mean():.1%}")
print(f"  bottom decile shared by both               {D['bottom_same'].mean():.1%}")
print(f"  correlation of the two spread series       "
      f"{P['skip']['LS'].corr(P['noskip']['LS']):.3f}")

r = summary(P["reversal"]["LS"])
print(f"\nThe skipped month ranked on its own (highest minus lowest last-month return): "
      f"{r[0]:.2%} a year, t = {r[3]:.2f}")
print(f"  its decile returns run {summary(P['reversal']['D1'])[0]:.2%} at D1 (worst last month) "
      f"to {summary(P['reversal']['D10'])[0]:.2%} at D10 (best last month)")

annual = pd.DataFrame({"12-1 skip": P["skip"]["LS"], "12-0 kept": P["noskip"]["LS"],
                       "last month only": P["reversal"]["LS"]})
annual = (1 + annual).groupby(annual.index.year).prod() - 1
print("\nWinners minus losers, calendar year total return (%)")
print(f"{'':<6}" + "".join(f"{c:>18}" for c in annual.columns))
for y, row in annual.iterrows():
    print(f"{y:<6}" + "".join(f"{row[c] * 100:>18.1f}" for c in annual.columns))

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "axes.edgecolor": "#333333",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0", "figure.dpi": 150, "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.1]})

pos = np.arange(10)
ax1.bar(pos - 0.2, [summary(P["skip"][f"D{k}"])[0] * 100 for k in range(1, 11)], 0.4,
        color="#3b82f6", label="12-1, most recent month skipped")
ax1.bar(pos + 0.2, [summary(P["noskip"][f"D{k}"])[0] * 100 for k in range(1, 11)], 0.4,
        color="#f97316", label="12-0, most recent month kept")
ax1.set_xticks(pos)
ax1.set_xticklabels(["D1\nlosers"] + [f"D{k}" for k in range(2, 10)] + ["D10\nwinners"])
ax1.set_ylabel("Return a year, percent")
ax1.set_ylim(0, 14)
ax1.set_title("Does skipping the most recent month improve momentum?\n"
              "S&P 500 momentum deciles, point-in-time membership, 2006-2026", pad=12)
ax1.legend(frameon=False, loc="upper left", fontsize=9)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

x = months.to_timestamp()
ax2.plot(x, (1 + P["skip"]["LS"]).cumprod(), color="#3b82f6", linewidth=2,
         label="12-1, most recent month skipped")
ax2.plot(x, (1 + P["noskip"]["LS"]).cumprod(), color="#f97316", linewidth=2,
         label="12-0, most recent month kept")
ax2.axhline(1, color="#333333", linewidth=0.8)
ax2.set_ylabel("Growth of $1, winners minus losers")
ax2.legend(frameon=False, loc="upper right", fontsize=9)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
