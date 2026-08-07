# Full write-up: https://xfinlink.com/blog/missing-best-worst-days-market-timing-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = {"SPY": "US large cap", "IWM": "US small cap", "EFA": "Developed intl", "EEM": "Emerging mkts"}
START, END = "1996-01-01", "2024-12-31"


def series(t):
    d = xfl.prices(t, start=START, end=END, fields=["close", "return_daily"]).sort_values("date")
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna(subset=["return_daily"]).set_index("date")["return_daily"]


def cagr(r):
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return ((1 + r).prod() ** (1 / yrs) - 1) * 100


def drop(r, n, which):
    if n == 0:
        return r
    order = r.sort_values()
    if which == "best":
        kill = order.index[-n:]
    elif which == "worst":
        kill = order.index[:n]
    else:
        kill = order.index[-n:].union(order.index[:n])
    return r.drop(kill)


rows = []
for t, label in FUNDS.items():
    r = series(t)
    row = {"ticker": t, "label": label, "n": len(r), "from": r.index.min().year, "full": cagr(r)}
    for n in (10, 20, 30):
        row[f"miss_best_{n}"] = cagr(drop(r, n, "best"))
        row[f"miss_worst_{n}"] = cagr(drop(r, n, "worst"))
        row[f"miss_both_{n}"] = cagr(drop(r, n, "both"))
    rows.append(row)
    if t == "SPY":
        spy = r
tab = pd.DataFrame(rows)

print(f"Daily total returns, {START} to {END}")
print(f"{'':4s} {'exposure':15s} {'days':>6s} {'all in':>7s} {'-10 best':>9s} {'-10 worst':>10s} {'-10 both':>9s}")
for r in tab.itertuples():
    print(f"{r.ticker:4s} {r.label:15s} {r.n:6d} {r.full:6.2f}% {r.miss_best_10:8.2f}% "
          f"{r.miss_worst_10:9.2f}% {r.miss_both_10:8.2f}%")

print(f"\nSPY only, deeper cuts")
print(f"{'removed':>9s} {'best':>8s} {'worst':>8s} {'both':>8s}")
s = tab[tab.ticker == "SPY"].iloc[0]
print(f"{0:9d} {s.full:7.2f}% {s.full:7.2f}% {s.full:7.2f}%")
for n in (10, 20, 30):
    print(f"{n:9d} {s[f'miss_best_{n}']:7.2f}% {s[f'miss_worst_{n}']:7.2f}% {s[f'miss_both_{n}']:7.2f}%")

# where do the extreme days live, and how close together are they
best20 = spy.nlargest(20).sort_index()
worst20 = spy.nsmallest(20).sort_index()
pos = {d: i for i, d in enumerate(spy.index)}
gaps = [min(abs(pos[b] - pos[w]) for w in worst20.index) for b in best20.index]
print(f"\n20 best days: median {int(np.median(gaps))} sessions from the nearest of the 20 worst; "
      f"{sum(1 for g in gaps if g <= 5)} of 20 within a week, {sum(1 for g in gaps if g <= 20)} within a month")

yrs = pd.Series([d.year for d in best20.index]).value_counts().sort_index()
yrsw = pd.Series([d.year for d in worst20.index]).value_counts().sort_index()
print("best-20 by year: ", ", ".join(f"{y}:{c}" for y, c in yrs.items()))
print("worst-20 by year:", ", ".join(f"{y}:{c}" for y, c in yrsw.items()))
both = sorted(set(yrs.index) & set(yrsw.index))
share = sum(yrs.get(y, 0) for y in both) / 20 * 100
print(f"years holding both a best-20 and a worst-20 day: {both} ({share:.0f}% of best days)")

print("\nthe five largest single-session gains and what preceded them")
for d, v in spy.nlargest(5).items():
    i = pos[d]
    prev5 = (1 + spy.iloc[max(0, i - 5):i]).prod() - 1
    print(f"  {d.date()}  +{v*100:5.2f}%   prior 5 sessions {prev5*100:7.2f}%")

# chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1.3]})
ns = [0, 10, 20, 30]
a1.plot(ns, [s.full] + [s[f"miss_best_{n}"] for n in (10, 20, 30)], "o-", color="#ef4444", label="Missing the best days")
a1.plot(ns, [s.full] + [s[f"miss_worst_{n}"] for n in (10, 20, 30)], "o-", color="#3b82f6", label="Missing the worst days")
a1.plot(ns, [s.full] + [s[f"miss_both_{n}"] for n in (10, 20, 30)], "o-", color="#9ca3af", label="Missing both")
a1.axhline(s.full, color="#3a3a3a", linewidth=0.8, linestyle="--")
a1.set_xlabel("Number of days removed")
a1.set_ylabel("Annual return (%)")
a1.set_title("SPY 1996-2024, days removed")
a1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", fontsize=8)
a2.scatter(best20.index, best20.values * 100, color="#3b82f6", s=28, label="20 best days")
a2.scatter(worst20.index, worst20.values * 100, color="#ef4444", s=28, marker="v", label="20 worst days")
a2.axhline(0, color="#3a3a3a", linewidth=0.8)
a2.set_ylabel("Daily total return (%)")
a2.set_title("The extreme days arrive together")
a2.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", fontsize=8)
plt.tight_layout()
plt.savefig("missing-best-worst-days-market-timing-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart written")
