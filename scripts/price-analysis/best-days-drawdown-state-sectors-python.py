# Full write-up: https://xfinlink.com/blog/best-days-drawdown-state-sectors-python
#
# Drawdown state at the market's best and worst sessions, measured per fund across
# SPY and the nine Select Sector SPDRs, 1999-2025, against the base rate of sessions
# spent in a drawdown. The removal test is carried as supporting evidence.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = {
    "SPY": "US large cap",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons Discretionary",
}
START, END = "1999-01-01", "2025-12-31"

# One call per fund: a ten-ticker daily request over 27 years exceeds the row cap.
ret, draw = {}, {}
for t in FUNDS:
    px = xfl.prices(t, start="1998-12-01", end=END, fields=["adj_close"])
    s = px.sort_values("date").set_index("date")["adj_close"]
    r = s.pct_change().dropna()
    ret[t] = r[r.index >= START]
    level = (1 + ret[t]).cumprod()
    draw[t] = (level / level.cummax() - 1).shift(1)  # drawdown at the previous close

screened = [t for t in FUNDS if ret[t].abs().max() > 0.50]
for t in screened:
    del ret[t], draw[t]

sessions = {t: len(r) for t, r in ret.items()}
per_year = ret["SPY"].groupby(ret["SPY"].index.year).size()
widest = max(ret, key=lambda t: ret[t].abs().max())


def annualised(r):
    years = (r.index[-1] - r.index[0]).days / 365.25
    return ((1 + r).prod() ** (1 / years) - 1) * 100


def without(r, n, side):
    order = r.sort_values()
    cut = {
        "best": order.index[-n:],
        "worst": order.index[:n],
        "both": order.index[-n:].union(order.index[:n]),
    }[side]
    return r.drop(cut)


best = {t: ret[t].nlargest(20).index for t in ret}
worst = {t: ret[t].nsmallest(20).index for t in ret}

table = pd.DataFrame(
    [
        {
            "fund": t,
            "sector": FUNDS[t],
            "all_in": annualised(ret[t]),
            "no_best": annualised(without(ret[t], 10, "best")),
            "no_worst": annualised(without(ret[t], 10, "worst")),
            "no_both": annualised(without(ret[t], 10, "both")),
            "dd_best": draw[t].loc[best[t]].median() * 100,
            "dd_worst": draw[t].loc[worst[t]].median() * 100,
            "dd_all": draw[t].median() * 100,
            "best_deep": int((draw[t].loc[best[t]] < -0.10).sum()),
        }
        for t in ret
    ]
)

deep_best = sum(int((draw[t].loc[best[t]] < -0.10).sum()) for t in ret)
deep_worst = sum(int((draw[t].loc[worst[t]] < -0.10).sum()) for t in ret)
very_deep_best = sum(int((draw[t].loc[best[t]] < -0.20).sum()) for t in ret)
base_10 = np.mean([(draw[t] < -0.10).mean() for t in ret]) * 100
base_20 = np.mean([(draw[t] < -0.20).mean() for t in ret]) * 100

shared = pd.Series([d for t in ret for d in best[t]]).value_counts()
universal = sorted(shared[shared == len(ret)].index)
by_year = pd.Series([d.year for t in ret for d in best[t]]).value_counts()

print(f"SPY and the nine Select Sector SPDRs, price returns from split-adjusted closes")
print(f"Window {ret['SPY'].index[0]:%Y-%m-%d} to {ret['SPY'].index[-1]:%Y-%m-%d}   "
      f"funds: {len(ret)}   sessions per fund: {min(sessions.values()):,}")
print(f"Sessions per calendar year: min {per_year.min()}, median {int(per_year.median())}, "
      f"max {per_year.max()}   every fund priced on every session: "
      f"{len(set(sessions.values())) == 1}")
print(f"Largest single session in the sample: {ret[widest].abs().max() * 100:.2f}% ({widest})   "
      f"funds screened out above 50%: {len(screened)}")

print("\nDrawdown at the previous close, by session type")
print(f"{'fund':<6}{'20 best':>10}{'20 worst':>10}{'all days':>10}{'best below -10%':>18}")
for _, x in table.iterrows():
    print(f"{x.fund:<6}{x.dd_best:>9.1f}%{x.dd_worst:>9.1f}%{x.dd_all:>9.1f}%"
          f"{x.best_deep:>13} of 20")

print(f"\nAcross the ten funds: {deep_best} of 200 best sessions and {deep_worst} of 200 worst "
      f"sessions landed while the fund")
print(f"was already more than 10% below its trailing peak, against {base_10:.1f}% of all sessions. "
      f"{very_deep_best} of 200")
print(f"best sessions landed more than 20% below it, against {base_20:.1f}% of all sessions.")
print(f"\nTop-20 best sessions shared by all ten funds: "
      f"{', '.join(d.strftime('%Y-%m-%d') for d in universal)}")
print(f"Of the 200 best sessions, {by_year.get(2008, 0) + by_year.get(2020, 0)} fall in 2008 "
      f"({by_year.get(2008, 0)}) or 2020 ({by_year.get(2020, 0)}).")

print("\nSupporting: annualised price return with ten sessions removed")
print(f"{'fund':<6}{'sector':<20}{'all in':>9}{'-10 best':>10}{'-10 worst':>11}{'-10 both':>10}")
for _, x in table.iterrows():
    print(f"{x.fund:<6}{x.sector:<20}{x.all_in:>8.2f}%{x.no_best:>9.2f}%"
          f"{x.no_worst:>10.2f}%{x.no_both:>9.2f}%")

# ----- chart -----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

spy_dd = draw["SPY"].dropna() * 100
ax1.fill_between(spy_dd.index, spy_dd.values, 0, color="#3b82f6", alpha=0.25, linewidth=0)
ax1.plot(spy_dd.index, spy_dd.values, color="#3b82f6", linewidth=0.7)
ax1.scatter(best["SPY"], draw["SPY"].loc[best["SPY"]] * 100, s=34, color="#22d3ee",
            zorder=3, label="20 best sessions")
ax1.scatter(worst["SPY"], draw["SPY"].loc[worst["SPY"]] * 100, s=34, color="#f59e0b",
            marker="v", zorder=3, label="20 worst sessions")
ax1.set_title("SPY: the best and worst sessions sit inside drawdowns, 1999-2025",
              color="#e0e0e0", fontsize=11)
ax1.set_ylabel("Percent below trailing peak")
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0", loc="lower right")
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

x = np.arange(len(table))
ax2.bar(x - 0.18, table.dd_best, 0.34, color="#22d3ee", label="median on the 20 best sessions")
ax2.bar(x + 0.18, table.dd_all, 0.34, color="#3b82f6", label="median on a typical session")
ax2.axhline(0, color="#666666", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(table.fund)
ax2.set_ylabel("Percent below trailing peak")
ax2.set_title("How far below its peak each fund sat when its best sessions arrived",
              color="#e0e0e0", fontsize=11)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           ncol=2, loc="lower right")
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout(h_pad=2.5)
plt.savefig("best-days-drawdown-state-sectors-python.png", dpi=150, facecolor="#0a0a0a")
