# Full write-up: https://xfinlink.com/blog/market-breadth-200-day-survivorship-bias-python
#
# How far does today's S&P 500 roster distort the history of a breadth indicator? On the
# last trading day of each month from December 1999 to December 2024, the share of members
# whose split-adjusted close is above its 200-day simple moving average is measured twice:
# on the roster in force that day (point-in-time) and on today's roster. A member counts
# when it has a close that day and at least 200 closes behind it. A timing rule then holds
# SPY (total return) in the following month when breadth is at or above 50%, and cash at 0%
# otherwise, once on each version of the indicator.
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Month-end trading days, taken from SPY's own sessions
spy = xfl.prices("SPY", start="1999-12-01", end="2024-12-31", fields=["return_daily"])
spy["date"] = pd.to_datetime(spy["date"])
month_ends = spy.groupby(spy["date"].dt.to_period("M"))["date"].max()

# Point-in-time rosters on each month end, and today's roster
pit = {d: set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).entity_id.dropna().astype(int))
       for d in month_ends}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

# Daily split-adjusted closes, keyed by entity id so a recycled ticker cannot swap companies
ids = sorted(set().union(*pit.values()) | today)
def fetch(i):
    return xfl.prices(entity_id=ids[i:i + 25], start="1999-01-01", end="2024-12-31",
                      fields=["adj_close"], max_rows=250_000)
with ThreadPoolExecutor(8) as pool:
    px = pd.concat(pool.map(fetch, range(0, len(ids), 25)))
px = px.dropna(subset=["adj_close"]).astype({"entity_id": int})
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date"])
px["ma200"] = px.groupby("entity_id")["adj_close"].transform(lambda s: s.rolling(200).mean())
obs = px[px["date"].isin(set(month_ends)) & px["ma200"].notna()]
above = obs.assign(up=obs["adj_close"] > obs["ma200"]).set_index(["date", "entity_id"])["up"]

def breadth(d, universe):
    """Share of the universe above its 200-day average on d, and how many members counted."""
    a = above.loc[d]
    a = a[a.index.isin(universe)]
    return a.mean() * 100, len(a)

rows = []
for d in month_ends:
    p, n_p = breadth(d, pit[d])
    s, n_s = breadth(d, today)
    k, n_k = breadth(d, pit[d] & today)  # members on d that are still members today
    rows.append({"date": d, "point_in_time": p, "todays_roster": s, "still_members": k,
                 "n_pit": n_p, "n_today": n_s, "n_still": n_k, "roster_size": len(pit[d])})
b = pd.DataFrame(rows).set_index("date")
b["gap"] = b["todays_roster"] - b["point_in_time"]
print(f"{len(b)} month ends, {len(ids)} securities; members counted per month: point-in-time "
      f"{b.n_pit.mean():.0f} of {b.roster_size.mean():.0f}, today's roster {b.n_today.mean():.0f}\n")

print("Share of members above their 200-day average (%), monthly averages")
periods = {"Dec 1999 - Dec 2024": ("1999", "2024"), "2000-2002 bear market": ("2000", "2002"),
           "2007-2009 crisis": ("2007-10", "2009-03"), "2010-2019": ("2010", "2019"),
           "2020-2024": ("2020", "2024")}
print(pd.DataFrame({k: b.loc[a:z, ["point_in_time", "todays_roster", "still_members", "gap"]].mean()
                    for k, (a, z) in periods.items()}).T.round(1).to_string(), "\n")

print("Readings on selected month ends (%)")
picks = ["2000-03", "2002-09", "2007-10", "2008-11", "2009-02", "2011-09", "2015-08", "2018-12",
         "2020-03", "2022-09", "2024-12"]
sel = pd.concat([b.loc[m] for m in picks])
sel.index = sel.index.strftime("%Y-%m")
print(sel[["point_in_time", "todays_roster", "gap"]].round(1).to_string(), "\n")
print(f"Gap (today's roster minus point-in-time): min {b.gap.min():.1f} ({b.gap.idxmin():%Y-%m}), "
      f"max {b.gap.max():.1f} ({b.gap.idxmax():%Y-%m}); months with gap above 10 points: "
      f"{(b.gap > 10).sum()} of {len(b)}\n")

print("Months in each zone")
zones = {"below 20% (washout)": lambda x: x < 20, "below 50%": lambda x: x < 50,
         "above 80%": lambda x: x > 80}
print(pd.DataFrame({z: {c: int(f(b[c]).sum()) for c in ["point_in_time", "todays_roster"]}
                    for z, f in zones.items()}).to_string(), "\n")

# Timing rule: the signal at a month end decides the following month
spy_m = spy.set_index("date")["return_daily"].fillna(0).groupby(pd.Grouper(freq="ME")).apply(
    lambda r: (1 + r).prod() - 1)
spy_m.index = spy_m.index.to_period("M")
sig = (b[["point_in_time", "todays_roster"]] >= 50).astype(int)
sig.index = sig.index.to_period("M") + 1
sig = sig.loc[sig.index <= spy_m.index[-1]]
ret = sig.mul(spy_m.reindex(sig.index), axis=0)
ret["SPY buy and hold"] = spy_m.reindex(sig.index)
def stats(r):
    wealth = (1 + r).cumprod()
    return pd.Series({"% a year": (wealth.iloc[-1] ** (12 / len(r)) - 1) * 100,
                      "max drawdown %": (wealth / wealth.cummax() - 1).min() * 100})
t = ret.apply(stats).T
t["months invested"] = list(sig.sum()) + [len(sig)]
print(f"Timing rule, {sig.index[0]} to {sig.index[-1]}: hold SPY next month when breadth >= 50%")
print(t.round(2).to_string())
dis = sig["point_in_time"] != sig["todays_roster"]
print(f"\nMonths the two signals disagree: {dis.sum()}; in {(dis & (sig.todays_roster == 1)).sum()} of them "
      f"only today's roster says invest. SPY in those months: mean "
      f"{spy_m.reindex(sig.index)[dis & (sig.todays_roster == 1)].mean() * 100:.2f}%")
print("  " + ", ".join(f"{m} ({spy_m[m] * 100:+.1f}%)" for m in sig.index[dis]))

# Chart: the two histories and the gap between them
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444"})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
a1.plot(b.index, b["point_in_time"], color="#3b82f6", lw=1.4, label="Point-in-time rosters")
a1.plot(b.index, b["todays_roster"], color="#f59e0b", lw=1.4, label="Today's roster")
a1.axhline(50, color="#888888", lw=0.8, ls="--")
a1.set_ylabel("Members above 200-day average (%)")
a1.set_ylim(0, 100)
a1.legend(frameon=False, loc="lower left", fontsize=9)
a2.fill_between(b.index, 0, b["gap"], color="#f59e0b", alpha=0.6, lw=0)
a2.axhline(0, color="#888888", lw=0.8)
a2.set_ylabel("Gap (points)")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Market Breadth: % of S&P 500 Members Above Their 200-Day Average, Month Ends 1999-2024",
             fontsize=12)
plt.tight_layout()
plt.savefig("market-breadth-200-day-survivorship-bias-python.png", dpi=150)
