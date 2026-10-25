# Full write-up: https://xfinlink.com/blog/short-term-reversal-survivorship-bias-python
#
# Does survivorship bias change the short-term reversal effect? S&P 500 members are
# sorted each month from January 2000 to December 2024 into ten equal-weight
# deciles by their return in the previous month, once on the roster in force at
# each month end (point-in-time) and once on today's roster. Each decile is held
# for one month. A holding that stops trading earns its return through its last
# trading day.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# The S&P 500 roster on the last business day of each month, Dec 1999 to Nov 2024,
# sets the universe for the following month
dates = pd.date_range("1999-12-01", "2024-11-30", freq="BME")
pit = {d.to_period("M") + 1: set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))
                                 .entity_id.dropna().astype(int)) for d in dates}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

# Monthly total returns, keyed by entity id so a recycled ticker cannot swap companies
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1999-12-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily")
signal = ret.shift(1)  # last month's return, known at the month end
ret, signal = ret.loc["2000-01":], signal.loc["2000-01":]
print(f"{len(pit)} monthly rosters, {len(ids)} securities, {len(ret)} holding months")

def membership(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

pit_mask, today_mask = membership(lambda m: pit[m]), membership(lambda m: today)
annual = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
results = {}
for name, mask in {"Point-in-time rosters": pit_mask, "Today's roster": today_mask}.items():
    s = signal.where(mask)
    decile = np.ceil(s.rank(axis=1, pct=True) * 10)  # 1 = last month's biggest losers
    d = {k: ret.where(decile == k).mean(axis=1) for k in range(1, 11)}
    print(f"{name}: {s.notna().sum(axis=1).mean():.0f} stocks sorted a month on average, "
          f"universe {annual(ret.where(s.notna()).mean(axis=1)):.2%} a year")
    print("  Deciles 1-10: " + " ".join(f"{annual(d[k]):.1%}" for k in range(1, 11)))
    for k, label in [(1, "Losers"), (10, "Winners")]:
        held = decile == k
        gone = (held & ~today_mask).sum(axis=1).div(held.sum(axis=1)).mean()
        print(f"  {label} (D{k}): last month {s.where(held).stack().mean():+.1%}, "
              f"next month {annual(d[k]):.2%} a year, not in today's index {gone:.1%}")
    spread = d[1] - d[10]
    years = ((1 + d[1]).groupby(d[1].index.year).prod()
             > (1 + d[10]).groupby(d[10].index.year).prod())
    print(f"  Losers minus winners: {annual(d[1]) - annual(d[10]):+.2%} a year "
          f"(2000-2012 {annual(d[1][:'2012']) - annual(d[10][:'2012']):+.2%}, "
          f"2013-2024 {annual(d[1]['2013':]) - annual(d[10]['2013':]):+.2%}); "
          f"t-stat {spread.mean() / spread.std() * len(spread) ** 0.5:.2f}; "
          f"losers ahead in {years.sum()} of {len(years)} years")
    results[name] = d

held = np.ceil(signal.where(pit_mask).rank(axis=1, pct=True) * 10) == 1
print(f"Point-in-time losers still in the index today: {annual(ret.where(held & today_mask).mean(axis=1)):.2%} a year; "
      f"since removed: {annual(ret.where(held & ~today_mask).mean(axis=1)):.2%} a year")

# Chart: annual return by last-month-return decile, point-in-time rosters vs today's roster
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 11)
for offset, (name, color) in zip([-0.2, 0.2], [("Point-in-time rosters", "#3b82f6"),
                                              ("Today's roster", "#f59e0b")]):
    vals = [annual(results[name][k]) * 100 for k in x]
    bars = ax.bar(x + offset, vals, width=0.4, color=color, label=name)
    ax.bar_label(bars, fmt="%.1f", padding=3, color="#e0e0e0", fontsize=8)
ax.set_xticks(x, ["D1\nbiggest\nlosers"] + [f"D{k}" for k in range(2, 10)] + ["D10\nbiggest\nwinners"])
ax.set_xlabel("Decile of last month's return")
ax.set_ylabel("Annual return next month, 2000-2024 (%)")
ax.set_title("Short-Term Reversal: Survivorship Bias Creates a Fake Edge (S&P 500)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("short-term-reversal-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
