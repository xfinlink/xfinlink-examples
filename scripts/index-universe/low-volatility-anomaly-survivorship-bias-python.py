# Full write-up: https://xfinlink.com/blog/low-volatility-anomaly-survivorship-bias-python
#
# Does survivorship bias change the low-volatility anomaly? S&P 500 members are
# sorted each month from January 2000 to December 2024 into five equal-weight
# quintiles by their volatility over the previous 36 months, once on the roster
# in force at each month end (point-in-time) and once on today's roster.
# A holding that stops trading earns its return through its last trading day.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# The S&P 500 roster on the last business day of each month, Dec 1999 to Nov 2024,
# sets the holdings for the following month
dates = pd.date_range("1999-12-01", "2024-11-30", freq="BME")
pit = {d.to_period("M") + 1: set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))
                                 .entity_id.dropna().astype(int)) for d in dates}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

# Monthly total returns from December 1996, so every month has 36 months of history behind it
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1996-12-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily")
# Volatility over the 36 months to the prior month end (at least 24 returns), annualised
vol = (ret.rolling(36, min_periods=24).std() * 12 ** 0.5).shift(1)
ret, vol = ret.loc["2000-01":], vol.loc["2000-01":]
print(f"{len(pit)} monthly rosters, {len(ids)} securities, {len(ret)} holding months")

def membership(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

pit_mask, today_mask = membership(lambda m: pit[m]), membership(lambda m: today)
annual = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
results = {}
for name, mask in {"Point-in-time rosters": pit_mask, "Today's roster": today_mask}.items():
    v = vol.where(mask)
    quintile = np.ceil(v.rank(axis=1, pct=True) * 5)  # 1 = lowest volatility
    print(f"{name}: {v.notna().sum(axis=1).mean():.0f} stocks sorted a month on average")
    q = {}
    for k in range(1, 6):
        held = quintile == k
        q[k] = ret.where(held).mean(axis=1)
        risk = q[k].std() * 12 ** 0.5
        removed = (held & ~today_mask).sum(axis=1).div(held.sum(axis=1)).mean()
        print(f"  Q{k}: past volatility {v.where(held).stack().mean():.1%}, "
              f"return {annual(q[k]):.2%} a year, volatility {risk:.2%}, "
              f"return per unit of volatility {annual(q[k]) / risk:.2f}, "
              f"not in today's index {removed:.1%}")
    yearly = (1 + q[1]).groupby(q[1].index.year).prod() - (1 + q[5]).groupby(q[5].index.year).prod()
    print(f"  Low minus high volatility: {annual(q[1]) - annual(q[5]):+.2%} a year "
          f"(2000-2012 {annual(q[1][:'2012']) - annual(q[5][:'2012']):+.2%}, "
          f"2013-2024 {annual(q[1]['2013':]) - annual(q[5]['2013':]):+.2%}); "
          f"low volatility ahead in {(yearly > 0).sum()} of {len(yearly)} years")
    results[name] = q

# Chart: annual return by volatility quintile, point-in-time rosters vs today's roster
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 6)
for offset, (name, color) in zip([-0.2, 0.2], [("Point-in-time rosters", "#3b82f6"),
                                              ("Today's roster", "#f59e0b")]):
    vals = [annual(results[name][k]) * 100 for k in x]
    bars = ax.bar(x + offset, vals, width=0.4, color=color, label=name)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#e0e0e0", fontsize=9)
ax.set_xticks(x, ["Q1\nlowest volatility", "Q2", "Q3", "Q4", "Q5\nhighest volatility"])
ax.set_ylabel("Annual return, 2000-2024 (%)")
ax.set_title("Low-Volatility Anomaly: Survivorship Bias Reverses It (S&P 500, 2000-2024)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("low-volatility-anomaly-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
