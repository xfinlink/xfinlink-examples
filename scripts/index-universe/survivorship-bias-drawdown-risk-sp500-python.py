# Full write-up: https://xfinlink.com/blog/survivorship-bias-drawdown-risk-sp500-python
#
# Does survivorship bias understate risk? An equal-weight S&P 500 portfolio,
# rebalanced monthly from January 2000 to December 2024, built two ways: from the
# roster in force at each month end (point-in-time) and from today's roster.
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

# Monthly total returns, keyed by entity id, for every company either book holds
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1999-12-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily").loc["2000-01":]
print(f"{len(pit)} monthly rosters, {len(ids)} securities, {len(ret)} holding months")

def membership(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

masks = {"Point-in-time rosters": membership(lambda m: pit[m]),
         "Today's roster": membership(lambda m: today)}

curves = {}
for name, mask in masks.items():
    held = ret.where(mask)
    r = held.mean(axis=1)  # equal weight across the members with a return that month
    wealth = pd.concat([pd.Series([1.0], index=[ret.index[0] - 1]), (1 + r).cumprod()])
    dd = wealth / wealth.cummax() - 1
    curves[name] = dd
    trough = dd.idxmin()
    peak = wealth[:trough].idxmax()
    back = wealth[trough:][wealth[trough:] >= wealth[peak]].index[0]
    n = held.notna().sum(axis=1)
    print(f"{name}: {n.iloc[0]} names in Jan 2000, {n.iloc[-1]} in Dec 2024")
    print(f"  return {wealth.iloc[-1] ** (12 / len(r)) - 1:.2%} a year, volatility {r.std() * 12 ** 0.5:.2%}, "
          f"Sharpe {r.mean() * 12 / (r.std() * 12 ** 0.5):.2f} (risk-free rate set to zero)")
    print(f"  max drawdown {dd.min():.1%}: peak {peak}, trough {trough}, back to the peak {back}")
    print(f"  deepest drawdown 2000-2002 {dd['2000-01':'2002-12'].min():.1%}, "
          f"2007-2009 {dd['2007-01':'2009-12'].min():.1%}; worst month {r.min():.1%} ({r.idxmin()})")
    # Each company's own path over the months this book held it
    path = {c: np.r_[1.0, (1 + held[c].dropna()).cumprod()] for c in held.columns[held.notna().any()]}
    worst = pd.Series({c: (w / np.maximum.accumulate(w)).min() - 1 for c, w in path.items()})
    end = pd.Series({c: w[-1] / w.max() - 1 for c, w in path.items()})
    print(f"  companies held {len(path)}: fell 80%+ from a peak while held {(worst <= -0.8).mean():.1%}, "
          f"still 80%+ below it in the last month held {(end <= -0.8).mean():.1%} ({(end <= -0.8).sum()})")

# Chart: drawdown from the running peak, both books
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"Point-in-time rosters": "#3b82f6", "Today's roster": "#f59e0b"}
for name, dd in curves.items():
    ax.plot(dd.index.to_timestamp(how="end"), dd.values * 100, color=colors[name], lw=1.8,
            label=f"{name} (worst {dd.min():.1%})")
ax.axhline(0, color="#e0e0e0", lw=0.6)
ax.set_title("Survivorship Bias and S&P 500 Drawdowns, 2000-2024 (equal weight, monthly)")
ax.set_ylabel("Drawdown from previous peak (%)")
ax.set_xlabel("Month end")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.savefig("survivorship-bias-drawdown-risk-sp500-python.png", dpi=150, facecolor="#0a0a0a")
