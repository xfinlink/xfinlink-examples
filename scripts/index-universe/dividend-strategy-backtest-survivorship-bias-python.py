# Full write-up: https://xfinlink.com/blog/dividend-strategy-backtest-survivorship-bias-python
#
# Does survivorship bias change a high-dividend-yield backtest? S&P 500 members are
# sorted each month from January 2000 to December 2024 by trailing 12-month dividend
# yield: non-payers form their own group and payers are split into five equal-weight
# quintiles. The sort runs once on the roster in force at each month end
# (point-in-time) and once on today's roster. Each group is held for one month.
# A holding that stops trading earns its return through its last trading day.
import matplotlib
matplotlib.use("Agg")
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

# Monthly bars from December 1998, so the first sort has 12 months of dividends behind it
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1998-12-01", end="2024-12-31", interval="1mo",
                          fields=["close", "adj_close", "return_daily", "dividend", "split_ratio"],
                          max_rows=100_000) for i in range(0, len(ids), 25)).reset_index(drop=True)
px["month"] = px["date"].dt.to_period("M")

def restate(rows):
    # Each dividend times adj_close / close on its own row: today's share basis
    factor = rows["adj_close"] / rows["close"]
    special = rows["dividend_type"].fillna("").str.contains("special")
    return pd.DataFrame({"regular": rows["dividend"].where(~special).fillna(0) * factor,
                         "all": rows["dividend"].fillna(0) * factor})

div = restate(px)
# In a month with a split, restate each dividend with the factor of its own ex-date
split = px[(px["split_ratio"].fillna(1) != 1) & px["dividend"].notna()]
for m, g in split.groupby("month"):
    d = xfl.prices(entity_id=g["entity_id"].tolist(), start=str(m.start_time.date()),
                   end=str(m.end_time.date()), fields=["close", "adj_close", "dividend"])
    div.loc[g.index] = restate(d).groupby(d["entity_id"]).sum().reindex(g["entity_id"]).fillna(0).values

ret = px.pivot(index="month", columns="entity_id", values="return_daily").loc["2000-01":]
adj = px.pivot(index="month", columns="entity_id", values="adj_close")
listed = adj.notna().rolling(12).sum() == 12  # a full 12 months of bars at the sort date

def trailing_yield(kind):  # dividends of the last 12 months over the month-end price, known a month ahead
    paid = px.assign(d=div[kind]).pivot(index="month", columns="entity_id", values="d").fillna(0)
    return (paid.rolling(12).sum() / adj).where(listed).shift(1).loc["2000-01":]

print(f"{len(pit)} monthly rosters, {len(ids)} securities, {len(ret)} holding months, "
      f"{len(split)} split months with a dividend restated day by day")

def membership(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

pit_mask, today_mask = membership(lambda m: pit[m]), membership(lambda m: today)
annual = lambda r: (1 + r).prod() ** (12 / len(r)) - 1

def backtest(signal, mask, cap=None):
    s, r = signal.where(mask), ret.clip(upper=cap)
    fifth = np.ceil(s.where(s > 0).rank(axis=1, pct=True) * 5)  # 5 = highest yield
    groups = {"Non-payers": s == 0, **{f"Q{k}": fifth == k for k in range(1, 6)}}
    return s, groups, {k: r.where(g).mean(axis=1) for k, g in groups.items()}, r.where(s.notna()).mean(axis=1)

regular = trailing_yield("regular")
for name, mask in {"Point-in-time rosters": pit_mask, "Today's roster": today_mask}.items():
    s, groups, g, market = backtest(regular, mask)
    print(f"{name}: {s.notna().sum(axis=1).mean():.0f} stocks sorted a month on average "
          f"({(s > 0).sum(axis=1).mean():.0f} payers), all members {annual(market):.2%} a year")
    for k, held in groups.items():
        gone = (held & ~today_mask).sum(axis=1).div(held.sum(axis=1)).mean()
        later = (held & ~pit_mask).sum(axis=1).div(held.sum(axis=1)).mean()
        print(f"  {k}: median yield {s.where(held).stack().median():.2%}, return {annual(g[k]):.2%} a year, "
              f"not in today's index {gone:.1%}, not yet a member {later:.1%}")
    spread = g["Q5"] - market
    print(f"  Q5 minus all members: {annual(g['Q5']) - annual(market):+.2%} a year "
          f"(2000-2012 {annual(g['Q5'][:'2012']) - annual(market[:'2012']):+.2%}, "
          f"2013-2024 {annual(g['Q5']['2013':]) - annual(market['2013':]):+.2%}); "
          f"t-stat {spread.mean() / spread.std() * len(spread) ** 0.5:.2f}; "
          f"Q5 minus non-payers: {annual(g['Q5']) - annual(g['Non-payers']):+.2%} a year")

s, groups, _, _ = backtest(regular, pit_mask)
for k in ["Q5", "Non-payers"]:
    print(f"Point-in-time {k} still in the index today: {annual(ret.where(groups[k] & today_mask).mean(axis=1)):.2%} "
          f"a year; since removed: {annual(ret.where(groups[k] & ~today_mask).mean(axis=1)):.2%} a year")
for label, signal, cap in [("Monthly returns capped at +100%", regular, 1.0),
                           ("Special dividends included", trailing_yield("all"), None)]:
    line = []
    for name, mask in {"point-in-time": pit_mask, "today": today_mask}.items():
        _, _, g, market = backtest(signal, mask, cap)
        line.append(f"{name} Q5 minus all {annual(g['Q5']) - annual(market):+.2%}, "
                    f"minus non-payers {annual(g['Q5']) - annual(g['Non-payers']):+.2%}")
    print(f"{label}: " + "; ".join(line))

# Chart: annual return by yield group, point-in-time rosters vs today's roster
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
labels = ["Non-payers", "Q1", "Q2", "Q3", "Q4", "Q5", "All members"]
x = np.arange(len(labels))
for offset, (name, mask, color) in zip([-0.2, 0.2], [("Point-in-time rosters", pit_mask, "#3b82f6"),
                                                    ("Today's roster", today_mask, "#f59e0b")]):
    _, _, g, market = backtest(regular, mask)
    vals = [annual(g[k] if k in g else market) * 100 for k in labels]
    bars = ax.bar(x + offset, vals, width=0.4, color=color, label=name)
    ax.bar_label(bars, fmt="%.1f", padding=3, color="#e0e0e0", fontsize=8)
ax.set_xticks(x, ["Non-\npayers", "Q1\nlowest\nyield", "Q2", "Q3", "Q4", "Q5\nhighest\nyield", "All\nmembers"])
ax.set_xlabel("Trailing 12-month dividend yield group")
ax.set_ylabel("Annual return, 2000-2024 (%)")
ax.set_title("Dividend Strategy Backtest: Survivorship Bias Flips Ranking (S&P 500)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center")
plt.tight_layout()
plt.savefig("dividend-strategy-backtest-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
