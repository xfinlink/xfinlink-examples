# Full write-up: https://xfinlink.com/blog/lookahead-bias-year-end-sp500-roster-python
#
# How much look-ahead bias does a year-end S&P 500 list add to a backtest?
# An equal-weight S&P 500 portfolio rebalanced monthly from January 2000 to
# December 2024, holding either the roster in force at each month end
# (point-in-time), the roster of 31 December of the same year applied to every
# month of that year (look-ahead), or the roster of 1 January applied to every
# month (stale, no look-ahead).
# A holding that stops trading earns its return through its last trading day.
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

def roster(d):
    return set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).entity_id.dropna().astype(int))

# Rosters on the last business day of each month, Dec 1999 to Dec 2024.
# The month-end roster sets the holdings for the following month.
month_ends = pd.date_range("1999-12-01", "2024-12-31", freq="BME")
at = {d.to_period("M"): roster(d) for d in month_ends}
pit = {m + 1: at[m] for m in at if m < pd.Period("2024-12", "M")}
year_end = {y: at[pd.Period(f"{y}-12", "M")] for y in range(2000, 2025)}   # known only on 31 Dec
year_start = {y: at[pd.Period(f"{y - 1}-12", "M")] for y in range(2000, 2025)}

# Monthly total returns for every company any of the rosters holds
ids = sorted(set().union(*at.values()))
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="2000-01-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily").loc["2000-01":"2024-12"]
print(f"{len(at)} month-end rosters, {len(ids)} securities, {len(ret)} holding months")

def mask(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

m_pit = mask(lambda m: pit[m])
m_end = mask(lambda m: year_end[m.year])
m_start = mask(lambda m: year_start[m.year])
books = {"Point-in-time rosters": m_pit,
         "December roster, applied all year": m_end,
         "January roster, applied all year": m_start}

annual = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
yearly = lambda r: (1 + r).groupby(r.index.year).prod() - 1
res = {name: ret.where(mk).mean(axis=1) for name, mk in books.items()}
for name, r in res.items():
    print(f"{name}: {annual(r):.2%} a year, $1 grows to ${(1 + r).prod():.2f}")

gap = yearly(res["December roster, applied all year"]) - yearly(res["Point-in-time rosters"])
stale = annual(res["January roster, applied all year"]) - annual(res["Point-in-time rosters"])
print(f"December roster beats point-in-time by "
      f"{annual(res['December roster, applied all year']) - annual(res['Point-in-time rosters']):.2%} "
      f"a year, ahead in {(gap > 0).sum()} of {len(gap)} years; January roster: {stale:+.2%} a year")

# Where the lift comes from: the two groups of holdings that differ
early = m_end & ~m_pit       # held before the month they joined the index
dropped = m_pit & ~m_end     # members the December list leaves out because they leave later that year
mid = ret.where(m_pit & m_end).mean(axis=1)
print(f"Average holdings differing per month: {early.sum(axis=1).mean():.1f} early additions, "
      f"{dropped.sum(axis=1).mean():.1f} members removed later that year")
print(f"Removing the later leavers alone: {annual(mid) - annual(res['Point-in-time rosters']):+.2%} a year; "
      f"adding the early arrivals: {annual(res['December roster, applied all year']) - annual(mid):+.2%} a year")
base = ret.where(m_pit).stack().dropna()
for label, g in [("Early additions", early), ("Members removed later that year", dropped)]:
    r = ret.where(g).stack().dropna()
    print(f"{label}: {len(r)} stock-months, average month {r.mean():.2%}, median {r.median():.2%} "
          f"(point-in-time holdings: {base.mean():.2%}, median {base.median():.2%})")

# Chart: the yearly return gap (December roster minus point-in-time)
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(gap.index, gap * 100, color=["#3b82f6" if g > 0 else "#f59e0b" for g in gap])
ax.axhline(0, color="#e0e0e0", lw=0.6)
ax.set_title("Look-Ahead Bias by Year: December S&P 500 Roster Applied All Year")
ax.set_ylabel("December roster minus point-in-time\n(percentage points of return)")
ax.set_xlabel("Calendar year")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("lookahead-bias-year-end-sp500-roster-python.png", dpi=150, facecolor="#0a0a0a")
