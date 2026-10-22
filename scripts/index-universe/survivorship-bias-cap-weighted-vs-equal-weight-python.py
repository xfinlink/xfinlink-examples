# Full write-up: https://xfinlink.com/blog/survivorship-bias-cap-weighted-vs-equal-weight-python
#
# Does cap-weighting shrink survivorship bias? An S&P 500 portfolio rebalanced
# monthly from January 2000 to December 2024, built from the roster in force at
# each month end (point-in-time) or from today's roster, and weighted either
# equally or by each holding's market value at the previous month end.
# A holding that stops trading earns its return through its last trading day.
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

# Monthly total returns and month-end market values for every company either roster holds
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1999-11-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily", "market_cap"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily")
cap = px.pivot(index="month", columns="entity_id", values="market_cap").shift(1)  # prior month end
ret, cap = ret.loc["2000-01":], cap.loc["2000-01":]
print(f"{len(pit)} monthly rosters, {len(ids)} securities, {len(ret)} holding months")

def membership(members):
    return pd.DataFrame([ret.columns.isin(list(members(m))) for m in ret.index],
                        index=ret.index, columns=ret.columns)

pit_mask, today_mask = membership(lambda m: pit[m]), membership(lambda m: today)
books = {"Point-in-time rosters": pit_mask,
         "Members then, still in today's index": pit_mask & today_mask,
         "Today's roster": today_mask}

def book_returns(mask):
    held = ret.where(mask)
    w = cap.where(mask & held.notna())
    return {"equal weight": held.mean(axis=1),
            "cap weight": (held * w).sum(axis=1) / w.sum(axis=1)}, w

annual = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
yearly = lambda r: (1 + r).groupby(r.index.year).prod() - 1
res, weights = {}, {}
for name, mask in books.items():
    res[name], weights[name] = book_returns(mask)
    for scheme, r in res[name].items():
        print(f"{name}, {scheme}: {annual(r):.2%} a year, volatility {r.std() * 12 ** 0.5:.2%}, "
              f"$1 grows to ${(1 + r).prod():.2f}")

gap = {}
for scheme in ["equal weight", "cap weight"]:
    pit_r, mid_r, today_r = (res[n][scheme] for n in books)
    gap[scheme] = yearly(today_r) - yearly(pit_r)
    print(f"{scheme}: today's roster beats point-in-time by {annual(today_r) - annual(pit_r):.2%} a year "
          f"({annual(mid_r) - annual(pit_r):.2%} from dropping later-removed members, "
          f"{annual(today_r) - annual(mid_r):.2%} from adding names not yet in the index); "
          f"ahead in {(gap[scheme] > 0).sum()} of {len(gap[scheme])} years")

# How much weight each biased group carries, averaged over the 300 months
def share(name, group):
    held = ret.where(books[name]).notna()
    w = weights[name]
    return (held & group).sum(axis=1).div(held.sum(axis=1)).mean(), \
        w.where(group).sum(axis=1).div(w.sum(axis=1)).mean()
ew, cw = share("Point-in-time rosters", ~today_mask)
print(f"Point-in-time book, members later removed: {ew:.1%} of holdings equal-weighted, {cw:.1%} cap-weighted")
ew, cw = share("Today's roster", ~pit_mask)
print(f"Today's-roster book, names not yet in the index: {ew:.1%} of holdings equal-weighted, {cw:.1%} cap-weighted")
w = weights["Point-in-time rosters"].iloc[0].dropna()
w = w / w.sum()
print(f"January 2000, point-in-time book: equal weight {1 / len(w):.2%} a name; cap weights "
      f"{w.min():.3%} to {w.max():.2%}, median {w.median():.3%}")

# Chart: the yearly return gap (today's roster minus point-in-time), both weightings
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
years = gap["equal weight"].index
ax.bar(years - 0.2, gap["equal weight"] * 100, width=0.4, color="#f59e0b",
       label="Equal weight")
ax.bar(years + 0.2, gap["cap weight"] * 100, width=0.4, color="#3b82f6",
       label="Cap weight")
ax.axhline(0, color="#e0e0e0", lw=0.6)
ax.set_title("Survivorship Bias by Year: Cap-Weighted vs Equal-Weight S&P 500 Backtests")
ax.set_ylabel("Today's roster minus point-in-time\n(percentage points of return)")
ax.set_xlabel("Calendar year")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("survivorship-bias-cap-weighted-vs-equal-weight-python.png", dpi=150, facecolor="#0a0a0a")
