# Full write-up: https://xfinlink.com/blog/sp500-roster-refresh-frequency-python
import warnings

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "1990-01-01", "2025-12-31"
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose


def all_events():
    """Every S&P 500 addition and removal on record, paged 1,000 rows at a time."""
    pages = [xfl.index_events("sp500", start="1925-12-31")]
    while len(pages[-1]) == 1000:
        pages.append(xfl.index_events("sp500", start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True)


def third_friday(year, month):
    d = pd.Timestamp(year, month, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


ev = all_events()
ev["key"] = np.where(ev["entity_id"].notna(), ev["entity_id"].astype("Int64").astype(str), "ticker " + ev["ticker"].fillna(""))
ev["date"] = pd.to_datetime(ev["effective_date"])

# 1. Membership spells: an addition opens a spell on its first day, a removal closes it on its last day
spells = []
for key, g in ev.sort_values(["date", "event_type"]).groupby("key"):
    opened = pd.Timestamp("1900-01-01")  # a removal with no addition on record: in since before 1925
    for _, row in g.iterrows():
        if row["event_type"] == "added":
            opened = row["date"]
        else:
            spells.append((key, opened, row["date"]))
            opened = None
    if opened is not None:
        spells.append((key, opened, pd.Timestamp("2099-12-31")))
spells = pd.DataFrame(spells, columns=["key", "first_day", "last_day"])

# 2. The true roster on every weekday, as a weekday x member matrix
days = pd.bdate_range(START, END)
keys = {k: i for i, k in enumerate(spells["key"].unique())}
member = np.zeros((len(days), len(keys)), dtype=bool)
for key, first, last in spells.itertuples(index=False):
    member[:, keys[key]] |= (days >= first) & (days <= last)

# 3. Refresh schedules: the roster is re-pulled on each date and held until the next one
years = range(int(START[:4]), int(END[:4]) + 1)
schedules = {
    "Yearly (1 January)": [pd.Timestamp(y, 1, 1) for y in years],
    "Twice a year": [pd.Timestamp(y, m, 1) for y in years for m in (1, 7)],
    "Quarterly (1st of Jan/Apr/Jul/Oct)": [pd.Timestamp(y, m, 1) for y in years for m in (1, 4, 7, 10)],
    "Quarterly (rebalance Monday)": [third_friday(y, m) + pd.Timedelta(days=3) for y in years for m in (3, 6, 9, 12)],
    "Monthly (1st)": [pd.Timestamp(y, m, 1) for y in years for m in range(1, 13)],
}
decade = np.where(days.year >= 2020, "2020-25", (days.year // 10 * 10).astype(str) + "s")
rows, by_year = [], {}
for name, dates in schedules.items():
    pulled = np.searchsorted(days, pd.DatetimeIndex(dates))  # first weekday on or after each refresh date
    held = pulled[np.searchsorted(pulled, np.arange(len(days)), side="right") - 1]
    ok = days >= days[pulled[0]]
    stale = (member[held] & ~member).sum(axis=1)   # still held, already removed
    missed = (~member[held] & member).sum(axis=1)  # already added, not yet held
    wrong = pd.Series(stale + missed, index=days)[ok]
    by_year[name] = wrong.groupby(wrong.index.year).mean()
    worst = wrong.idxmax()
    rows.append({"schedule": name, **wrong.groupby(decade[ok]).mean().round(1).to_dict(),
                 "all": round(wrong.mean(), 1), "share": f"{wrong.mean() / member[ok].sum(axis=1).mean():.1%}",
                 "worst": int(wrong.max()), "worst_day": worst.date()})
print("Names wrong on an average weekday (held after removal + missing after addition)")
print(pd.DataFrame(rows).set_index("schedule").to_string())

# 4. How many changes land in the rebalance window (third Friday of Mar/Jun/Sep/Dec to the Monday after)
ch = ev[(ev["date"] >= "1980-01-01") & (ev["date"] <= END)].copy()
tf = [third_friday(d.year, d.month) for d in ch["date"]]
ch["on_rebalance"] = [(d.month % 3 == 0) and (t <= d <= t + pd.Timedelta(days=3)) for d, t in zip(ch["date"], tf)]
first = ch["date"].dt.year // 5 * 5
last = (first + 4).clip(upper=int(END[:4]))
ch["period"] = np.where(first == last, first.astype(str), first.astype(str) + "-" + last.astype(str).str[-2:])
summary = ch.groupby("period").agg(events=("on_rebalance", "size"), in_window=("on_rebalance", "sum"))
summary["per_year"] = (summary["events"] / [min(5, int(END[:4]) - int(p[:4]) + 1) for p in summary.index]).round(1)
summary["in_window_share"] = (summary["in_window"] / summary["events"]).map("{:.0%}".format)
print("\nAdditions and removals by period, and the share in the rebalance window")
print(summary.to_string())

# 5. Chart: average names wrong by year for four schedules
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444"})
fig, ax = plt.subplots(figsize=(10, 5))
styles = {"Yearly (1 January)": ("#3b82f6", "-"), "Quarterly (1st of Jan/Apr/Jul/Oct)": ("#f59e0b", "-"),
          "Quarterly (rebalance Monday)": ("#a3a3a3", "--"), "Monthly (1st)": ("#22c55e", "-")}
for name, (color, ls) in styles.items():
    s = by_year[name]
    ax.plot(s.index, s.values, color=color, linestyle=ls, linewidth=2, label=name)
ax.set_title("How Often Does the S&P 500 Change? Stale Rosters by Refresh Schedule", fontsize=12)
ax.set_xlabel("Year")
ax.set_ylabel("Names wrong on an average weekday")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
plt.tight_layout()
plt.savefig("sp500-roster-refresh-frequency-python.png", dpi=150)
