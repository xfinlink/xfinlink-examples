# Full write-up: https://xfinlink.com/blog/sp500-members-1957-originals-python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LAUNCH = "1957-03-04"  # first day of the 500-stock index


def all_events():
    """Every S&P 500 addition and removal on record, paged 1,000 rows at a time."""
    pages = [xfl.index_events("sp500", start="1925-12-31")]
    while len(pages[-1]) == 1000:
        pages.append(xfl.index_events("sp500", start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True)


def listings(df):
    """One key per listed security: its entity_id, or its ticker where no id is linked."""
    linked = set(df["entity_id"].dropna().astype(int))
    return linked | set("ticker " + df.loc[df["entity_id"].isna(), "ticker"])


ev = all_events()
opening = xfl.index("sp500", as_of="1925-12-31")  # the record's opening roster has no addition rows
launch = xfl.index("sp500", as_of=LAUNCH)
today = xfl.index("sp500")

# 1. Every listing that has ever been a member, and every one that was a member on or after the launch
ever = listings(ev) | listings(opening)
since_launch = listings(ev[ev["effective_date"] >= LAUNCH]) | listings(launch)

# 2. The launch-day roster over time; removed_date ends the membership that was open on LAUNCH
launch["last_day"] = launch["removed_date"].fillna("9999-12-31")
unbroken = pd.Series({y: int((launch["last_day"] >= f"{y}-03-04").sum()) for y in range(1957, 2027)})

# 3. Launch-day listings in today's roster: unbroken since 1957, or removed and later re-added
now = launch.merge(today[["entity_id", "ticker", "entity_name", "added_date"]],
                   on="entity_id", suffixes=("_1957", "_now"))
now["unbroken"] = now["added_date_now"] <= LAUNCH
stayed = now[now["unbroken"]]

# 4. The same counts on the dates of two published figures (S&P, 2 March 2007; Reuters, 10 April 2013)
checks = {}
for day in ("2007-03-02", "2013-04-10"):
    on_day = set(xfl.index("sp500", as_of=day)["entity_id"].dropna().astype(int))
    checks[day] = (int((launch["last_day"] >= day).sum()), len(on_day & set(launch["entity_id"])))

# 5. Today's roster by the decade its current membership began
today["decade"] = today["added_date"].str[:3] + "0s"

# ---- printed output ----
print(f"S&P 500 event log: {len(ev):,} rows, {ev['effective_date'].min()} to {ev['effective_date'].max()}")
print(f"Listings ever in the index (record opens 1925-12-31): {len(ever):,}")
print(f"Listings in the index on or after {LAUNCH}:          {len(since_launch):,}")
print(f"Members today: {len(today)} ({len(today) / len(ever):.1%} of all listings ever)")
print()
print(f"Launch-day roster ({LAUNCH}): {len(launch)} listings")
print(f"  in today's roster:                  {len(now)}")
print(f"  of which never left:                {len(stayed)}")
print(f"  removed and later re-added:         {len(now) - len(stayed)}")
print(f"  never-left listings on a new ticker: {(stayed['ticker_1957'] != stayed['ticker_now']).sum()}")
print()
print("Launch-day listings still in without a break, on 4 March of:")
for y in (1967, 1977, 1987, 1997, 2007, 2017, 2026):
    print(f"  {y}: {unbroken[y]:>3}")
print(f"  today: {len(stayed):>3}")
recent = launch[launch["removed_date"].between("2026-03-04", "9999-12-30")]
print("  left since 4 March 2026:", ", ".join(f"{r.ticker} ({r.entity_name}, last day {r.removed_date})"
                                              for r in recent.itertuples()))
half = unbroken[unbroken < len(launch) / 2].index.min()
print(f"  first 4 March below half the roster: {half}")
big_day = launch["removed_date"].value_counts().idxmax()
next_day = (pd.Timestamp(big_day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
print(f"  largest one-day exit: {(launch['removed_date'] == big_day).sum()} on {big_day} "
      f"({((ev['event_type'] == 'removed') & (ev['effective_date'] == big_day)).sum()} removals that day, "
      f"{((ev['event_type'] == 'added') & (ev['effective_date'] == next_day)).sum()} additions on {next_day})")
print()
for day, (unbroken_n, any_n) in checks.items():
    print(f"On {day}: never left {unbroken_n}, in the index (re-additions counted) {any_n}")
print()
print("Today's members by decade their current membership began:")
print(today["decade"].value_counts().sort_index().to_string())
print()
print("Never left since 1957, ticker then -> now (earliest current membership first):")
show = stayed.sort_values(["added_date_now", "ticker_now"])
for r in show.itertuples():
    moved = "" if r.ticker_1957 == r.ticker_now else "  (new ticker)"
    print(f"  {r.ticker_1957:<4} -> {r.ticker_now:<5} {r.entity_name_now:<34} since {r.added_date_now}{moved}")
print()
print("Re-added after leaving:")
for r in now[~now["unbroken"]].sort_values("added_date_now").itertuples():
    print(f"  {r.ticker_1957:<4} -> {r.ticker_now:<5} {r.entity_name_now:<34} back since {r.added_date_now}")

# ---- chart: launch-day listings still in the index without a break ----
import matplotlib.pyplot as plt

bg, ink, blue, amber = "#0a0a0a", "#e0e0e0", "#3b82f6", "#d97706"
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.tick_params(colors=ink, labelsize=9)
for s in ax.spines.values():
    s.set_visible(False)
today_x = pd.Timestamp.today().year + (pd.Timestamp.today().dayofyear - 63) / 365.25  # years after 4 March
line = pd.concat([unbroken, pd.Series({today_x: len(stayed)})])
ax.plot(line.index, line.values, color=blue, lw=2.2,
        label="Launch-day listings never removed (same listed security)")
published = {2007: 86, 2013: 69}  # S&P 2007 statement; Reuters 2013; neither states its unit
ax.scatter(list(published), list(published.values()), color=amber, zorder=3, s=36,
           label="Published original-company counts (S&P 2007, Reuters 2013)")
for y, v in published.items():
    ax.annotate(str(v), (y, v), textcoords="offset points", xytext=(0, 8), ha="center", color=amber, fontsize=9)
ax.annotate(f"{len(stayed)} today", (today_x, len(stayed)), textcoords="offset points",
            xytext=(-6, 10), ha="right", color=ink, fontsize=9)
ax.set_ylim(0, 520)
ax.set_xlim(1955, 2028)
ax.set_ylabel("Listings still in the index", color=ink)
ax.set_xlabel("4 March of each year", color=ink)
ax.set_title(f"S&P 500 Members: {len(stayed)} of the 1957 Launch-Day Roster Never Left", color=ink, fontsize=11)
ax.legend(frameon=False, labelcolor=ink, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("sp500-members-1957-originals-python.png", dpi=150, facecolor=bg)
