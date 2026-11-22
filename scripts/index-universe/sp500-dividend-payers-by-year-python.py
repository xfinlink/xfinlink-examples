# Full write-up: https://xfinlink.com/blog/sp500-dividend-payers-by-year-python
#
# How many S&P 500 companies paid a dividend each year, 1980-2025, counted on the
# roster in force on the last business day of each December? A company is a payer
# in a year when it paid at least one cash dividend with an ex-date in that calendar
# year (special dividends included). Share classes and tracking stocks of one company
# count once. A lone payment under 2 cents a share, with no other payment in the year
# before or after, is a rights redemption, not a dividend. The same count on today's
# roster, traced back through each member's own payment history (only for companies
# trading that December), shows what a survivor list would report instead.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# S&P 500 rosters on the last business day of each December 1979-2025, and today's
dates = pd.date_range("1979-12-01", "2025-12-31", freq="BYE")
rosters = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in dates}
rosters["today"] = xfl.index("sp500")

def companies(r):
    """One entry per company: index lines that share a company name count once."""
    r = r.dropna(subset=["entity_id"])
    return r.assign(e=r["entity_id"].astype(int)).groupby("entity_name")["e"].apply(frozenset).tolist()

cos = {k: companies(r) for k, r in rosters.items()}
ids = sorted(set().union(*[set(r["entity_id"].dropna().astype(int)) for r in rosters.values()]))

# Monthly bars with each month's cash dividends, 1979-2025
px = pd.concat(xfl.prices(entity_id=ids[i:i + 50], start="1979-01-01", end="2025-12-31", interval="1mo",
                          fields=["close", "dividend"], max_rows=250_000)
               for i in range(0, len(ids), 50)).reset_index(drop=True)
px["year"] = px["date"].dt.year
dec = set(zip(px.loc[px["date"].dt.month == 12, "entity_id"], px.loc[px["date"].dt.month == 12, "year"]))
pay = px[px["dividend"] > 0]
years_paid = set(zip(pay["entity_id"], pay["year"]))
alone = pay.groupby(["entity_id", "year"])["dividend"].transform("size").eq(1) & np.array(
    [(e, y - 1) not in years_paid and (e, y + 1) not in years_paid for e, y in zip(pay["entity_id"], pay["year"])])
rights = pay[alone & (pay["dividend"] < 0.02)]
pay = pay.drop(rights.index)
P = set(zip(pay["entity_id"], pay["year"]))
only_special = pay.groupby(["entity_id", "year"])["dividend_type"].agg(lambda s: set(s) == {"special_dividends"})
SPECIAL = set(only_special[only_special].index)

traded = lambda c, y: any((e, y) in dec for e in c)           # had a December bar that year
paid = lambda c, y: any((e, y) in P for e in c)
def share(group, y):
    g = [c for c in group if traded(c, y)]
    return len(g), sum(paid(c, y) for c in g)

today = cos["today"]
in_today = set().union(*today)
rows = []
for y in range(1980, 2026):
    roster = cos[y]
    on_roster = set().union(*roster)
    stay = [c for c in roster if c & in_today]                 # still in the index today
    gone = [c for c in roster if not c & in_today]             # removed since
    later = [c for c in today if not c & on_roster]            # today's members, not yet in the index
    both = [c for c in roster if c & set().union(*cos[y - 1]) and traded(c, y - 1)]
    n, p = share(roster, y)
    nt, pt = share(today, y)
    row = {"year": y, "companies": n, "payers": p, "share": p / n, "today_n": nt, "today_share": pt / nt}
    for k, g in [("stay", stay), ("gone", gone), ("later", later)]:
        gn, gp = share(g, y)
        row[k + "_n"], row[k] = gn, gp / gn if gn else np.nan
    row["stopped"] = sum(paid(c, y - 1) and not paid(c, y) for c in both)
    row["started"] = sum(paid(c, y) and not paid(c, y - 1) for c in both)
    row["special_only"] = sum(paid(c, y) and all((e, y) not in P or (e, y) in SPECIAL for e in c)
                              for c in roster if traded(c, y))
    rows.append(row)
t = pd.DataFrame(rows).set_index("year")
t["gap"] = t["today_share"] - t["share"]

# ---------------- report ----------------
print(f"{len(dates)} December rosters 1979-2025 plus today's, {len(ids):,} securities; "
      f"{len(rights)} lone sub-2-cent payments set aside as rights redemptions")
print(f"\n{'year':>4}{'members':>9}{'payers':>8}{'share':>8}{'today':>8}{'gap':>7}"
      f"{'still in':>10}{'removed':>9}{'not yet':>9}{'stopped':>9}{'started':>9}")
for y in [1980, 1985, 1990, 1995, 2000, 2001, 2002, 2005, 2008, 2009, 2010, 2015, 2019, 2020, 2021, 2025]:
    r = t.loc[y]
    print(f"{y:>4}{r.companies:9.0f}{r.payers:8.0f}{r.share:8.1%}{r.today_share:8.1%}{r.gap * 100:+7.1f}"
          f"{r.stay:10.1%}{r.gone:9.1%}{r.later:9.1%}{r.stopped:9.0f}{r.started:9.0f}")
low = t["payers"].idxmin()
print(f"\nPayers: {t.loc[1980, 'payers']:.0f} of {t.loc[1980, 'companies']:.0f} in 1980, low {t.loc[low, 'payers']:.0f} "
      f"({t.loc[low, 'share']:.1%}) in {low}, {t.loc[2025, 'payers']:.0f} of {t.loc[2025, 'companies']:.0f} in 2025")
print(f"Today's roster traced back: {t.loc[1980, 'today_n']:.0f} companies trading in 1980 ({t.loc[1980, 'today_share']:.1%} payers), "
      f"low {t['today_share'].min():.1%} in {t['today_share'].idxmin()}")
print(f"Gap (today's roster minus real roster): mean {t['gap'].mean() * 100:+.1f} points, "
      f"widest {t['gap'].min() * 100:+.1f} in {t['gap'].idxmin()}, below zero in {(t['gap'] < 0).sum()} of {len(t)} years")
print(f"Years whose only payments were special dividends: at most {t['special_only'].max():.0f} companies a year")
for y in [1980, 2000]:
    r = t.loc[y]
    print(f"{y}: real roster = {r.stay_n:.0f} still in the index ({r.stay:.1%}) + {r.gone_n:.0f} removed since ({r.gone:.1%}); "
          f"today's roster = {r.stay_n:.0f} still in + {r.later_n:.0f} not yet members ({r.later:.1%})")
ids95, ids02 = set().union(*cos[1995]), set().union(*cos[2002])
joined = [c for c in cos[2002] if not c & ids95 and traded(c, 2002)]
left = [c for c in cos[1995] if not c & ids02 and traded(c, 1995)]
print(f"1995 to 2002: {len(joined)} companies joined ({np.mean([paid(c, 2002) for c in joined]):.1%} paid in 2002), "
      f"{len(left)} left ({np.mean([paid(c, 1995) for c in left]):.1%} paid in 1995)")
s = t.loc[2001:2025]
print(f"2001-2025 among members on consecutive rosters: {s['stopped'].sum():.0f} stopped, {s['started'].sum():.0f} started")

# Chart: share of companies paying a dividend, real rosters vs today's roster traced back
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t.index, t["share"] * 100, color="#3987e5", lw=2, label="Roster in force each December")
ax.plot(t.index, t["today_share"] * 100, color="#d95926", lw=2, label="Today's roster, traced back")
ax.fill_between(t.index, t["share"] * 100, t["today_share"] * 100, color="#3987e5", alpha=0.12, lw=0)
for y, col, dy in [(1980, "share", 8), (low, "share", -16), (2025, "share", 8), (1980, "today_share", -14)]:
    v = t.loc[y, col] * 100
    ax.annotate(f"{v:.1f}%", (y, v), xytext=(0, dy), textcoords="offset points",
                ha="center", color="#e0e0e0", fontsize=9)
ax.set_ylim(55, 100)
ax.set_xlim(1979, 2026)
ax.set_xlabel("Calendar year (roster on the last business day of December)")
ax.set_ylabel("Companies that paid a dividend (%)")
ax.set_title("S&P 500 Dividend Payers by Year, 1980-2025")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.savefig("sp500-dividend-payers-by-year-python.png", dpi=150, facecolor="#0a0a0a")
