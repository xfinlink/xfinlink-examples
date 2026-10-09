# Full write-up: https://xfinlink.com/blog/largest-sp500-companies-top-10-history-python
import xfinlink as xfl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = range(1980, 2027)
# Companies are valued on all of their listed common share classes.
# Berkshire Hathaway's Class A shares trade but are not an index line, so they are added.
EXTRA_CLASSES = {10006133: 10025221}  # BRK-A -> BRK-B
# Pairs of share classes that are both index lines, mapped to one company id
SAME_COMPANY = {
    10003590: 10030636,  # GOOG -> GOOGL (Alphabet)
    10032892: 10031087,  # Discovery
    10006698: 10006697,  # Fox Corp
    10003102: 10003101,  # News Corp
    10030753: 10030752,  # Twenty-First Century Fox
    10031256: 10004763,  # Under Armour
}
CLASS_OF = {**EXTRA_CLASSES, **SAME_COMPANY}
# The two Dutch members S&P removed after the close on 2002-07-19
# (Royal Dutch Petroleum, Unilever N.V.) are left out of the ranking.
LEFT_OUT = {10012474, 10013712}

# Short display names for the chart (era names in brackets)
NAMES = {
    10000314: "AT&T Corp. (pre-2005)", 10000473: "Atlantic Richfield", 10001434: "Exxon Mobil",
    10001611: "General Electric", 10001962: "IBM", 10003358: "Schlumberger (SLB)",
    10001629: "General Motors (pre-2009)", 10003589: "Chevron", 10007583: "Amoco",
    10038087: "Mobil", 10003432: "Shell Oil", 10001359: "Eastman Kodak",
    10003399: "Sears, Roebuck", 10001315: "E. I. du Pont", 10017903: "BellSouth",
    10010234: "Merck", 10003045: "Altria (Philip Morris)", 10012940: "Ford Motor",
    10003309: "RJR Nabisco", 10007442: "Bristol-Myers Squibb", 10016650: "Walmart",
    10001013: "Coca-Cola", 10006466: "Procter & Gamble", 10009646: "Johnson & Johnson",
    10003006: "PepsiCo", 10000086: "Microsoft", 10017053: "Intel", 10009498: "Pfizer",
    10025118: "Lucent", 10019672: "Cisco", 10018494: "Citigroup",
    10020848: "America Online / Time Warner", 10018030: "AIG", 10017063: "Bank of America",
    10017933: "AT&T Inc.", 10003635: "Apple", 10014673: "Wells Fargo",
    10015682: "JPMorgan Chase", 10025221: "Berkshire Hathaway", 10030636: "Alphabet",
    10026356: "Amazon", 10002684: "Meta", 10032737: "Visa", 10033507: "Tesla",
    10032777: "UnitedHealth", 10027758: "Nvidia", 10016035: "Eli Lilly", 10033094: "Broadcom",
}

# first trading day of each January, read off IBM's daily bars
ibm = xfl.prices(entity_id=10001962, start="1980-01-01", end="2026-01-31", fields=["close"])
ibm["date"] = pd.to_datetime(ibm["date"])
jan = ibm[ibm["date"].dt.month == 1]
first_days = jan.groupby(jan["date"].dt.year)["date"].min().dt.strftime("%Y-%m-%d")

rows = []
for year in YEARS:
    day = first_days[year]
    roster = xfl.index("sp500", as_of=day)          # survivorship-free roster on that day
    roster = roster.dropna(subset=["entity_id"]).copy()
    roster["entity_id"] = roster["entity_id"].astype(int)
    roster = roster[~roster["entity_id"].isin(LEFT_OUT)]
    ids = roster["entity_id"].tolist()
    ids += [a for a, b in EXTRA_CLASSES.items() if b in ids]
    px = xfl.prices(entity_id=ids, start=day, end=day, fields=["close", "market_cap"])
    px["company"] = px["entity_id"].replace(CLASS_OF)
    mv = px.groupby("company")["market_cap"].sum(min_count=1)
    # one company, one rank: its share classes are summed under one id
    snap = roster.assign(company=roster["entity_id"].replace(CLASS_OF)).sort_values("added_date")
    co = (snap.groupby("company", as_index=False)
              .agg(ticker=("ticker", "first"), entity_name=("entity_name", "first"))
              .rename(columns={"company": "entity_id"}))
    co["market_cap"] = co["entity_id"].map(mv)
    co = co.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
    co["rank"] = range(1, len(co) + 1)
    co["year"], co["date"] = year, day
    co["members"], co["priced"] = len(roster), int(roster["entity_id"].replace(CLASS_OF).map(mv).notna().sum())
    co["index_cap"] = co["market_cap"].sum()
    rows.append(co.head(10))

top = pd.concat(rows, ignore_index=True)
top["share"] = top["market_cap"] / top["index_cap"]

# where is each top-10 company today? the roster without as_of is today's
today = xfl.index("sp500")
members_now = set(today["entity_id"].dropna().astype(int))
by_co = top.sort_values("year").groupby("entity_id")
companies = pd.DataFrame({
    "first_top10": by_co["year"].min(), "last_top10": by_co["year"].max(),
    "years_top10": by_co["year"].nunique(),
})
companies["name"] = companies.index.map(NAMES)
companies["in_index_today"] = companies.index.isin(members_now)

# last day in the index for those that left (from the membership record)
exits = {}
for eid, r in companies[~companies["in_index_today"]].iterrows():
    roster = xfl.index("sp500", as_of=first_days[r["last_top10"]])
    exits[eid] = roster.loc[roster["entity_id"] == eid, "removed_date"].iloc[0]
companies["removed_date"] = companies.index.map(exits)

# persistence: of each January's top 10, how many are still top 10 / still members k years on
sets = top.groupby("year")["entity_id"].apply(set)
def persistence(k):
    out = []
    for y in sets.index:
        if y + k in sets.index:
            later = xfl.index("sp500", as_of=first_days[y + k])
            later_ids = set(later["entity_id"].dropna().astype(int))
            out.append((y, len(sets[y] & sets[y + k]), len(sets[y] & later_ids)))
    return pd.DataFrame(out, columns=["year", "still_top10", "still_member"])

p10, p20 = persistence(10), persistence(20)

# ---- output ----
cov = top.groupby("year").first()
print(f"{len(cov)} Januaries {cov.index.min()}-{cov.index.max()}; members ranked per roster "
      f"{cov['members'].min()}-{cov['members'].max()}, with a market value on the day "
      f"{cov['priced'].min()}-{cov['priced'].max()}")

for y in [1980, 1990, 2000, 2010, 2020, 2026]:
    t = top[top["year"] == y]
    alive = t["entity_id"].isin(members_now).sum()
    print(f"\nTop 10 on {t['date'].iloc[0]}  (top 10's share of ranked value {t['share'].sum():.1%}; "
          f"S&P 500 members today: {alive} of 10)")
    for _, r in t.iterrows():
        print(f"  {r['rank']:>2}. {r['ticker']:<6} {NAMES[r['entity_id']]:<28} ${r['market_cap']/1e9:>8,.1f}bn")

gone = companies[~companies["in_index_today"]].sort_values("first_top10")
print(f"\nCompanies in a January top 10, 1980-2026: {len(companies)}")
print(f"  still S&P 500 members today:  {companies['in_index_today'].sum()}")
print(f"  no longer in the index:       {len(gone)}")
print(f"  median Januaries in top 10:   {companies['years_top10'].median():.0f}")
print("\nTop-10 companies no longer in the index:")
for eid, r in gone.iterrows():
    print(f"  {r['name']:<28} top 10 {r['first_top10']}-{r['last_top10']}, "
          f"{r['years_top10']:>2} of {len(cov)} Januaries; last day in index {r['removed_date']}")

print("\nMost Januaries in the top 10:")
for eid, r in companies.sort_values(["years_top10", "first_top10"], ascending=[False, True]).head(6).iterrows():
    print(f"  {r['name']:<28} {r['years_top10']} of {len(cov)}")

for k, p in [(10, p10), (20, p20)]:
    print(f"\nEach January's top 10, {k} years later ({len(p)} starting years, {p['year'].min()}-{p['year'].max()}):")
    print(f"  still in the top 10:  mean {p['still_top10'].mean():.1f} of 10 (range {p['still_top10'].min()}-{p['still_top10'].max()})")
    print(f"  still in the S&P 500: mean {p['still_member'].mean():.1f} of 10 (range {p['still_member'].min()}-{p['still_member'].max()})")

# ---- chart: one row per company, a filled cell for each January in the top 10 ----
order = companies.sort_values(["first_top10", "years_top10"], ascending=[True, False])
fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
for i, (eid, r) in enumerate(order.iterrows()):
    yrs = top.loc[top["entity_id"] == eid, "year"]
    colour = "#3b82f6" if r["in_index_today"] else "#f59e0b"
    ax.barh([i] * len(yrs), [1] * len(yrs), left=yrs - 0.5, height=0.72, color=colour)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order["name"], fontsize=7, color="#e0e0e0")
ax.invert_yaxis()
ax.set_xlim(1979.5, 2026.5)
ax.set_xticks(range(1980, 2027, 5))
ax.tick_params(axis="x", colors="#e0e0e0", labelsize=8)
ax.tick_params(axis="y", length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("January of year", color="#e0e0e0")
ax.set_title("Largest S&P 500 Companies: Every January Top 10, 1980-2026", color="#e0e0e0", fontsize=11)
ax.legend(handles=[Patch(color="#3b82f6", label="Still an S&P 500 member today"),
                   Patch(color="#f59e0b", label="No longer in the index")],
          loc="lower left", frameon=False, labelcolor="#e0e0e0", fontsize=8)
plt.tight_layout()
plt.savefig("largest-sp500-companies-top-10-history-python.png", dpi=150, facecolor="#0a0a0a")
