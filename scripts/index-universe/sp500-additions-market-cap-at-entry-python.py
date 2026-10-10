# Full write-up: https://xfinlink.com/blog/sp500-additions-market-cap-at-entry-python
import xfinlink as xfl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Second share classes -> the company's first index line; BRK-A trades but is not an index line
CLASS_OF = {10003590: 10030636, 10032892: 10031087, 10006698: 10006697, 10003102: 10003101,
            10030753: 10030752, 10004763: 10031256, 10029970: 10029932, 10006133: 10025221}

def events(kind, start, end=None):
    pages = [xfl.index_events("sp500", event_type=kind, start=start, end=end)]
    while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
        pages.append(xfl.index_events("sp500", event_type=kind, start=start, end=end,
                                      offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True)

ev = events("added", "1990-01-01", "2025-12-31").astype({"entity_id": int})
exits = events("removed", "1990-01-01")

def first_day(r):
    """First day in the index: its close x the first share count reported; whether the stock
    began trading under 30 days earlier; split-adjusted share change over the next 75 days."""
    eff = pd.Timestamp(r.effective_date)
    p = xfl.prices(entity_id=r.entity_id, start=(eff - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                   end=(eff + pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                   fields=["close", "adj_close", "market_cap"])
    p = p.dropna(subset=["close"]).assign(date=lambda x: pd.to_datetime(x["date"])).sort_values("date")
    new_listing = p["date"].iloc[0] > eff - pd.Timedelta(days=30)
    p = p[p["date"] >= eff]
    s = p.dropna(subset=["market_cap"])
    s = s[s["date"] <= s["date"].iloc[0] + pd.Timedelta(days=75)]
    shares = s["market_cap"] / s["adj_close"]  # in today's split terms
    return (p["date"].iloc[0].strftime("%Y-%m-%d"),
            p["close"].iloc[0] * s["market_cap"].iloc[0] / s["close"].iloc[0],
            new_listing, shares.iloc[-1] / shares.iloc[0])

def priced(ids, day):
    chunks = [ids[i:i + 25] for i in range(0, len(ids), 25)]
    with ThreadPoolExecutor(8) as ex:
        out = ex.map(lambda c: xfl.prices(entity_id=c, start=day, end=day, fields=["market_cap"]), chunks)
    return pd.concat(out).set_index("entity_id")["market_cap"]

with ThreadPoolExecutor(8) as ex:
    ev["day"], ev["line_cap"], ev["new_listing"], ev["share_change"] = zip(*ex.map(first_day, ev.itertuples()))

rows = []
for day, g in ev.groupby("day"):
    roster = xfl.index("sp500", as_of=day).dropna(subset=["entity_id"])
    ids = roster["entity_id"].astype(int).tolist()
    ids += [10006133] if 10025221 in ids else []
    caps = priced(ids, day).combine_first(g.set_index("entity_id")["line_cap"])
    co = caps.groupby(lambda e: CLASS_OF.get(e, e)).sum(min_count=1).dropna()
    for r in g[~g["entity_id"].isin(CLASS_OF)].itertuples():  # a second class is not a new member
        others = co.drop(r.entity_id)
        out = exits[(exits["entity_id"] == r.entity_id) & (exits["effective_date"] >= r.effective_date)]
        rows.append({"entity_id": r.entity_id, "name": r.entity_name, "date": r.effective_date,
                     "cap_bn": co[r.entity_id] / 1e9, "pctile": 100 * (others < co[r.entity_id]).mean(),
                     "median_member_bn": others.median() / 1e9, "members_valued": len(others),
                     "new_listing": r.new_listing, "share_change": r.share_change,
                     "last_day": out["effective_date"].min()})
adds = pd.DataFrame(rows)
adds["year"] = adds["date"].str[:4].astype(int)
adds["placement"] = (pd.to_datetime(adds["last_day"]) - pd.to_datetime(adds["date"])).dt.days < 30
moved = (adds["share_change"] > 1.2) | (adds["share_change"] < 1 / 1.2)  # mostly stock-financed mergers
sample = adds[~moved].copy()

today = xfl.index("sp500").dropna(subset=["entity_id"])
now_ids = set(today["entity_id"].astype(int).map(lambda e: CLASS_OF.get(e, e)))
sample["member_today"] = sample["entity_id"].isin(now_ids)
spells = set(zip(today["entity_id"].astype(int), today["added_date"]))
sample["in_todays_list"] = [(e, d) in spells for e, d in zip(sample["entity_id"], sample["date"])]

# ---- output ----
print(f"S&P 500 additions effective 1990-2025: {len(ev)} in the log")
print(f"  second share classes of members set aside: {ev['entity_id'].isin(CLASS_OF).sum()}")
print(f"  share count moved >20% within 75 days of joining, set aside: {moved.sum()}")
print(f"  additions valued: {len(sample)}; members valued on each entry day: "
      f"{sample['members_valued'].min()}-{sample['members_valued'].max()}")
print(f"  began trading under 30 days before joining: {sample['new_listing'].sum()}; "
      f"left the index within 30 days: {sample['placement'].sum()}")

bins = [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023, 2025]
labels = ["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23", "2024-25"]
sample["period"] = pd.cut(sample["year"], bins, labels=labels)
t = sample.groupby("period", observed=True).agg(
    n=("cap_bn", "size"), entry_bn=("cap_bn", "median"), member_bn=("median_member_bn", "median"),
    pctile=("pctile", "median"), new=("new_listing", "mean"), still=("member_today", "mean"))
core_pct = sample[~sample["new_listing"]].groupby("period", observed=True)["pctile"].median()
print("\nperiod   adds  median entry  median member  entry pctile  (not newly listed)  newly listed  in index today")
for p, r in t.iterrows():
    print(f"{p}  {int(r['n']):>5}  ${r['entry_bn']:>9.1f}bn  ${r['member_bn']:>10.1f}bn  {r['pctile']:>11.1f}"
          f"  {core_pct.get(p, float('nan')):>18.1f}  {r['new']:>11.0%}  {r['still']:>13.0%}")

print("\nSurvivors-only rebuild (today's members by their added date) vs the full log")
sample["decade"] = (sample["year"] // 10 * 10).clip(upper=2020)
for d, g in sample.groupby("decade"):
    s = g[g["in_todays_list"]]
    print(f"  {d}s{'' if d < 2020 else ' (2020-25)'}: log {len(g):>3} adds, median ${g['cap_bn'].median():.1f}bn, "
          f"pctile {g['pctile'].median():.1f} | survivors {len(s):>3}, "
          f"median ${s['cap_bn'].median():.1f}bn, pctile {s['pctile'].median():.1f}")

old = sample[sample["year"] <= 2009].copy()
old["quartile"] = pd.cut(old["pctile"], [-1, 25, 50, 75, 100],
                         labels=["bottom quarter", "second quarter", "third quarter", "top quarter"])
print(f"\nAdditions 1990-2009 ({len(old)}) by entry percentile: share still in the index today")
for q, g in old.groupby("quartile", observed=True):
    print(f"  {q:<15} {len(g):>4} adds  median ${g['cap_bn'].median():>5.1f}bn  still members {g['member_today'].mean():.0%}")

cols = ["date", "name", "cap_bn", "pctile"]
show = lambda n: n.title() if n.isupper() else n
print("\nSmallest at entry:")
for _, r in sample.nsmallest(5, "cap_bn")[cols].iterrows():
    print(f"  {r['date']}  {show(r['name']):<34} ${r['cap_bn']:>6.2f}bn  pctile {r['pctile']:>5.1f}")
print("Largest at entry:")
for _, r in sample.nlargest(5, "cap_bn")[cols].iterrows():
    print(f"  {r['date']}  {show(r['name']):<34} ${r['cap_bn']:>6.1f}bn  pctile {r['pctile']:>5.1f}")

# ---- chart: entry size in dollars (top) and inside the index (bottom), by year ----
import matplotlib.pyplot as plt
y = sample.groupby("year").agg(entry=("cap_bn", "median"), member=("median_member_bn", "median"),
                               pctile=("pctile", "median"))
bg, ink, blue, amber = "#0a0a0a", "#e0e0e0", "#3b82f6", "#d97706"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
fig.patch.set_facecolor(bg)
for ax in (a1, a2):
    ax.set_facecolor(bg)
    ax.tick_params(colors=ink, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
a1.plot(y.index, y["entry"], color=blue, lw=2, marker="o", ms=4, label="Median new member on its first day")
a1.plot(y.index, y["member"], color=amber, lw=2, label="Median existing member that day")
a1.set_yscale("log")
a1.set_yticks([1, 2, 5, 10, 20, 50])
a1.set_yticklabels(["$1bn", "$2bn", "$5bn", "$10bn", "$20bn", "$50bn"])
a1.set_ylabel("Market value (log scale)", color=ink)
a1.legend(frameon=False, labelcolor=ink, fontsize=9, loc="upper left")
a1.set_title("S&P 500 Additions: How Big Are New Members at Entry? 1990-2025", color=ink, fontsize=11)
a2.bar(y.index, y["pctile"], color=blue, width=0.7)
a2.axhline(50, color="#555555", lw=1)
a2.set_ylim(0, 100)
a2.set_ylabel("Median rank in index\n(percentile, 100 = largest)", color=ink)
a2.set_xlabel("Year of addition", color=ink)
plt.tight_layout()
plt.savefig("sp500-additions-market-cap-at-entry-python.png", dpi=150, facecolor=bg)
