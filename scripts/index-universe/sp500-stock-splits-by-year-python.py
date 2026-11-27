# Full write-up: https://xfinlink.com/blog/sp500-stock-splits-by-year-python
#
# How many S&P 500 companies split their stock each year since 1980? A split counts when
# the company was in the index on the split's ex-date (point-in-time membership, removal
# dates inclusive). A forward split is a ratio of 1.25 or more (5-for-4 and up) that is a
# plain fraction p/q with q <= 4; smaller stock dividends and merger exchange ratios are
# not splits. A reverse split is a plain ratio of 0.8 or less. Share classes of one company
# count once. The same count on today's roster, traced back through each member's own
# history, shows what a survivor list reports instead.
#
# Optional: set XFL_CACHE_DIR=/some/dir to save every API response and replay it
# later with no API calls. Without it the script runs plainly against the API.
import hashlib
import json
import os
import pickle
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

CACHE = os.environ.get("XFL_CACHE_DIR")
if CACHE:
    def _cached(name, fn):
        def wrapper(*args, **kwargs):
            key = hashlib.md5(json.dumps([name, args, kwargs], sort_keys=True, default=str).encode()).hexdigest()
            path = os.path.join(CACHE, f"{name}_{key}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            out = fn(*args, **kwargs)
            os.makedirs(CACHE, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(out, f)
            return out
        return wrapper
    for _name in ("index", "index_events", "prices"):
        setattr(xfl, _name, _cached(_name, getattr(xfl, _name)))

# ---- Membership spells: December rosters 1979-2025, today's roster, and the dated event log
days = pd.date_range("1979-12-01", "2025-12-31", freq="BYE")
ros = [xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in days]
today = xfl.index("sp500")
ev = pd.concat([xfl.index_events("sp500", start="1979-01-01", end="2026-11-26", limit=1000, offset=o)
                for o in (0, 1000, 2000)], ignore_index=True)
sp = pd.concat(ros + [today]).drop_duplicates(["entity_id", "added_date"])
sp = sp[["entity_id", "entity_name", "added_date", "removed_date"]]
known_add, known_rem = set(zip(sp.entity_id, sp.added_date)), set(zip(sp.entity_id, sp.removed_date))
adds = ev[(ev.event_type == "added") & ~pd.Series(list(zip(ev.entity_id, ev.effective_date))).isin(known_add)]
rems = ev[(ev.event_type == "removed") & ~pd.Series(list(zip(ev.entity_id, ev.effective_date))).isin(known_rem)]
short = []  # spells that start and end between two December rosters
for a in adds.dropna(subset=["entity_id"]).itertuples():
    r = rems[(rems.entity_id == a.entity_id) & (rems.effective_date >= a.effective_date)].effective_date.min()
    short.append({"entity_id": a.entity_id, "entity_name": a.entity_name, "added_date": a.effective_date,
                  "removed_date": r if isinstance(r, str) else None})
sp = pd.concat([sp, pd.DataFrame(short)], ignore_index=True).dropna(subset=["entity_id"])
sp["entity_id"] = sp.entity_id.astype(int)
sp["a"] = pd.to_datetime(sp.added_date)
sp["r"] = pd.to_datetime(sp.removed_date).fillna(pd.Timestamp("2099-12-31"))
sp = sp[sp.r >= "1980-01-01"]
ids = sorted(set(sp.entity_id))

# ---- Monthly bars: split_ratio is the product of the month's ratios; close/adj_close is the factor
px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start="1979-12-01", end="2026-11-26", interval="1mo",
                           fields=["close", "adj_close", "split_ratio", "shares_outstanding"], max_rows=250_000)
                for i in range(0, len(ids), 100)], ignore_index=True).sort_values(["entity_id", "date"])
k = px.close / px.adj_close
px["factor"] = k.groupby(px.entity_id).shift(1) / k
# One company = index lines that carry the same name on the same roster (share classes)
root = {e: e for e in ids}
def find(e):
    while root[e] != e:
        e = root[e]
    return e
for r in ros + [today]:
    for _, g in r.dropna(subset=["entity_id"]).groupby("entity_name"):
        e = sorted(set(g.entity_id.astype(int)) & set(ids))
        for x in e[1:]:
            root[find(x)] = find(e[0])
for a in sp.itertuples():  # short spells: same recorded name, overlapping dates
    o = sp[(sp.entity_name == a.entity_name) & (sp.entity_id != a.entity_id) & (sp.a <= a.r) & (sp.r >= a.a)]
    for x in o.entity_id:
        root[find(x)] = find(a.entity_id)
cur = px.groupby("entity_id").entity_name.last()
name = {e: cur.get(find(e), cur.get(e)) for e in ids}               # company label: its current name
coid = {e: find(e) for e in ids}
sp["co"] = sp.entity_id.map(coid)

def plain(r):
    f = Fraction(r if r >= 1 else 1 / r).limit_denominator(4)
    return abs(float(f) - (r if r >= 1 else 1 / r)) < 0.002 and f.denominator <= 4

# Google, Discovery, Under Armour, Wrigley share dividends filed as splits: Google (April 2014),
# Discovery (August 2014) and Under Armour (April 2016) one Class C share per share (2-for-1);
# Wrigley (April 2006) one Class B share per four shares (5-for-4). The old line's factor steps
# while its split_ratio stays 1.
CLASS_C = {(10030636, "2014-04"): 2.0, (10031087, "2014-08"): 2.0, (10031256, "2016-04"): 2.0, (10004401, "2006-04"): 1.25}
px["ratio"] = [CLASS_C.get((e, d.strftime("%Y-%m")), r) for e, d, r in zip(px.entity_id, px.date, px.split_ratio)]
cand = px[(px.date >= "1980-01-01") & (px.date <= "2025-12-31") & ((px.ratio >= 1.25) | (px.ratio <= 0.8))].copy()
cand["plain"] = cand.ratio.map(plain)
cand["agree"] = (cand.factor / cand.ratio - 1).abs() < 0.03

# ---- Ex-date membership: whole month inside a spell, or a daily pull when a spell starts or ends that month
# (ex holds the month's bar date, or the exact ex-date where a daily pull was needed)
F = ["close", "adj_close", "split_ratio", "return_daily", "shares_outstanding"]
def ex_dates(row):
    m0, m1 = row.date.to_period("M").start_time, row.date.to_period("M").end_time.normalize()
    s = sp[sp.entity_id == row.entity_id]
    if ((s.a <= m0) & (s.r >= m1)).any():
        return row.date, True
    if not ((s.a <= m1) & (s.r >= m0)).any():
        return row.date, False
    d = xfl.prices(entity_id=int(row.entity_id), start=str(m0.date()), end=str(m1.date()), fields=F)
    x = d.loc[d.split_ratio.notna(), "date"].min()
    return x, bool(((s.a <= x) & (s.r >= x)).any())
cand["ex"], cand["member"] = zip(*[ex_dates(r) for r in cand.itertuples()])
cand["year"] = cand.ex.map(lambda x: x.year)
cand["co"] = cand.entity_id.map(coid)
cand["so_r"] = (px.shares_outstanding / px.groupby("entity_id").shares_outstanding.shift(1)).reindex(cand.index)
# Two signals: the recorded ratio, confirmed by the adj_close factor (3%) or the share count (15%)
cand["confirmed"] = cand.agree | ((cand.so_r / cand.ratio - 1).abs() < 0.15)
ok = cand[cand.plain & cand.confirmed]
fwd = ok[ok.ratio >= 1.25]
# A reverse split needs the factor to agree, or the company's own filing where a spin-off on the
# same day moves the factor too (AT&T 2002, Allegheny 1999, Tyco 2007, Time Warner 2009,
# Motorola 2011, Expedia 2011).
FILED_REVERSE = {(10000314, "2002-11"), (10015128, "1999-11"), (10015379, "2007-07"), (10020848, "2009-03"),
                 (10010259, "2011-01"), (10031090, "2011-12")}
# Viacom's 2006-01 1-for-2 is the share conversion at the Viacom/CBS separation (each old share became
# half a CBS share and half a new Viacom share), not a reverse split.
NOT_REVERSE = {(10019803, "2006-01")}
key = pd.Series([(e, d.strftime("%Y-%m")) for e, d in zip(ok.entity_id, ok.date)], index=ok.index)
rev = ok[(ok.ratio <= 0.8) & ok.member & (ok.agree | key.isin(FILED_REVERSE)) & ~key.isin(NOT_REVERSE)]

# ---- Year table, point-in-time roster vs today's roster traced back
today_co = set(today.dropna(subset=["entity_id"]).entity_id.astype(int).map(coid).dropna())
bars = px.assign(year=px.date.dt.year, co=px.entity_id.map(coid))
trading = bars.groupby("year").co.apply(set)
rows = []
for y in range(1980, 2026):
    y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    members = set(sp[(sp.a <= y1) & (sp.r >= y0)].co)
    split_pit = set(fwd[(fwd.year == y) & fwd.member].co)
    today_trading = today_co & trading.get(y, set())
    split_today = set(fwd[fwd.year == y].co) & today_trading
    rows.append({"year": y, "members": len(members), "splits": len(split_pit), "share": len(split_pit) / len(members),
                 "still_in": len(split_pit & today_co), "today_n": len(today_trading), "today_splits": len(split_today),
                 "today_share": len(split_today) / len(today_trading),
                 "stay_n": len(members & today_co), "stay_split": len(split_pit & today_co), "reverse": rev[rev.year == y].co.nunique()})
t = pd.DataFrame(rows).set_index("year")
full = t

# ---------------- report ----------------
print(f"{len(days)} December rosters 1979-2025, today's roster ({len(today)} lines), {len(ev):,} index events, "
      f"{len(ids):,} securities")
print(f"Ratios of 1.25+ or 0.8- on member and non-member lines, 1980-2025: {len(cand):,}; adj_close factor agrees on "
      f"{cand.agree.sum():,}, share count alone on {(cand.confirmed & ~cand.agree).sum()}; set aside: "
      f"{(~cand.plain).sum()} not a plain ratio (merger exchange, recapitalisation), "
      f"{(cand.plain & ~cand.confirmed).sum()} unconfirmed by either")
print(f"\n{'year':>5}{'members':>9}{'splits':>8}{'share':>7}{'still in':>10}{'today':>7}{'splits':>8}{'share':>7}{'reverse':>9}")
for y, r in t.iterrows():
    print(f"{y:>5}{r.members:9.0f}{r.splits:8.0f}{r.share:7.1%}{r.still_in:10.0f}{r.today_n:7.0f}{r.today_splits:8.0f}"
          f"{r.today_share:7.1%}{r.reverse:9.0f}")
print()
for a, b in [(1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2025)]:
    d = full.loc[a:b]
    print(f"{a}-{b}: {d.splits.sum():.0f} splits, {d.splits.mean():.1f} a year ({d.share.mean():.1%} of members); "
          f"today's roster {d.today_splits.mean():.1f} a year ({d.today_share.mean():.1%})")
top = full.splits.max()
print(f"Most: {top:.0f} in {', '.join(str(y) for y in full.index[full.splits == top])}; fewest: {full.splits.min():.0f} in "
      f"{', '.join(str(y) for y in full.index[full.splits == full.splits.min()])}")
gap = full.today_share - full.share
print(f"Today's-roster share minus point-in-time share: mean {gap.mean() * 100:+.1f} points, above zero in "
      f"{(gap > 0).sum()} of {len(full)} years; 1980-1999 {gap.loc[:1999].mean() * 100:+.1f}, "
      f"2000-2025 {gap.loc[2000:].mean() * 100:+.1f}")
m = fwd[fwd.member & fwd.year.between(1980, 2025)].drop_duplicates(["co", "year"])
for a, b in [(1980, 1999), (2000, 2019), (2020, 2025)]:
    q = m[m.year.between(a, b)].ratio
    print(f"Split ratios {a}-{b}: {(q.round(3) == 2).mean():.0%} two-for-one, {(q < 2).mean():.0%} below it, "
          f"{(q >= 4).mean():.0%} four-for-one or more ({len(q)} splits)")
stay = full.stay_split / full.stay_n
gone = (full.splits - full.stay_split) / (full.members - full.stay_n)
print(f"Within the real roster, mean yearly split rate: {stay.mean():.1%} for companies still in the index today, "
      f"{gone.mean():.1%} for companies removed since (higher in {(stay > gone).sum()} of {len(full)} years)")
print(f"Point-in-time splitters 1980-2025 still in the index today: {full.still_in.sum():.0f} of {full.splits.sum():.0f}")

dated_rosters = list(zip(days, ros)) + [(pd.Timestamp.today().normalize(), today)]
def name_then(e, x):  # the company name on the last roster before the ex-date (else the nearest one)
    on = [r for r in dated_rosters if (r[1].entity_id == e).any()]
    r = max((r for r in on if r[0] <= x), key=lambda r: r[0], default=None) or min(on, key=lambda r: abs(r[0] - x))
    r = r[1]
    return r.loc[r.entity_id == e, "entity_name"].iloc[0]
print(f"\nReverse splits by members on the ex-date, 1980-2025: {len(rev)}")
for x in rev.sort_values("ex").itertuples():
    f = Fraction(1 / x.ratio).limit_denominator(4)
    print(f"  {x.ex:%Y-%m}  {x.ticker:<6}{name_then(x.entity_id, x.ex)[:32]:<33} {f.denominator}-for-{f.numerator}")
print("\nForward splits while a member, spot checks:")
for c in ["APPLE INC", "MICROSOFT CORP", "WALMART INC", "NVIDIA CORP", "AMAZON COM INC", "ALPHABET INC", "TESLA INC"]:
    s = fwd[fwd.member & (fwd.co.map(name) == c)].drop_duplicates(["co", "ex"]).sort_values("ex")
    print(f"  {c:<15}" + ", ".join(f"{x.ex:%Y-%m} {Fraction(x.ratio).limit_denominator(4)}" for x in s.itertuples()))

# ---- Chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(full.index, full.share * 100, color="#3b82f6", width=0.75, label="Roster in force on the ex-date")
ax.plot(full.index, full.today_share * 100, color="#f97316", lw=2, marker="o", ms=3.5, label="Today's roster, traced back")
for x, y, txt in [(1986.5, 1987, "105 each"), (1997, 1997, "97"), (2009, 2009, "1"), (2024, 2024, "16")]:
    ax.annotate(txt, (x, full.loc[y, "share"] * 100), xytext=(0, 4), textcoords="offset points",
                ha="center", color="#e0e0e0", fontsize=8)  # labels: number of companies
ax.set_xlim(1979, 2026)
ax.set_xlabel("Calendar year of the ex-date")
ax.set_ylabel("Companies splitting 5-for-4 or more (% of members)")
ax.set_title("S&P 500 Stock Splits by Year, 1980-2025")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("sp500-stock-splits-by-year-python.png", dpi=150, facecolor="#0a0a0a")
