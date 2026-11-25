# Full write-up: https://xfinlink.com/blog/sp500-dual-class-shares-history-python
#
# How many S&P 500 companies held two index lines (share classes or tracking stocks)
# at once, year by year since 1980, and how much does giving each line its own slot
# move an equal-weight backtest of each December roster?
#
# Optional: set XFL_CACHE_DIR=/some/dir to save every API response and replay it
# later with no API calls. Without it the script runs plainly against the API.
import hashlib
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Optional: set XFL_CACHE_DIR to save every API response as a pickle and replay it on later runs.
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

# S&P 500 roster on the last business day of each December, 1980-2025
days = pd.date_range("1980-12-01", "2025-12-31", freq="BYE")
ros = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in days}

# One company = index lines that share a name on the same roster, or that the override joins.
# Tracking stocks can carry different names, so lines whose names open with the same run of
# initials ("U S X", "I T T", "A T & T") are screened below and ruled on one by one.
SAME_COMPANY = {"U S X MARATHON GROUP": "U S X CORP",       # USX tracking stocks, 1991-05-07 to
                "U S X US STEEL GROUP INC": "U S X CORP"}   # the 2001-12-31 separation
PAIR_END = {"U S X CORP": "2001-12-31"}  # separation date; no index removal is recorded for either line
initials = lambda n: (m.group(1).replace(" ", "") if (m := re.match(r"((?:[A-Z&] )+[A-Z&]) ", n or "")) else None)
for r in ros.values():
    r["company"] = r["entity_name"].map(lambda n: SAME_COMPANY.get(n, n))
screen = {}
for y, r in ros.items():
    r = r.assign(stem=r["entity_name"].map(initials)).dropna(subset=["stem"])
    for st, g in r.groupby("stem"):
        if g["entity_name"].nunique() > 1:
            key = (st, " | ".join(sorted(set(g["entity_name"]))), "one company" if g["company"].nunique() == 1 else "separate")
            screen.setdefault(key, []).append(y)
count = pd.DataFrame({y: {"lines": len(r), "companies": r["company"].nunique(),
                          "multi": (r["company"].value_counts() > 1).sum()} for y, r in ros.items()}).T
pairs = {}
for y, r in ros.items():
    for name, g in r[r.duplicated("company", keep=False)].groupby("company"):
        pairs[frozenset(g["entity_id"])] = name  # latest name

# Dated additions and removals catch two-line spells that start and end between year-ends
ev = pd.concat([xfl.index_events("sp500", start="1980-01-01", end="2025-12-31", offset=o) for o in (0, 1000, 2000)])
add = ev[ev["event_type"] == "added"]
for name, g in add.groupby("entity_name"):
    ids = sorted(set(g["entity_id"]))
    for i in ids:
        for j in ids:
            if i < j and not any({i, j} <= set(k) for k in pairs):
                pairs[frozenset({i, j})] = name
spell = {}
for y, r in ros.items():
    spell.update({e: (a, d if isinstance(d, str) else None)
                  for e, a, d in r[["entity_id", "added_date", "removed_date"]].itertuples(index=False)})
for e, g in ev.groupby("entity_id"):
    if e not in spell:
        spell[e] = (g.loc[g["event_type"] == "added", "effective_date"].min(),
                    g.loc[g["event_type"] == "removed", "effective_date"].max())
both = []
for ids, name in pairs.items():
    a, b = sorted(ids)
    start = max(spell[a][0], spell[b][0])
    end = min(spell[a][1] or "9999", spell[b][1] or "9999", PAIR_END.get(name, "9999"))
    if start <= end:
        both.append({"company": name, "a": a, "b": b, "from": start, "to": end.replace("9999", "present")})
both = pd.DataFrame(both).sort_values("from")

# Monthly total returns for every line on the 1995-2024 rosters, delisting returns included
ids = sorted(set().union(*[set(ros[y]["entity_id"]) for y in range(1995, 2025)]))
px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start="1996-01-01", end="2025-12-31", interval="1mo",
                           fields=["return_daily", "close"], max_rows=500_000) for i in range(0, len(ids), 100)])
px["m"] = px["date"].dt.to_period("M")
R = px.drop_duplicates(["entity_id", "m"]).pivot(index="m", columns="entity_id", values="return_daily")
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2025-12", "M")]
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    secs = g.index.tolist()
    for c in [secs[i:i + 100] for i in range(0, len(secs), 100)]:
        d = xfl.prices(entity_id=c, start=f"{y}-01-01", end="2025-12-31", interval="1mo", fields=["return_daily"])
        for e, v in (d.attrs["meta"].get("delisting") or {}).items():
            e, dr = int(e), v.get("delisting_return")
            if dr is not None and not np.isnan(dr):
                m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
                if m <= R.index[-1]:
                    R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

def held(w, y):
    """Weights w bought in January of y+1 and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12"].reindex(columns=w.index)
    return float((w * (1 + r.fillna(0)).prod()).sum() / w.sum() - 1)

res = []
for y in range(1995, 2025):
    r = ros[y]
    by_row = pd.Series(1.0, index=r["entity_id"])
    by_co = pd.Series((1 / r.groupby("company")["entity_id"].transform("size")).values, index=r["entity_id"])
    res.append({"year": y + 1, "row_ew": held(by_row, y), "company_ew": held(by_co, y)})
res = pd.DataFrame(res).set_index("year")
def label(e, start):
    """Ticker on the first year-end roster after the two lines overlap, else at the addition."""
    on = [r.set_index("entity_id")["ticker"] for y, r in sorted(ros.items()) if str(y) >= start[:4] and e in set(r["entity_id"])]
    return on[0][e] if on else add.loc[add["entity_id"] == e, "ticker"].iloc[-1]
both["lines"] = [f"{label(a, f)}/{label(b, f)}" for a, b, f in zip(both["a"], both["b"], both["from"])]
def corr(a, b, start, end):
    if a not in R or b not in R:
        return np.nan
    r = R.loc[pd.Period(start, "M") + 1:pd.Period(min(end, "2025-12"), "M") - 1, [a, b]].dropna()
    return r[a].corr(r[b]) if len(r) >= 12 else np.nan
both["monthly_corr"] = [corr(a, b, f, t) for a, b, f, t in zip(both["a"], both["b"], both["from"], both["to"])]
res["gap_bp"] = (res["row_ew"] - res["company_ew"]) * 1e4

# ---------------- report ----------------
print("Screen: lines on one roster whose names open with the same initials")
for (st, names, ruling), ys in sorted(screen.items(), key=lambda kv: kv[1][0]):
    print(f"  {st:<5} {ys[0]}-{ys[-1]}  {ruling:<12} {names}")
same = count[count["lines"] == count["companies"]]
print(f"\n{len(count)} December rosters 1980-2025; {len(same)} have one line per company "
      f"({same['lines'].min()} lines, {same['companies'].min()} companies)")
print("\nYear-ends with more index lines than companies")
print(f"{'year':>6}{'lines':>7}{'companies':>11}{'2+ lines':>10}")
for y, r in count.loc[count["lines"] != count["companies"]].iterrows():
    print(f"{y:>6}{r.lines:>7}{r.companies:>11}{r.multi:>10}")

print("\nCompanies with two index lines at once (rosters plus dated changes)")
print(f"{'company':<30}{'lines':<13}{'from':<12}{'to':<12}{'monthly return corr':>20}")
for r in both.to_dict("records"):
    c = "" if np.isnan(r["monthly_corr"]) else f"{r['monthly_corr']:.3f}"
    print(f"{r['company']:<30}{r['lines']:<13}{r['from']:<12}{r['to']:<12}{c:>20}")
on_year_end = {e for r in ros.values() for e in r["entity_id"]}
short = both[[not ({a, b} <= on_year_end) for a, b in zip(both["a"], both["b"])]]
print(f"Two-line spells on no year-end roster: {', '.join(short['company'])}")

# Single-line checks for dual-class companies with one class in the index
for root in ["BRK", "BF"]:
    t = pd.concat(r.assign(year=y) for y, r in ros.items())
    t = t[t["ticker"].fillna("").str.match(rf"^{root}[-.]")]
    print(f"{root}: tickers {sorted(set(t['ticker']))}, at most {t.groupby('year').size().max()} line per roster")
nulls = sum(r["entity_id"].isna().sum() for r in ros.values())
issuer_rows = sum((r["member_grain"] != "security").sum() for r in ros.values())
print(f"Rows with no entity_id: {nulls}; issuer-grain rows: {issuer_rows}")

print("\nEqual-weight portfolio of each December roster, held the next calendar year")
print(f"{'year':>6}{'per line':>10}{'per company':>13}{'gap (bp)':>10}")
dup = res[res["gap_bp"] != 0]
for y, r in dup.iterrows():
    print(f"{y:>6}{r.row_ew:>10.2%}{r.company_ew:>13.2%}{r.gap_bp:>+10.1f}")
growth = (1 + res[["row_ew", "company_ew"]]).prod()
print(f"{len(dup)} of {len(res)} holding years differ; mean gap {dup['gap_bp'].mean():+.1f} bp, "
      f"mean absolute {dup['gap_bp'].abs().mean():.1f} bp, largest {dup['gap_bp'].abs().max():.1f} bp "
      f"in {dup['gap_bp'].abs().idxmax()}")
print(f"$1 invested 1996-2025: per line {growth['row_ew']:.2f}, per company {growth['company_ew']:.2f}")
print(f"Lines whose series ended before December 2025, checked for a delisting return: {len(ended)}")

# ---------------- chart ----------------
import matplotlib.pyplot as plt

BG, FG, ACCENT, ORANGE, GREY = "#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b", "#6b7280"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GREY, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
extra = count["lines"] - count["companies"]
colors = [ORANGE if y < 2014 else ACCENT for y in extra.index]
ax1.bar(extra.index, extra.values, color=colors, width=0.8)
ax1.set_ylabel("Extra index lines\n(lines minus companies)")
ax1.set_ylim(0, 6.2)
for x in (2014.7, 2016.6, 2022.3):  # Sep 2015, Jul 2017, Apr 2023 (bars sit on December rosters)
    ax1.axvline(x, color=GREY, lw=0.8, ls="--")
ax1.text(2014.5, 6.1, "Sep 2015: every\nlisted class line", ha="right", va="top", fontsize=8.5, color=FG)
ax1.text(2016.8, 6.1, "Jul 2017: new\nmulti-class\ncompanies barred", ha="left", va="top", fontsize=8.5, color=FG)
ax1.text(2022.5, 6.1, "Apr 2023:\nbar lifted", ha="left", va="top", fontsize=8.5, color=FG)
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=ORANGE, label="Tracking stocks (USX, U S West, Sprint)"),
                    Patch(color=ACCENT, label="Share classes (Alphabet, Discovery, Fox, News Corp, Under Armour)")],
           loc="upper left", frameon=False, fontsize=8.5)
ax1.set_title("S&P 500 dual-class shares history: extra lines and their backtest effect", color=FG)
ax2.bar(res.index - 1, res["gap_bp"], color=[ORANGE if y < 2014 else ACCENT for y in res.index - 1], width=0.8)
ax2.axhline(0, color=GREY, lw=0.8)
ax2.set_ylabel("Per-line minus per-company\nequal-weight return (bp)")
ax2.set_xlabel("December roster (held the following year)")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("sp500-dual-class-shares-history-python.png", dpi=150, facecolor=BG)
