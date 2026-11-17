# Full write-up: https://xfinlink.com/blog/sp500-additions-profitability-rule-python
"""S&P 500 additions and the earnings rule, on figures known the day before joining.

For every S&P 500 addition effective 2011-01-01 to 2026-09-30:
  1. the earnings test on the quarterly GAAP net income filed by the day before
     the effective date (sum of the four most recent quarters > 0 and the most
     recent quarter > 0);
  2. who did not pass, split by how long the stock had traded before joining;
  3. how often additions that passed reported a trailing-four-quarter loss
     within 1, 2 and 3 years, against incumbent members that passed the same
     test on the preceding 31 December, and whether those additions left the
     index sooner.
"""
import warnings

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

warnings.simplefilter("ignore")  # paging and coverage notices
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST, LAST = pd.Timestamp("2011-01-01"), pd.Timestamp("2026-09-30")
HORIZON = pd.Timestamp("2026-06-30")  # last quarter end used for outcomes
NI = ["net_income", "net_income_attributable_to_parent"]


def quarters(f):
    """One row per quarter in period_end order; a filing row beats an archive row within 45 days."""
    f = f.assign(ni=f["net_income_attributable_to_parent"].fillna(f["net_income"]),
                 pri=(f["source"] != "filing").astype(int)).sort_values(["period_end", "pri"])
    keep = []
    for r in f.itertuples():
        if keep and (r.period_end - keep[-1].period_end).days < 45:
            if r.pri < keep[-1].pri:
                keep[-1] = r
            continue
        keep.append(r)
    return pd.DataFrame(keep, columns=f.columns.tolist() if not keep else None)


def ttm_series(q):
    """Trailing four-quarter net income at each quarter end (consecutive quarters only)."""
    q = q[q["ni"].notna()].reset_index(drop=True)
    out = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3:i + 1]
        if 240 <= (w["period_end"].iloc[-1] - w["period_end"].iloc[0]).days <= 300:
            out.append((w["period_end"].iloc[-1], w["ni"].sum(), w["ni"].iloc[-1], (w["ni"] > 0).all()))
    return pd.DataFrame(out, columns=["pe", "ttm", "lastq", "all4"])


def earnings_test(f, known):
    """(verdict, ttm, last quarter, all four positive, last quarter end) on the quarters known at `known`."""
    if f is None or len(f) == 0:
        return "untested", np.nan, np.nan, False, pd.NaT
    t = ttm_series(quarters(f))
    q = quarters(f)
    q = q[q["ni"].notna()]
    if len(q) < 4 or len(t) == 0 or t["pe"].iloc[-1] != q["period_end"].iloc[-1] \
            or (known - t["pe"].iloc[-1]).days > 230:
        return "untested", np.nan, np.nan, False, pd.NaT   # four consecutive recent quarters not on file
    x = t.iloc[-1]
    return ("pass" if x.ttm > 0 and x.lastq > 0 else "fail"), x.ttm, x.lastq, bool(x.all4), x.pe


def pit(tickers, known, entity_id=None):
    """Quarterly net income as filed by `known`, 30 months back."""
    kw = dict(period_type="quarterly", as_of=known.strftime("%Y-%m-%d"),
              start=(known - pd.DateOffset(months=30)).strftime("%Y-%m-%d"), fields=NI)
    try:
        return xfl.fundamentals(entity_id=entity_id, **kw) if entity_id else xfl.fundamentals(tickers, **kw)
    except xfl.XfinlinkError:   # no company under the ticker that day (a when-issued spin-off, say)
        return pd.DataFrame()


# 1. Dated additions and removals, one year per call, each checked against meta.total
ev = []
for y in range(2010, 2027):
    e = xfl.index_events("sp500", start=f"{y}-01-01", end=f"{y}-12-31" if y < 2026 else None)
    assert e.attrs["meta"]["total"] == len(e)
    ev.append(e)
ev = pd.concat(ev, ignore_index=True)
ev["d"] = pd.to_datetime(ev["effective_date"])
add = ev[ev["event_type"] == "added"].copy()
add["known"] = add["d"] - pd.Timedelta(days=1)          # figures filed by the day before joining

# 2. Point-in-time quarters for each addition, one call per effective date
P = {}
for known, g in add.dropna(subset=["ticker"]).groupby("known"):
    f = pit(sorted(set(g["ticker"])), known)
    for t in g["ticker"]:
        P[(known, t)] = f[f["ticker"] == t] if len(f) else f

# 3. Roster and point-in-time quarters on each 31 December (incumbent control, issuer ids)
rosters, R = {}, []
for y in range(2010, 2025):
    d = pd.Timestamp(f"{y}-12-31")
    ro = xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))
    assert ro.attrs["meta"]["total"] == len(ro)
    rosters[d] = ro
    tk = sorted(set(ro["ticker"].dropna()))
    for i in range(0, len(tk), 100):
        f = pit(tk[i:i + 100], d)
        R.append(f.assign(known=d))
R = pd.concat(R, ignore_index=True)
issuer_on = R.groupby(["known", "ticker"])["entity_id"].first()   # ticker -> issuer, each 31 December

# 4. First and last monthly price of every security in the event log
ids = sorted(set(ev["entity_id"].dropna().astype(int)))
px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start="2008-01-01", end="2026-11-16",
                           interval="1mo", fields=["close"], max_rows=100_000)
                for i in range(0, len(ids), 100)])
first_px, last_px = px.groupby("entity_id")["date"].min(), px.groupby("entity_id")["date"].max()

# 5. Classify each addition in the sample window
add = add[(add["d"] >= FIRST) & (add["d"] <= LAST)].copy()
rows = []
for a in add.itertuples():
    prev = max([d for d in rosters if d < a.d], default=None)
    ro = rosters.get(prev)
    members = set() if ro is None else set(ro.loc[ro["removed_date"].isna() | (pd.to_datetime(ro["removed_date"]) >= a.d), "entity_name"])
    members |= set(ev.loc[(ev["d"] > prev) & (ev["d"] < a.d) & (ev["event_type"] == "added"), "entity_name"])
    same_day = add[(add["d"] == a.d) & (add["entity_name"] == a.entity_name)]
    share_class = a.entity_name in members or same_day.index[0] != a.Index
    if pd.isna(a.entity_id) or share_class:
        continue                       # another class of a company already in, or no company record
    f = P.get((a.known, a.ticker), pd.DataFrame())
    issuer = f["entity_id"].iloc[0] if len(f) else np.nan
    later = next((issuer_on.get((d, a.ticker)) for d in sorted(rosters) if d > a.d and (d, a.ticker) in issuer_on), np.nan)
    new_listing = pd.isna(first_px.get(a.entity_id)) or (a.d - first_px[a.entity_id]).days < 365
    if (len(f) == 0 and pd.notna(later)) or (new_listing and pd.notna(later) and later != issuer):
        f = pit(None, a.known, entity_id=int(later))   # pin the company the ticker named after joining
        issuer = later
    elif new_listing and pd.notna(issuer) and issuer in set(issuer_on[issuer_on.index.get_level_values(0) < a.d]):
        f, issuer = pd.DataFrame(), np.nan     # the ticker still named the parent company the day before
    status, ttm, lastq, all4, lastpe = earnings_test(f, a.known)
    removed = ev[(ev["entity_id"] == a.entity_id) & (ev["event_type"] == "removed") & (ev["d"] > a.d)]["d"].min()
    rows.append(dict(d=a.d, ticker=a.ticker, name=a.entity_name, sec=a.entity_id, issuer=issuer,
                     new_listing=new_listing, status=status, ttm=ttm, lastq=lastq, all4=all4, lastpe=lastpe,
                     removed=removed, last_px=last_px.get(a.entity_id)))
S = pd.DataFrame(rows)

# 6. Later trailing losses on first-reported quarters, additions and incumbents
iss = sorted(set(R["entity_id"].astype(int)) | set(S["issuer"].dropna().astype(int)))
H = pd.concat([xfl.fundamentals(entity_id=iss[i:i + 100], period_type="quarterly", version="original",
                                start="2010-01-01", end="2026-09-30", fields=NI, max_rows=50_000)
               for i in range(0, len(iss), 100)])
TT = {e: ttm_series(quarters(g)) for e, g in H.groupby("entity_id")}


def loss_within(e, t0, years):
    """True if a trailing four-quarter sum ending in (t0, t0 + years] is negative; None if not observable."""
    t1 = t0 + pd.DateOffset(years=years)
    if t1 > HORIZON or e not in TT:
        return None
    w = TT[e][(TT[e]["pe"] > t0) & (TT[e]["pe"] <= t1)]
    return None if len(w) == 0 else bool((w["ttm"] < 0).any())


inc = []
for d, ro in rosters.items():
    old = ro[pd.to_datetime(ro["added_date"]) <= d - pd.Timedelta(days=365)]
    for e in set(issuer_on.loc[d].reindex(old["ticker"].dropna()).dropna().astype(int)):
        if earnings_test(R[(R["known"] == d) & (R["entity_id"] == e)], d)[0] == "pass":
            inc.append(dict(R=d, year=d.year + 1, issuer=e, **{f"loss{n}": loss_within(e, d, n) for n in (1, 2, 3)}))
INC = pd.DataFrame(inc)
ok = S[(S["status"] == "pass") & S["issuer"].notna()].copy()
for n in (1, 2, 3):
    ok[f"loss{n}"] = [loss_within(int(e), d, n) for e, d in zip(ok["issuer"], ok["d"])]
ok["year"] = ok["d"].dt.year

# 7. Results
# Route into the index for established listings that failed, from S&P Dow Jones Indices' announcements
ROUTE = {"VRTX": "from MidCap 400", "MLM": "from MidCap 400", "MNK": "from MidCap 400", "ENDP": "from MidCap 400",
         "EQIX": "from MidCap 400", "AMD": "from MidCap 400", "ARE": "from MidCap 400", "NKTR": "from SmallCap 600",
         "CZR": "from MidCap 400", "PENN": "from MidCap 400", "CDAY": "from MidCap 400", "EQT": "from MidCap 400",
         "FSLR": "from MidCap 400", "VST": "from MidCap 400", "EXE": "from MidCap 400", "SATS": "from MidCap 400",
         "AGL": "from MidCap 400"}
est, new = S[~S["new_listing"]], S[S["new_listing"]]
n_window = len(add)
print(f"Additions {FIRST.date()} to {LAST.date()}: {n_window}; another share class of a member or no company record: "
      f"{n_window - len(S)}; sample: {len(S)}")
for t in ["TSLA", "UBER", "ABNB"]:
    x = S[S["ticker"] == t].iloc[0]
    print(f"  check {t} {x.d.date()}: four-quarter net income {x.ttm:,.0f}m, latest quarter {x.lastq:,.0f}m -> {x.status}")
print(f"Passed on figures filed by the day before joining: {(S.status == 'pass').sum()} of {len(S)} ({(S.status == 'pass').mean():.1%})")
for lab, g in [("Traded 12+ months before joining", est), ("Traded under 12 months (spin-offs, merger-created)", new)]:
    c = g["status"].value_counts()
    print(f"  {lab}: {len(g)}  pass {c.get('pass', 0)} ({c.get('pass', 0) / len(g):.1%})  fail {c.get('fail', 0)}  "
          f"not testable {c.get('untested', 0)}")
pre, post = est[est["d"] < "2014-03-24"], est[est["d"] >= "2014-03-24"]
print(f"Passers that also had four positive quarters (the pre-2014 test): before 24 Mar 2014 "
      f"{pre[pre.status == 'pass'].all4.sum()} of {(pre.status == 'pass').sum()}, since {post[post.status == 'pass'].all4.sum()} "
      f"of {(post.status == 'pass').sum()}")
f = est[est["status"] == "fail"].assign(route=lambda x: x["ticker"].map(ROUTE).fillna("outside addition"))
print("\nEstablished listings that failed (net income, $m)")
print(f.assign(joined=f["d"].dt.strftime("%Y-%m-%d"))[["joined", "ticker", "ttm", "lastq", "route"]]
      .rename(columns={"ttm": "4 quarters", "lastq": "latest"})
      .to_string(index=False, formatters={"4 quarters": "{:,.0f}".format, "latest": "{:,.0f}".format}))
f = new[new["status"] == "fail"]
print("New listings that failed: " + ", ".join(f"{t} {d.year}" for t, d in zip(f["ticker"], f["d"])))

# Today's restated figures: the same four quarters, and the four that had ended (not been filed) by joining
L = pd.concat([xfl.fundamentals(entity_id=sorted(set(S["issuer"].dropna().astype(int)))[i:i + 100],
                                period_type="quarterly", start="2008-07-01", end="2026-09-30", fields=NI, max_rows=50_000)
               for i in range(0, S["issuer"].nunique(), 100)])
TL = {e: ttm_series(quarters(g)) for e, g in L.groupby("entity_id")}
same, ended = [], []
for a in S[S["status"].isin(["pass", "fail"])].itertuples():
    t = TL.get(a.issuer)
    if t is None:
        continue
    for x, out in [(t[(t["pe"] - a.lastpe).abs() <= pd.Timedelta(days=10)], same), (t[t["pe"] < a.d], ended)]:
        if len(x):
            v = "pass" if x["ttm"].iloc[-1] > 0 and x["lastq"].iloc[-1] > 0 else "fail"
            if v != a.status:
                out.append(f"{a.ticker} {a.d.year} {a.status}->{v}")
n_tested = S["status"].isin(["pass", "fail"]).sum()
print(f"\nVerdicts that change on today's figures for the same quarters: {len(same)} of {n_tested}: {', '.join(same)}")
print(f"... on the four quarters that had ended (filed or not) before joining: {len(ended)}: {', '.join(ended)}")

print("\nTrailing four-quarter loss reported within N years of joining (additions that passed at entry)")
rates = []
for n in (1, 2, 3):
    a = ok.dropna(subset=[f"loss{n}"])
    c = INC.dropna(subset=[f"loss{n}"])
    w = a["year"].map(c.groupby("year")[f"loss{n}"].mean()).dropna().mean()   # incumbents, additions' year mix
    k, m = int(a[f"loss{n}"].sum()), len(a)
    rates.append((k / m, w))
    print(f"  {n} year(s): additions {k}/{m} = {k / m:.1%}   incumbents {w:.1%} ({len(c):,} member-years)   "
          f"binomial p {stats.binomtest(k, m, w).pvalue:.3f}")

today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))
a3 = ok.dropna(subset=["loss3"]).assign(member_today=lambda x: x["sec"].isin(today))
s3 = a3[a3["member_today"]]
print(f"  3 years, additions still in the index today only: {int(s3.loss3.sum())}/{len(s3)} = {s3.loss3.mean():.1%}")

a5 = ok[ok["d"] <= "2021-06-30"].dropna(subset=["loss3"]).copy()
a5["out5"] = a5["removed"].notna() & (a5["removed"] <= a5["d"] + pd.DateOffset(years=5))
a5["acquired"] = a5["out5"] & (a5["last_px"] <= a5["removed"] + pd.DateOffset(months=2))
print("\nRemoved within 5 years of joining (additions to June 2021)")
for lab, g in a5.groupby("loss3"):
    print(f"  {'loss within 3 years' if lab else 'no loss':20s} {len(g):4d}  removed {g.out5.mean():.1%}  "
          f"(still trading after removal {(g.out5 & ~g.acquired).mean():.1%}, stopped trading {g.acquired.mean():.1%})")
print(f"  Fisher exact p = {stats.fisher_exact(pd.crosstab(a5['loss3'], a5['out5']))[1]:.4f}")

# 8. Chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BG, FG, MUTED = "#0a0a0a", "#e0e0e0", "#a3a3a3"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#404040", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.05, 1]})
groups = [("Traded 12+ months\nbefore joining", est), ("Spin-offs and\nmerger-created\n(under 12 months)", new)]
cols = [("pass", "Passed", "#3b82f6"), ("fail", "Failed", "#d97706"), ("untested", "Not testable\n(fewer than four recent quarters)", "#a855f7")]
for i, (lab, g) in enumerate(groups):
    left = 0
    for key, name, c in cols:
        v = (g["status"] == key).sum()
        ax1.barh(i, v, left=left, color=c, height=0.55, edgecolor=BG, linewidth=2, label=name if i == 0 else None)
        if v >= 30:
            ax1.text(left + v / 2, i, str(v), ha="center", va="center", color="#ffffff", fontsize=10, fontweight="bold")
        left += v
    small = [f"{(g['status'] == k).sum()} {n.split(chr(10))[0].lower()}" for k, n, _ in cols if (g["status"] == k).sum() < 30]
    ax1.text(left + 4, i, "\n".join(small), va="center", fontsize=9, color=FG)
ax1.set_yticks([0, 1], [g[0] for g in groups])
ax1.invert_yaxis()
ax1.set_xlim(0, 340)
ax1.set_xlabel("Additions, 2011 to September 2026")
ax1.set_title("Earnings test on figures filed\nby the day before joining", fontsize=11)
ax1.legend(frameon=False, loc="center right", fontsize=9)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

x = np.arange(3)
for j, (name, c) in enumerate([("Additions that passed", "#3b82f6"), ("Incumbents that passed, same years", MUTED)]):
    vals = [r[j] * 100 for r in rates]
    ax2.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=c, label=name)
    for xi, v in zip(x, vals):
        ax2.text(xi + (j - 0.5) * 0.36, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color=FG)
ax2.set_xticks(x, ["Within 1 year", "Within 2 years", "Within 3 years"])
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
ax2.set_ylabel("Share reporting a trailing four-quarter loss")
ax2.set_title("Later losses: additions vs incumbents", fontsize=11)
ax2.legend(frameon=False, loc="upper left", fontsize=9)
ax2.set_ylim(0, 22)
ax2.set_yticks([0, 5, 10, 15, 20])
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.suptitle("S&P 500 New Additions and the Profitability Rule", fontsize=13, color=FG)
plt.tight_layout()
plt.savefig("sp500-additions-profitability-rule-python.png", dpi=150, facecolor=BG)
