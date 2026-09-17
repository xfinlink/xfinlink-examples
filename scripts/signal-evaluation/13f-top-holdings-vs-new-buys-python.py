# Full write-up: https://xfinlink.com/blog/13f-top-holdings-vs-new-buys-python
#
# Copying institutional filings is a real strategy with real products behind it.
# This measures which part of a Form 13F carries the return. Fifteen selective
# managers, every quarter end from 2013 to mid-2024, three equal-weighted portfolios
# rebuilt from each disclosed book: the ten largest positions, positions eleven
# to fifty, and every position the manager did not hold a quarter earlier. Each
# is bought two months after the quarter end, held one quarter, and measured
# against SPY, with an equal-weighted index fund as a second yardstick.

import time
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

MANAGERS = {
    146: "Dodge & Cox", 263: "Harris Associates", 683: "Artisan Partners",
    1310: "First Eagle", 3481: "Capital Research Global", 446: "Ruane Cunniff",
    9948: "Southeastern Asset", 616: "Akre Capital", 58: "Berkshire Hathaway",
    9349: "Lone Pine Capital", 10151: "Viking Global", 776: "Tiger Global",
    1608: "Baupost Group", 662: "Appaloosa", 505: "Third Point",
}
CHUNKS = [("2012-10-01", "2015-12-31"), ("2016-01-01", "2018-12-31"),
          ("2019-01-01", "2021-12-31"), ("2022-01-01", "2024-06-30")]
FIRST_Q, LAST_Q, DEEP = "2013Q1", "2024Q2", 50
PNG = "13f-top-holdings-vs-new-buys-python.png"
BG, FG, ACCENT, WARM, GREEN, GREY, RED = ("#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b",
                                          "#10b981", "#8a8a8a", "#ef4444")


def pull(mid, s, e, depth=0):
    """A single call returns at most 5,000 rows, so halve a window that fills up."""
    df = xfl.manager_holdings(mid, start=s, end=e)
    if len(df) >= 5000 and depth < 4:
        cut = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        return pd.concat([pull(mid, s, cut.strftime("%Y-%m-%d"), depth + 1),
                          pull(mid, (cut + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), e, depth + 1)],
                         ignore_index=True)
    return df


def fetch(**kw):
    """Retry a request that the connection drops mid-transfer."""
    for attempt in range(5):
        try:
            return xfl.prices(**kw)
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("price request failed after 5 attempts")


# ------------------------------------------------------------------ 13F books
books = []
for mid, nm in MANAGERS.items():
    df = pd.concat([p for p in (pull(mid, s, e) for s, e in CHUNKS) if len(p)], ignore_index=True)
    df["manager_name"] = nm
    books.append(df)
    print(f"  {nm:24s} {len(df):6d} rows  {df['report_date'].nunique():3d} quarters", flush=True)
bk = pd.concat(books, ignore_index=True)
bk["q"] = bk["report_date"].dt.to_period("Q")

# one issuer can be reported through two securities, so value is summed per entity
pos = bk.groupby(["manager_name", "q", "entity_id"], as_index=False)["value_usd"].sum()
pos["rank"] = pos.groupby(["manager_name", "q"])["value_usd"].rank(ascending=False, method="first")
held = {k: set(g["entity_id"]) for k, g in pos.groupby(["manager_name", "q"])}
pos["is_new"] = [(e not in held[(m, q - 1)]) if (m, q - 1) in held else False
                 for m, q, e in zip(pos["manager_name"], pos["q"], pos["entity_id"])]

QS = [q for q in sorted(pos["q"].unique()) if pd.Period(FIRST_Q) <= q <= pd.Period(LAST_Q)]
sel = pos[pos["q"].isin(QS) & ((pos["rank"] <= DEEP) | pos["is_new"])]

# ------------------------------------------- monthly total returns, per window
parts = []
for q in QS:
    m0 = q.asfreq("M", "end") + 2                     # month the filing deadline falls in
    ids = sorted(int(x) for x in sel.loc[sel["q"] == q, "entity_id"].unique())
    for i in range(0, len(ids), 400):
        p = fetch(entity_id=ids[i:i + 400], start=str((m0 + 1).start_time.date()),
                  end=str((m0 + 3).end_time.date()), interval="1mo",
                  fields=["return_daily"], max_rows=200000)
        parts.append(p[["entity_id", "date", "return_daily"]])
px = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["entity_id", "date"])
px["m"] = px["date"].dt.to_period("M")
ret = px.pivot_table(index="m", columns="entity_id", values="return_daily")

bench = {}
for t in ["SPY", "RSP"]:
    b = fetch(ticker=t, start="2013-01-01", end="2024-12-31", interval="1mo", fields=["return_daily"])
    bench[t] = b.set_index(b["date"].dt.to_period("M"))["return_daily"]


def forward(eids, months):
    """Quarter-ahead total return per position, equal weighted. A position that
    stops pricing inside the window is carried to its last observed month."""
    out = {}
    for e in eids:
        if e not in ret.columns:
            continue
        vals = [ret.at[m, e] if m in ret.index else np.nan for m in months]
        if pd.isna(vals[0]):
            continue
        g = 1.0
        for x in vals:
            if pd.isna(x):
                break
            g *= 1 + x
        out[e] = g - 1
    return out


rows, thin, unpriced = [], 0, 0
for q in QS:
    m0 = q.asfreq("M", "end") + 2
    months = [m0 + 1, m0 + 2, m0 + 3]
    b = {t: float((1 + s.reindex(months)).prod() - 1) if s.reindex(months).notna().all() else np.nan
         for t, s in bench.items()}
    for mgr, g in sel[sel["q"] == q].groupby("manager_name"):
        top = forward(g.loc[g["rank"] <= 10, "entity_id"], months)
        nxt = forward(g.loc[(g["rank"] > 10) & (g["rank"] <= DEEP), "entity_id"], months)
        new = forward(g.loc[g["is_new"], "entity_id"], months)
        unpriced += int((g["rank"] <= DEEP).sum()) - len(top) - len(nxt)
        if len(top) < 8 or len(nxt) < 8:
            thin += 1
            continue
        rows.append(dict(manager=mgr, q=q, spy=b["SPY"], rsp=b["RSP"],
                         top=np.mean(list(top.values())), nxt=np.mean(list(nxt.values())),
                         new=np.mean(list(new.values())) if len(new) >= 3 else np.nan,
                         n_top=len(top), n_nxt=len(nxt), n_new=len(new)))

pm = pd.DataFrame(rows)
qtr = pm.groupby("q")[["top", "nxt", "new", "spy", "rsp"]].mean()
cagr = lambda s: (1 + s).prod() ** (4 / len(s)) - 1

# ------------------------------------------------------------------ output
w = [("Ten largest positions", "top"), ("Positions 11-50", "nxt"),
     ("Positions opened that quarter", "new"), ("SPY", "spy")]
print(f"\nForm 13F books, {len(MANAGERS)} managers, {QS[0]} to {QS[-1]} ({len(QS)} quarter ends)")
print(f"Book rows read: {len(bk):,}   manager-quarters used: {len(pm)}   "
      f"below the depth screen: {thin}")
print(f"Holding windows: {qtr.index[0].asfreq('M','end')+3} to {qtr.index[-1].asfreq('M','end')+5}"
      f"   positions priced: {int(pm[['n_top','n_nxt']].sum().sum()):,}   unpriced: {unpriced}")
print(f"Positions per manager-quarter: top {pm['n_top'].mean():.1f}, next {pm['n_nxt'].mean():.1f}, "
      f"new {pm['n_new'].mean():.1f}")
print(f"Quarters with a complete return for every portfolio: {int(qtr[['top','nxt','new','spy']].notna().all(axis=1).sum())} of {len(qtr)}"
      f"   dates ordered: {qtr.index.is_monotonic_increasing}")
print(f"SPY quarterly window range: {qtr['spy'].min()*100:+.2f}% to {qtr['spy'].max()*100:+.2f}%")

print("\nEqual-weighted, rebuilt every quarter, held one quarter\n")
print(f"{'portfolio':32s}{'mean qtr':>10}{'CAGR':>8}{'$1 grew to':>12}{'vs SPY':>9}{'t':>7}{'p':>8}")
for name, col in w:
    s = qtr[col]
    d = s - qtr["spy"]
    t = stats.ttest_1samp(d, 0.0)
    gap = "" if col == "spy" else f"{(cagr(s)-cagr(qtr['spy']))*100:+8.2f}"
    tt = "" if col == "spy" else f"{t.statistic:7.2f}{t.pvalue:8.3f}"
    print(f"{name:32s}{s.mean()*100:9.2f}%{cagr(s)*100:7.2f}%{(1+s).prod():11.2f}x{gap}{tt}")

sp = qtr["top"] - qtr["nxt"]
nn = qtr["new"] - qtr["nxt"]
for label, d in [("Ten largest minus positions 11-50", sp), ("New positions minus positions 11-50", nn)]:
    t = stats.ttest_1samp(d, 0.0)
    b = stats.binomtest(int((d > 0).sum()), len(d), 0.5)
    print(f"\n{label}: {d.mean()*100:+.2f} pts per quarter (median {d.median()*100:+.2f}), "
          f"t={t.statistic:.2f}, p={t.pvalue:.3f}")
    print(f"  positive in {int((d>0).sum())} of {len(d)} quarters (sign test p={b.pvalue:.3f}), "
          f"worst {d.min()*100:+.1f} pts in {d.idxmin()}")

sub = qtr[qtr["rsp"].notna()]
print(f"\nAgainst an equal-weighted index fund, {sub.index[0]} to {sub.index[-1]} ({len(sub)} quarters), CAGR")
for name, col in [("Ten largest positions", "top"), ("Positions 11-50", "nxt"),
                  ("Positions opened that quarter", "new"), ("RSP, equal weight", "rsp"),
                  ("SPY, capitalisation weight", "spy")]:
    print(f"  {name:32s}{cagr(sub[col])*100:7.2f}%")

per = pm.groupby("manager").agg(n=("q", "nunique"), top=("top", "mean"), nxt=("nxt", "mean"))
per["top"] = ((1 + per["top"]) ** 4 - 1) * 100
per["nxt"] = ((1 + per["nxt"]) ** 4 - 1) * 100
per["spread"] = per["top"] - per["nxt"]
per = per.sort_values("spread", ascending=False)
print(f"\nPer manager, annualised: ten largest against positions 11-50 "
      f"({int((per['spread']>0).sum())} of {len(per)} positive)\n")
print(f"{'manager':26s}{'qtrs':>6}{'top 10':>9}{'11-50':>9}{'spread':>9}")
for m, r in per.iterrows():
    print(f"{m:26s}{int(r['n']):6d}{r['top']:8.2f}%{r['nxt']:8.2f}%{r['spread']:+8.2f}")

# ------------------------------------------------------------------ chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG,
                               gridspec_kw={"height_ratios": [1.3, 1]})
x = [q.asfreq("M", "end").to_timestamp() for q in qtr.index]
for col, lab, c in [("spy", "SPY", GREY), ("top", "Ten largest positions", ACCENT),
                    ("nxt", "Positions 11-50", WARM), ("new", "Positions opened that quarter", GREEN)]:
    ax1.plot(x, (1 + qtr[col]).cumprod(), color=c, lw=1.6 if col == "spy" else 2,
             ls="--" if col == "spy" else "-", label=lab)
ax1.set_facecolor(BG)
ax1.set_title("Copying a 13F: what one dollar became, 2013 to 2024", color=FG, fontsize=13, pad=10)
ax1.set_ylabel("Growth of $1", color=FG)
ax1.legend(facecolor=BG, edgecolor="#333", labelcolor=FG, fontsize=9, loc="upper left")

ax2.set_facecolor(BG)
ax2.barh(range(len(per))[::-1], per["spread"],
         color=[ACCENT if v > 0 else RED for v in per["spread"]])
ax2.set_yticks(range(len(per))[::-1])
ax2.set_yticklabels(per.index, fontsize=8)
ax2.axvline(0, color="#555", lw=1)
ax2.set_title("Ten largest positions minus positions 11-50, annualised points, by manager",
              color=FG, fontsize=11, pad=8)
for ax in (ax1, ax2):
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#333")
plt.tight_layout(h_pad=2.2)
plt.savefig(PNG, dpi=150, facecolor=BG)
print(f"\nchart written to {PNG}")
