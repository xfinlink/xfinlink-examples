# Full write-up: https://xfinlink.com/blog/stocks-removed-from-sp500-before-removal-python
#
# What happens to S&P 500 members before they are removed? Every removal in the dated
# log from 1995 to 2024 is followed by entity_id over its last 24 months in the index,
# and its total return is measured against SPY's. Removals are split by what the price
# series shows next: stopped trading within 5 trading days (mostly acquisitions) or
# still trading 63+ trading days later (demoted).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

DAYS, STEP = 504, 21          # 24 months of trading days; one month = 21 trading days
CHART = "stocks-removed-from-sp500-before-removal-python.png"

# Every S&P 500 addition and removal on record, paged 1,000 at a time
ev = pd.concat([xfl.index_events("sp500", end="2025-06-30", offset=o)
                for o in range(0, 4000, 1000)], ignore_index=True)
ev = ev.dropna(subset=["entity_id"]).drop_duplicates()
ev["eff"] = pd.to_datetime(ev["effective_date"])
ev["entity_id"] = ev["entity_id"].astype(int)
rem = ev[(ev["event_type"] == "removed") & ev["eff"].between("1995-01-01", "2024-12-31")]
adds = ev[ev["event_type"] == "added"]

spy = xfl.prices("SPY", start="1993-01-01", end="2025-06-30", fields=["return_daily"])
spy = spy.set_index("date")["return_daily"].fillna(0.0)
cal = spy.index

# Daily total returns for each year's removed members, followed by entity_id, not ticker
px = {}
for y, g in rem.groupby(rem["eff"].dt.year):
    start = (pd.Timestamp(f"{y}-01-01") - pd.DateOffset(months=26)).date().isoformat()
    end = min(pd.Timestamp(f"{y + 1}-04-30"), pd.Timestamp("2025-06-30")).date().isoformat()
    p = xfl.prices(entity_id=sorted(set(g["entity_id"])), start=start, end=end,
                   fields=["return_daily"], max_rows=250_000)
    for eid, s in p.dropna(subset=["return_daily"]).groupby("entity_id"):
        px[(y, eid)] = s.drop_duplicates("date").set_index("date")["return_daily"]

rows = []
for _, r in rem.iterrows():
    s = px.get((r["eff"].year, r["entity_id"]))
    joined = adds.loc[(adds["entity_id"] == r["entity_id"]) & (adds["eff"] <= r["eff"]), "eff"].max()
    if s is None or (pd.notna(joined) and joined > cal[cal.searchsorted(r["eff"], "right") - 1 - DAYS]):
        rows.append({"group": "member under 24 months"})
        continue
    k0 = cal.get_loc(s.index[s.index <= r["eff"]].max())   # last close as a member = month 0
    window = cal[k0 - DAYS + 1:k0 + 1]
    after = int((s.index > r["eff"]).sum())                # trading days after the removal
    group = ("stopped trading" if after <= 5 else
             "still trading" if after >= 63 else "stopped within 3 months")
    stock = (1 + s.reindex(window).fillna(0.0)).cumprod() - 1
    mkt = (1 + spy.loc[window]).cumprod() - 1
    adj = (stock - mkt).iloc[STEP - 1::STEP].values        # months -23 ... 0
    rows.append({"group": group, "eff": r["eff"], "ticker": r["ticker"], "name": r["entity_name"],
                 "entity_id": r["entity_id"], "raw": stock.iloc[-1], "spy": mkt.iloc[-1],
                 **{m: v for m, v in zip(range(-23, 1), adj)}, -24: 0.0})

res = pd.DataFrame(rows)
kept = res[res["group"] != "member under 24 months"].copy()
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))
kept["period"] = pd.cut(kept["eff"].dt.year, [1994, 2004, 2014, 2024],
                        labels=["1995-2004", "2005-2014", "2015-2024"])

# ---- output ------------------------------------------------------------------
n = res["group"].value_counts()
print(f"S&P 500 removals 1995-2024: {len(res)}")
print(f"  member for under 24 months before removal (left out): {n['member under 24 months']}")
print(f"  stopped trading within 5 trading days of removal:     {n['stopped trading']}")
print(f"  still trading 63+ trading days after removal:         {n['still trading']}")
print(f"  stopped trading 6-62 trading days after removal:      {n['stopped within 3 months']}")
print(f"  of the {len(kept)} measured, members of today's S&P 500: "
      f"{kept['entity_id'].isin(today).sum()}")

months = [-24, -18, -12, -9, -6, -3, -1, 0]
print("\nReturn minus SPY's since month -24 (total return, percentage points)")
print("month                    " + "".join(f"{m:>7}" for m in months))
for g in ["still trading", "stopped trading", "stopped within 3 months"]:
    k = kept[kept["group"] == g]
    for qn, q in [("25th pct", 0.25), ("median", 0.5), ("75th pct", 0.75)]:
        vals = k[months].quantile(q) * 100
        print(f"{g[:15]:<15} {qn:<8} " + "".join(f"{v:>7.1f}" for v in vals))

print()
for g in ["still trading", "stopped trading", "stopped within 3 months"]:
    k = kept[kept["group"] == g]
    print(f"{g} (n={len(k)}): own return median {k['raw'].median():+.1%}, SPY median "
          f"{k['spy'].median():+.1%}; behind SPY at month -12 {(k[-12] < 0).mean():.0%}, "
          f"at month 0 {(k[0] < 0).mean():.0%}; mean gap at month 0 {k[0].mean() * 100:+.1f} pts")
s = kept[kept["group"] == "stopped trading"]
print(f"stopped trading with an own 24-month loss beyond 80% (failures and rescues): "
      f"{(s['raw'] < -0.8).sum()} of {len(s)}")
k = kept[kept["group"] == "still trading"]
ratio = ((1 + k["raw"]) / (1 + k["spy"]) - 1).median()
print(f"still trading: median shortfall reached by month -12 {k[-12].median() / k[0].median():.0%}, "
      f"by month -6 {k[-6].median() / k[0].median():.0%}; "
      f"median relative wealth (1+stock)/(1+SPY)-1 {ratio:+.1%}")

print("\nBy period of removal: median gap to SPY at month -12 and month 0 (points)")
tab = (kept[kept["group"] != "stopped within 3 months"]
       .groupby(["period", "group"], observed=True)
       .agg(n=(0, "size"), m12=(-12, "median"), m0=(0, "median"), behind=(0, lambda x: (x < 0).mean())))
for (per, g), t in tab.iterrows():
    print(f"  {per}  {g:<16} n={int(t['n']):>3}  month -12 {t['m12'] * 100:+6.1f}  month 0 {t['m0'] * 100:+6.1f}  "
          f"behind SPY {t['behind']:.0%}")

def show(df, label):
    print(label)
    for _, t in df.iterrows():
        print(f"  {t['eff']:%Y-%m-%d}  {t['ticker']:<5} {t['name'][:30]:<30} own {t['raw']:+7.1%}  "
              f"SPY {t['spy']:+7.1%}  gap {t[0] * 100:+6.1f} pts")

print()
show(k.nsmallest(5, 0), "Still trading, five widest gaps")
show(k.nlargest(3, 0), "Still trading, three that beat SPY by most")
checks = [("M", "2020-04-03"), ("KSS", "2020-09-18"), ("JCP", "2013-11-29"),
          ("RTN", "2020-04-03"), ("WFM", "2017-08-28"), ("LEH", "2008-09-16"),
          ("AAL", "1997-01-14"), ("AAL", "2024-09-20")]
sel = pd.concat([kept[(kept["ticker"] == t) & (kept["eff"] == d)] for t, d in checks])
show(sel, "Named checks (AAL appears twice: two different companies)")
for _, t in sel.iterrows():
    print(f"  {t['ticker']:<4} {t['eff']:%Y}: {t['group']}")

# ---- chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#a3a3a3", "ytick.color": "#a3a3a3", "axes.edgecolor": "#404040"})
fig, ax = plt.subplots(figsize=(10, 5))
x = list(range(-24, 1))
for g, colour, label in [("still trading", "#3b82f6", "Still trading after removal (demoted)"),
                         ("stopped trading", "#d97706", "Stopped trading at removal (mostly acquired)")]:
    q = kept.loc[kept["group"] == g, x].quantile([0.25, 0.5, 0.75]) * 100
    ax.fill_between(x, q.loc[0.25], q.loc[0.75], color=colour, alpha=0.18, linewidth=0)
    ax.plot(x, q.loc[0.5], color=colour, linewidth=2, label=f"{label}, n={(kept['group'] == g).sum()}")
    ax.annotate(f"{q.loc[0.5, 0]:+.1f}", (0, q.loc[0.5, 0]), xytext=(6, 0),
                textcoords="offset points", va="center", color="#e0e0e0", fontsize=10)
ax.axhline(0, color="#737373", linewidth=0.8)
ax.set_xlim(-24, 1.8)
ax.set_xticks(range(-24, 1, 3))
ax.set_xlabel("Months before the last close as an S&P 500 member")
ax.set_ylabel("Total return minus SPY since month -24 (points)")
ax.set_title("Stocks removed from the S&P 500: the 24 months before removal, 1995-2024\n"
             "median, with 25th-75th percentile band", fontsize=12)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.savefig(CHART, dpi=150)
