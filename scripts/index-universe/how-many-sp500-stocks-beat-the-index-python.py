# Full write-up: https://xfinlink.com/blog/how-many-sp500-stocks-beat-the-index-python
#
# How many S&P 500 stocks beat the index they belonged to?
# Every member of the roster in force on the first trading day of 2000 is bought and held
# from January 2000 to December 2024 on monthly total returns. A holding ends earlier when the
# stock leaves trading (a recorded delisting return is applied in the exit month) or when its
# monthly series first pauses; it is not refilled if the series resumes later. "Beat the index"
# means the stock's compounded total return over its own months exceeds SPY's over the same
# months. The same statistics are computed for a list built from survivors (today's members
# that traded in January 2000), and for ten-year windows starting each January 1995-2015.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# SPY's monthly total return, and the first and last trading day of each month
spy = xfl.prices("SPY", start="1993-01-01", end="2024-12-31", fields=["return_daily"])
spy["date"] = pd.to_datetime(spy["date"])
mo = spy["date"].dt.to_period("M")
spy_m = spy.groupby(mo)["return_daily"].apply(lambda r: (1 + r.fillna(0)).prod() - 1)
first_day, last_day = spy.groupby(mo)["date"].min(), spy.groupby(mo)["date"].max()


def roster(d):
    r = xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).dropna(subset=["entity_id"])
    return r.assign(entity_id=r["entity_id"].astype(int))


# Rosters on the first trading day of each January, and on the last trading day of each
# window's final December
starts = list(range(1995, 2016))
start_ros = {y: roster(first_day[pd.Period(f"{y}-01", "M")]) for y in sorted(set(starts) | {2000})}
end_ros = {y: roster(last_day[pd.Period(f"{y}-12", "M")])
           for y in sorted({y + 9 for y in starts} | {2024})}

# Monthly total returns for every security on any of those rosters, plus recorded exits
ids = sorted(set().union(*[set(r["entity_id"]) for r in [*start_ros.values(), *end_ros.values()]]))
parts, exits = [], {}
for i in range(0, len(ids), 13):   # 13 ids x 360 months fit on one 5,000-row page, so each
    p = xfl.prices(entity_id=ids[i:i + 13], start="1995-01-01", end="2024-12-31",   # page's
                   interval="1mo", fields=["close", "return_daily"],               # exits
                   max_rows=100_000)                                               # are kept
    assert len(p) < 5000, "batch spans more than one page"
    parts.append(p)
    exits.update(p.attrs.get("meta", {}).get("delisting", {}) or {})
px = pd.concat(parts)
px["month"] = pd.to_datetime(px["date"]).dt.to_period("M")
ret = px.pivot_table(index="month", columns="entity_id", values="return_daily", aggfunc="first")
ret = ret.reindex(pd.period_range("1995-01", "2024-12", freq="M"))
exit_month = {}
for eid, ev in exits.items():      # recorded delisting return, added in the exit month
    eid, dl, m = int(eid), ev.get("delisting_return"), pd.Period(ev["event_date"], "M")
    if eid not in ret.columns or m not in ret.index:
        continue
    exit_month[eid] = (m, dl)
    if dl is not None:
        r0 = ret.at[m, eid]
        ret.at[m, eid] = (1 + (0 if pd.isna(r0) else r0)) * (1 + dl) - 1
names = px.drop_duplicates("entity_id", keep="last").set_index("entity_id")["entity_name"]


def hold(ids, m0, m1):
    """Buy each id at the start of m0; hold to m1 or to the month before its series first stops."""
    out = []
    for e in ids:
        s = ret.loc[m0:m1, e] if e in ret.columns else pd.Series(np.nan, ret.loc[m0:m1].index)
        ok = s.notna().to_numpy()
        if not ok[0]:
            out.append({"entity_id": e, "held": False})
            continue
        stop = len(s) if ok.all() else int(np.argmin(ok))
        months = s.index[:stop]
        out.append({"entity_id": e, "held": True, "last": months[-1], "full": stop == len(s),
                    "paused": bool(ok[stop:].any()),          # series resumes after the stop
                    "tr": (1 + s.iloc[:stop]).prod() - 1,
                    "spy": (1 + spy_m.loc[months]).prod() - 1})
    return pd.DataFrame(out)


def summary(c):
    h = c[c["held"]]
    gain = h["tr"].sort_values(ascending=False)
    return {"members": len(c), "held": len(h), "beat SPY %": (h["tr"] > h["spy"]).mean() * 100,
            "lost money %": (h["tr"] < 0).mean() * 100,
            "median total return %": h["tr"].median() * 100,
            "mean total return %": h["tr"].mean() * 100,
            "top 10 share of gain %": gain.iloc[:10].sum() / gain.sum() * 100,
            "names for half the gain": int((gain.cumsum() / gain.sum() < 0.5).sum() + 1)}


m0, m1 = pd.Period("2000-01", "M"), pd.Period("2024-12", "M")
traded_jan = lambda r, m: [e for e in r["entity_id"] if e in ret.columns and pd.notna(ret.at[m, e])]
pit = hold(sorted(start_ros[2000]["entity_id"]), m0, m1)
surv = hold(sorted(traded_jan(end_ros[2024], m0)), m0, m1)
print(f"Roster on {first_day[m0].date()}: {len(start_ros[2000])} securities; "
      f"roster on {last_day[m1].date()}: {len(end_ros[2024])}, of which {len(surv)} traded in 2000-01 "
      f"({len(set(surv['entity_id']) - set(start_ros[2000]['entity_id']))} of them not members in 2000)")
print(f"SPY 2000-01 to 2024-12: {(1 + spy_m.loc[m0:m1]).prod() * 100 - 100:.1f}% total, "
      f"{((1 + spy_m.loc[m0:m1]).prod() ** (12 / 300) - 1) * 100:.2f}% a year\n")
print(pd.DataFrame({"point-in-time 2000 roster": summary(pit),
                    "today's members, traded in 2000": summary(surv)}).round(1).to_string())

h = pit[pit["held"]].copy()
ex = h[~h["full"].astype(bool)]
paused = ex["paused"].astype(bool)
rec = ex.apply(lambda r: r["entity_id"] in exit_month and exit_month[r["entity_id"]][1] is not None
               and exit_month[r["entity_id"]][0] == r["last"], axis=1) & ~paused
print(f"\nPoint-in-time cohort: {len(ex)} holdings ended before 2024-12; {rec.sum()} with a recorded "
      f"delisting return in the exit month, {(~rec & ~paused).sum()} valued at the last exchange close, "
      f"{h['paused'].sum()} stopped where the monthly series pauses and later resumes")
h["name"], h["last"] = h["entity_id"].map(names), h["last"].astype(str)
h["vs SPY"] = (1 + h["tr"]) / (1 + h["spy"]) - 1
cols = ["name", "last", "tr", "spy", "vs SPY"]
print("Largest gains:\n" + h.nlargest(5, "tr")[cols].round(3).to_string(index=False))
print("Largest losses:\n" + h.nsmallest(5, "tr")[cols].round(4).to_string(index=False) + "\n")

# Ten-year windows: point-in-time roster vs a list built from the window's survivors
rows = []
for y in starts:
    a, b = pd.Period(f"{y}-01", "M"), pd.Period(f"{y + 9}-12", "M")
    p = summary(hold(sorted(start_ros[y]["entity_id"]), a, b))
    s = summary(hold(sorted(traded_jan(end_ros[y + 9], a)), a, b))
    rows.append({"window": f"{y}-{y + 9}", "point-in-time n": p["held"],
                 "point-in-time beat %": p["beat SPY %"], "survivors n": s["held"],
                 "survivors beat %": s["beat SPY %"],
                 "gap, points": s["beat SPY %"] - p["beat SPY %"],
                 "SPY total %": ((1 + spy_m.loc[a:b]).prod() - 1) * 100})
win = pd.DataFrame(rows).set_index("window")
print("Ten-year windows, share of names that beat SPY over their own months")
print(win.round(1).to_string())
print(f"Point-in-time beat share: {win['point-in-time beat %'].min():.1f}% to "
      f"{win['point-in-time beat %'].max():.1f}%, below 50% in "
      f"{(win['point-in-time beat %'] < 50).sum()} of {len(win)} windows; survivor list higher "
      f"in {(win['gap, points'] > 0).sum()} of {len(win)}, by {win['gap, points'].min():.1f} "
      f"to {win['gap, points'].max():.1f} points (median {win['gap, points'].median():.1f})")

# Chart: beat share by start year, and the 2000 cohort's returns relative to SPY
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.array(starts)
for col, label, color in [("survivors beat %", "Survivor list", "#d97706"),
                          ("point-in-time beat %", "Point-in-time roster", "#3b82f6")]:
    ax1.plot(x, win[col], color=color, lw=2, marker="o", ms=4, label=label)
ax1.axhline(50, color="#666666", lw=1, ls="--")
ax1.set_xticks(x[::2])
ax1.set_ylim(20, 80)
ax1.set_xlabel("First year of the ten-year window")
ax1.set_ylabel("Members that beat SPY (%)")
ax1.set_title("Ten-year windows, 1995-2024", fontsize=11)
ax1.legend(frameon=False, loc="lower left")
bins = np.linspace(-6, 6, 25)
for c, label, color in [(surv, "Survivor list", "#d97706"), (pit, "Point-in-time roster", "#3b82f6")]:
    k = c[c["held"]]
    rel = np.clip(np.log2((1 + k["tr"]) / (1 + k["spy"])), -5.99, 5.99)
    ax2.hist(rel, bins=bins, histtype="step", lw=2, color=color, label=label)
ax2.axvline(0, color="#666666", lw=1, ls="--")
ax2.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax2.set_xticklabels(["\u22641/64", "1/16", "1/4", "1x", "4x", "16x", "64x"])
ax2.set_xlabel("Ending wealth relative to SPY, same months (log scale)")
ax2.set_ylabel("Number of stocks")
ax2.set_title("2000 roster held to 2024", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("How Many S&P 500 Stocks Beat the Index?", fontsize=13)
plt.tight_layout()
plt.savefig("how-many-sp500-stocks-beat-the-index-python.png", dpi=150)
