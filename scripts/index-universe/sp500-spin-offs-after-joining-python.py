# Full write-up: https://xfinlink.com/blog/sp500-spin-offs-after-joining-python
#
# How did spin-offs do after they joined the S&P 500?
# Every spin-off addition effective February 1993 to 2021, followed for 12, 24 and 36 months
# against SPY, on the dated additions log rather than today's member list.

import warnings

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose

START, END, LAST_DAY = "1993-02-01", "2021-12-31", "2024-12-31"  # SPY trades from 29 January 1993
HORIZONS = {"12m": 252, "24m": 504, "36m": 756}                  # trading days

# Route classification from https://xfinlink.com/blog/how-long-after-ipo-sp500-python
# Default: a first entry with under 0.1 years of trading is a spin-off. Hand overrides by ticker:year.
SECOND_CLASS = {10003590, 10032892, 10006698, 10003102, 10030753, 10004763, 10029970}
ROUTE = {
    "carve-out IPO": """SFR:1990 KRB:1992 DWD:1993 ATI:1994 FDC:1994 GLD:1994 ALL:1995 CSE:1996
        LU:1996 UPR:1996 GDT:1996 AFS:1998 COF:1998 DPH:1999 COC:1999 TSG:2000 A:2000 CVG:2000
        CIT:2000 PALM:2000 MIR:2001 PBG:2001 MON:2002 TAPB:2002 COH:2004 CIT:2004 FSLB:2004
        GNW:2005 AIZ:2007 MJN:2009 ZTS:2013 SYF:2015 CFG:2016 KVUE:2023""",
    "spin-off": """AGN:1992 LEH:1998 NCR:2000 CNXT:2000 RJR:2002 EXPE:2007 DPS:2008 DISCA:2010
        MNK:2014 FBHS:2016 KEYS:2018 LW:2018 SNDK:2025""",
    "other": "FRE:1992 VIA:1994 NDAQ:2008 TWC:2009 DVMT:2016",  # DVMT: tracking stock issued in the EMC merger
    "reorganisation": """CVN:1990 X:1991 TMC:1995 FCX:1995 UMG:1995 PNU:1995 FD:1995 GRA:1996
        ROK:1996 TEN:1996 SEE:1998 MAR:1998 UMG:1998 SRE:1998 EDS:1998 PCS:1998 MCK:1999 WLP:1999
        DVN:2000 DYN:2000 AET:2000 FLR:2000 XL:2001 CMCSA:2002 NWSA:2004 SHLD:2005 HOT:2006
        DTV:2006 NYX:2007 IVZ:2008 ESV:2012 LYB:2012 PNR:2012 KHC:2015 FTI:2017 DWDP:2017
        LIN:2018 AMCR:2019 VTRS:2020 WBD:2022 KDP:2022 SW:2024 TPL:2024 EXE:2025 PSKY:2025 CRH:2025""",
}
ROUTE_OF = {k: r for r, keys in ROUTE.items() for k in keys.split()}


def events(kind):
    pages = [xfl.index_events("sp500", event_type=kind, start="1925-12-31")]
    while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
        pages.append(xfl.index_events("sp500", event_type=kind, start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True).dropna(subset=["entity_id"]).astype({"entity_id": int})


def bars(ids, start, end, interval):
    fields = ["close"] if interval == "1w" else ["close", "return_daily"]
    return pd.concat([xfl.prices(entity_id=ids[i:i + 150], start=start, end=end, interval=interval,
                                 fields=fields, max_rows=400_000) for i in range(0, len(ids), 150)])


adds, rems = events("added"), events("removed")

# A stay under 30 days is a placement: S&P adds a member's spin-off and drops it when it does not fit
adds["last_day"] = [rems.loc[(rems["entity_id"] == r.entity_id) & (rems["effective_date"] >= r.effective_date),
                             "effective_date"].min() for r in adds.itertuples()]
adds["placement"] = (pd.to_datetime(adds["last_day"]) - pd.to_datetime(adds["effective_date"])).dt.days < 30
placement_exits = set(zip(adds.loc[adds["placement"], "entity_id"], adds.loc[adds["placement"], "last_day"]))
real_adds = adds[~adds["placement"]]
real_exits = rems[[(e, d) not in placement_exits for e, d in zip(rems["entity_id"], rems["effective_date"])]]
w = adds[adds["effective_date"].between(START, END) & ~adds["entity_id"].isin(SECOND_CLASS)].copy()
w["returning"] = [((real_adds["entity_id"] == r.entity_id) & (real_adds["effective_date"] < r.effective_date)).any()
                  or ((real_exits["entity_id"] == r.entity_id) & (real_exits["effective_date"] < r.effective_date)).any()
                  for r in w.itertuples()]
w = w[w["placement"] | ~w["returning"]]                   # first entries, plus placements
w["year"] = w["effective_date"].str[:4].astype(int)
w["key"] = w["ticker"] + ":" + w["year"].astype(str)      # ticker in force at the addition

# Screen on weekly bars, then date the first trading day of each candidate on daily bars
first_week = {}
for y, g in w.groupby("year"):
    wk = bars(sorted(set(g["entity_id"])), f"{y - 1}-10-23", f"{y}-12-31", "1w")
    first_week.update(wk.dropna(subset=["close"]).groupby("entity_id")["date"].min())
w["age_wk"] = (pd.to_datetime(w["effective_date"]) - w["entity_id"].map(first_week)).dt.days
cand = w[(w["age_wk"] < 60) | w["key"].isin(ROUTE_OF)]
daily = pd.concat([bars(sorted(set(g["entity_id"])), f"{y - 1}-10-23", min(f"{y + 3}-12-31", LAST_DAY), "1d")
                   for y, g in cand.groupby("year")]).drop_duplicates(["entity_id", "date"])
daily = daily.dropna(subset=["close"]).sort_values(["entity_id", "date"])
gap = daily.groupby("entity_id")["close"].pct_change()  # a long run with no close leaves return_daily null: bridge it
daily["return_daily"] = daily["return_daily"].fillna(gap)
first_bar = daily.groupby("entity_id")["date"].min()
cand = cand.assign(years=((pd.to_datetime(cand["effective_date"]) - cand["entity_id"].map(first_bar)).dt.days / 365.25).clip(lower=0))
cand["route"] = [ROUTE_OF.get(k, "spin-off" if y < 0.1 else "older listing") for k, y in zip(cand["key"], cand["years"])]
spin = cand[cand["route"] == "spin-off"].copy()
spin["group"] = np.where(spin["placement"], "placed", "kept")

spy = xfl.prices("SPY", start="1993-01-01", end=LAST_DAY, fields=["return_daily"]).set_index("date")["return_daily"].fillna(0.0)
cal = spy.index
end24 = xfl.index("sp500", as_of="2024-12-31").dropna(subset=["entity_id"])
still_in = set(zip(end24["entity_id"].astype(int), end24["added_date"].astype(str).str[:10]))  # same membership spell at end-2024

rows = []
for r in spin.itertuples():
    s = daily[daily["entity_id"] == r.entity_id].set_index("date")
    t0 = s.index[s.index >= r.effective_date].min()    # first close as a member (or first bar if later)
    k0, last = cal.get_loc(t0), s.index.max()
    row = {"ticker": r.ticker, "effective_date": r.effective_date, "group": r.group, "last_bar": last.date(),
           "survivor": (r.entity_id, r.effective_date) in still_in}
    for h, n in HORIZONS.items():
        days = cal[k0 + 1:k0 + n + 1]
        alive = days <= last
        stock = s["return_daily"].reindex(days).fillna(0.0).where(alive, spy.reindex(days))  # after the last close: SPY
        row[h] = np.prod(1 + stock) - np.prod(1 + spy.reindex(days))                         # total return minus SPY's
        row[h + "_stopped"] = not alive.all()
    rows.append(row)
res = pd.DataFrame(rows)
kept, placed = res[res["group"] == "kept"], res[res["group"] == "placed"]


def summary(x, label):
    cells = [f"{x[h].median():+7.1%} {x[h].mean():+7.1%} {(x[h] > 0).mean():4.0%} {stats.ttest_1samp(x[h], 0).statistic:+5.2f}"
             for h in HORIZONS]
    print(f"{label:<44}{len(x):>4}  " + "   ".join(cells))


print(f"Spin-off additions effective {START} to {END}: {len(spin)} ({len(kept)} kept, {len(placed)} placed and removed within days)")
print(f"  joined after their first month of trading: {(spin['years'] >= 0.1).sum()}; "
      f"first bar after the effective date: {(spin['entity_id'].map(first_bar) > pd.to_datetime(spin['effective_date'])).sum()}")
print(f"  kept spin-offs whose shares stopped trading within 36 months: {kept['36m_stopped'].sum()}; "
      f"still in the index at end-2024 (same membership): {kept['survivor'].sum()}\n")
print("Total return minus SPY's from the first close as a member (percentage points)")
print(f"{'':<50}" + "   ".join(f"{h + ' months':^26}" for h in ("12", "24", "36")))
print(f"{'':<44}{'n':>4}  " + "   ".join(f"{'median':>7} {'mean':>7} {'beat':>4} {'t':>5}" for _ in HORIZONS))
summary(kept, "Kept, point-in-time")
summary(kept[kept["survivor"]], "Kept, still in the index at end-2024 only")
summary(kept[~kept["survivor"]], "Kept, no longer in the index at end-2024")
summary(placed, "Placed and removed within days")
for h in HORIZONS:
    t = stats.ttest_ind(kept.loc[kept["survivor"], h], kept.loc[~kept["survivor"], h], equal_var=False)
    print(f"  {h}: survivors minus the rest, mean gap {kept.loc[kept['survivor'], h].mean() - kept.loc[~kept['survivor'], h].mean():+.1%} "
          f"(Welch t = {t.statistic:+.2f}, p = {t.pvalue:.2f})")

print("\nKept spin-offs by period of addition, 36 months")
for lo, hi in [("1993", "2001"), ("2002", "2011"), ("2012", "2021")]:
    x = kept[kept["effective_date"].str[:4].between(lo, hi)]
    print(f"  {lo}-{hi}  n={len(x):>2}  median {x['36m'].median():+6.1%}  beat SPY {(x['36m'] > 0).mean():.0%}  "
          f"still in the index {x['survivor'].sum()}")
print("\nPlaced spin-offs, 2015-2020:")
print(placed[["effective_date", "ticker", "12m", "36m"]].round(3).to_string(index=False))
print("\nKept spin-offs, best and worst 36 months:")
ranked = kept.sort_values("36m")
print(pd.concat([ranked.tail(3), ranked.head(3)])[["effective_date", "ticker", "survivor", "36m"]].round(3).to_string(index=False))

# ---- chart: median return minus SPY's by horizon and group ----
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444"})
fig, ax = plt.subplots(figsize=(10, 5))
months = [0, 12, 24, 36]
lines = [(f"Kept, still in the index at end-2024 ({kept['survivor'].sum()})", kept[kept["survivor"]], "#93c5fd", "--"),
         (f"Kept, all point-in-time ({len(kept)})", kept, "#3b82f6", "-"),
         (f"Placed and removed within days, 2015-2020 ({len(placed)})", placed, "#9ca3af", "-")]
for label, x, c, ls in lines:
    ys = [0] + [x[h].median() * 100 for h in HORIZONS]
    ax.plot(months, ys, color=c, ls=ls, lw=2.2, marker="o", label=label)
    ax.annotate(f"{ys[-1]:+.1f}", (36, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=c)
ax.axhline(0, color="#666666", lw=1)
ax.set_xticks(months)
ax.set_xlim(-1, 40)
ax.set_xlabel("Months after the first close as a member")
ax.set_ylabel("Median total return minus SPY (points)")
ax.set_title("S&P 500 spin-off additions vs SPY, 1993-2021")
ax.legend(frameon=False, loc="lower left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("sp500-spin-offs-after-joining-python.png", dpi=150)
