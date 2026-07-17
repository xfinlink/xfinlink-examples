# Full write-up: https://xfinlink.com/blog/growth-screen-survivorship-bias-python
#
# How much of a growth screen's backtested edge is survivorship bias?
# Runs one identical top-decile revenue-growth screen over two universes:
#   1. POINT-IN-TIME  : S&P 500 membership as it stood on 2021-06-30
#   2. CONTAMINATED   : S&P 500 membership as it stands today
# Formation 2021-06-30, held to 2026-07-16. The gap is the survivorship premium.
#
# Built from SEC EDGAR public filings and market data.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM = "2021-06-30"   # portfolio formation date
END = "2026-07-16"    # last available price date
LAG_CUTOFF = "2021-03-31"  # 90-day reporting lag: only fiscal years ending
                           # on/before this are public knowledge at formation

# ---------------------------------------------------------------- universes
pit = xfl.index("sp500", as_of=FORM)   # membership AS IT STOOD in June 2021
now = xfl.index("sp500")               # membership today
pit_e, now_e = set(pit["entity_id"]), set(now["entity_id"])
removed_e, added_e = pit_e - now_e, now_e - pit_e

lookup = pd.concat([pit, now]).drop_duplicates("entity_id").set_index("entity_id")
print(f"S&P 500 members on {FORM}: {len(pit)}")
print(f"S&P 500 members on {END}: {len(now)}")
print(f"In the 2021 index, gone today: {len(removed_e)}")
print(f"In today's index, absent in 2021: {len(added_e)}")

# ------------------------------------------------- what happened to the leavers
# Key on entity_id, not ticker: a rename (GPS -> GAP) is not a termination.
tick_rm = sorted(lookup.loc[sorted(removed_e), "ticker"])
rm_px = pd.concat([xfl.prices(tick_rm[i:i+20], start="2021-06-25", end=END, fields=["close"])
                   for i in range(0, len(tick_rm), 20)], ignore_index=True)
rm_px["date"] = pd.to_datetime(rm_px["date"])
rm_px = rm_px[rm_px["entity_id"].isin(removed_e)]
last_px = rm_px.groupby("entity_id")["date"].max()
still = (last_px >= pd.Timestamp("2026-07-10")).sum()
print(f"  of those {len(removed_e)}: {still} still have a live price series, "
      f"{len(last_px) - still} stopped trading (acquired, taken private or failed),")
print(f"  {len(removed_e) - len(last_px)} unreachable by ticker (recycled to an unrelated company)")

# ------------------------------------------------------------------ the screen
# Revenue growth from the last fiscal year that was public at formation.
allt = sorted(set(pit["ticker"]) | set(now["ticker"]))
fund = pd.concat([xfl.fundamentals(allt[i:i+40], period_type="annual", fields=["revenue"],
                                   start="2018-01-01", end=LAG_CUTOFF)
                  for i in range(0, len(allt), 40)], ignore_index=True)

rows = []
for eid, g in fund[fund["revenue"].notna()].sort_values("period_end").groupby("entity_id"):
    g = g.drop_duplicates("period_end", keep="last")
    if len(g) < 2:
        continue
    t0, t1 = g.iloc[-1], g.iloc[-2]
    gap = (t0["period_end"] - t1["period_end"]).days
    if not (300 <= gap <= 430) or t1["revenue"] <= 0:
        continue
    rows.append({"entity_id": eid, "sector": t0["gics_sector"],
                 "growth": t0["revenue"] / t1["revenue"] - 1})
gr = pd.DataFrame(rows)

def screen(universe):
    g = gr[gr["entity_id"].isin(universe)]
    return set(g.nlargest(int(round(len(g) * 0.10)), "growth")["entity_id"])

s_pit, s_now = screen(pit_e), screen(now_e)
print(f"\nScreenable (revenue history available): "
      f"point-in-time {len(gr[gr.entity_id.isin(pit_e)])}/{len(pit_e)}, "
      f"today {len(gr[gr.entity_id.isin(now_e)])}/{len(now_e)}")
print(f"Naive screen picks that were NOT index members at formation: "
      f"{len(s_now - pit_e)}/{len(s_now)}")
for e in sorted(s_now - pit_e, key=lambda x: -gr.set_index('entity_id').loc[x, 'growth'])[:6]:
    print(f"    {lookup.loc[e,'ticker']:5s} {lookup.loc[e,'entity_name'][:28]:30s} "
          f"FY growth {gr.set_index('entity_id').loc[e,'growth']:+.0%}")

# ------------------------------------------------------------ forward returns
# RULE FOR LEAVERS: a name is held from formation until its last available
# price. If it stops trading (acquisition or failure), the proceeds are
# redeployed equally across the names still trading. Nothing is dropped.
tick_scr = sorted(lookup.loc[sorted(s_pit | s_now), "ticker"])
px = pd.concat([xfl.prices(tick_scr[i:i+25], start="2021-06-25", end=END,
                           fields=["close", "return_daily"])
                for i in range(0, len(tick_scr), 25)], ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
px = px[px["entity_id"].isin(s_pit | s_now)]
investable = set(px[px["date"] == FORM]["entity_id"])  # must be buyable at formation
wide = px[(px["date"] > FORM) & (px["date"] <= END)].pivot_table(
    index="date", columns="entity_id", values="return_daily")

def backtest(ids):
    cols = [c for c in wide.columns if c in ids & investable]
    r = wide[cols].mean(axis=1, skipna=True)  # equal weight across live names
    cum = (1 + r.fillna(0)).cumprod()
    yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    return cum, cum.iloc[-1] ** (1 / yrs) - 1, len(cols)

spy = xfl.prices("SPY", start="2021-06-25", end=END, fields=["return_daily"])
spy["date"] = pd.to_datetime(spy["date"])
spy = spy[(spy["date"] > FORM) & (spy["date"] <= END)].sort_values("date")
cum_spy = (1 + spy.set_index("date")["return_daily"].fillna(0)).cumprod()
yrs = (cum_spy.index[-1] - cum_spy.index[0]).days / 365.25

cum_pit, cagr_pit, n_pit = backtest(s_pit)
cum_now, cagr_now, n_now = backtest(s_now)
print(f"\n{'':38s}{'total':>9s}{'CAGR':>9s}")
for lbl, ids in [("Honest screen (point-in-time)", s_pit), ("Naive screen (today's index)", s_now),
                 ("  picks common to both", s_pit & s_now),
                 ("  honest-screen-only picks", s_pit - s_now),
                 ("  naive-screen-only picks", s_now - s_pit)]:
    c, cagr, n = backtest(ids)
    print(f"{lbl:33s} n={n:<3d}{c.iloc[-1]-1:>9.2%}{cagr:>9.2%}")
print(f"{'S&P 500 ETF (SPY)':33s} n=1  {cum_spy.iloc[-1]-1:>9.2%}"
      f"{cum_spy.iloc[-1]**(1/yrs)-1:>9.2%}")
print(f"\nSURVIVORSHIP PREMIUM: {cum_now.iloc[-1]-cum_pit.iloc[-1]:+.2%} over the window, "
      f"{(cagr_now-cagr_pit)*1e4:+.0f} bp per year")

# ----------------------------------------------------------- sector churn
# Sector labels from fundamentals cover only firms with revenue history, which
# under-counts the leavers. Backfill from the price pull, then from resolve(),
# so the churn table reflects the whole index rather than the survivors.
sec = (fund.dropna(subset=["gics_sector"]).drop_duplicates("entity_id")
       .set_index("entity_id")["gics_sector"].to_dict())
for e, s in (rm_px.dropna(subset=["gics_sector"]).drop_duplicates("entity_id")
             .set_index("entity_id")["gics_sector"].items()):
    sec.setdefault(e, s)
gap_e = [e for e in (pit_e | now_e) if e not in sec]
gap_t = sorted(set(lookup.loc[gap_e, "ticker"]))
for i in range(0, len(gap_t), 10):          # resolve() accepts at most 10 tickers
    for v in xfl.resolve(gap_t[i:i+10], include=["classifications"])["data"].values():
        for ent in v["entities"]:           # recycled tickers return several entities
            s = (ent.get("classifications") or {}).get("gics_sector")
            if ent["entity_id"] in gap_e and s:
                sec[ent["entity_id"]] = s

cov = lambda s: sum(1 for e in s if sec.get(e))
print(f"\nSector labels resolved: members {cov(pit_e)}/{len(pit_e)}, "
      f"removed {cov(removed_e)}/{len(removed_e)}, added {cov(added_e)}/{len(added_e)}")
tab = pd.DataFrame({
    "in_2021": pd.Series({e: sec.get(e) for e in pit_e}).value_counts(),
    "removed": pd.Series({e: sec.get(e) for e in removed_e}).value_counts(),
    "added": pd.Series({e: sec.get(e) for e in added_e}).value_counts()}).fillna(0)
tab["churn_rate"] = (tab["removed"] + tab["added"]) / tab["in_2021"]
tab = tab.sort_values("churn_rate", ascending=False)
print("\nIndex churn by sector:\n", tab.to_string())

# ----------------------------------------------------------------- chart
BG, FG, ACC, RED, GREY = "#0a0a0a", "#e0e0e0", "#3b82f6", "#ef4444", "#6b7280"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1.55, 1])
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#333333")

ax1.plot(cum_now.index, (cum_now - 1) * 100, color=RED, lw=1.9,
         label="Naive screen: today's index members (survivorship-contaminated)")
ax1.plot(cum_pit.index, (cum_pit - 1) * 100, color=ACC, lw=1.9,
         label="Honest screen: index members as of 30 Jun 2021 (point-in-time)")
ax1.plot(cum_spy.index, (cum_spy - 1) * 100, color=GREY, lw=1.4, ls="--", label="S&P 500 ETF (SPY)")
ax1.fill_between(cum_pit.index, (cum_pit - 1) * 100, (cum_now - 1) * 100, color=RED, alpha=0.10)
ax1.axhline(0, color="#333333", lw=0.8)
ax1.set_ylabel("Cumulative total return (%)", color=FG, fontsize=10)
ax1.set_title("How much of a growth screen's edge is survivorship bias?\n"
              "Same top-decile revenue-growth screen, formed 30 Jun 2021, two universes",
              color=FG, fontsize=12, pad=12)
ax1.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, fontsize=8.5, loc="upper left")
ax1.text(0.40, 0.60, f"survivorship premium\n{(cum_now.iloc[-1]-cum_pit.iloc[-1])*100:.0f} points "
         f"over 5 years\n({(cagr_now-cagr_pit)*1e4:.0f} bp per year)",
         transform=ax1.transAxes, color=RED, fontsize=9.5, ha="center", va="center")

s = tab.sort_values("churn_rate")
ypos = range(len(s))
ax2.barh(ypos, s["churn_rate"] * 100, color=[ACC if v < 0.30 else RED for v in s["churn_rate"]])
ax2.set_yticks(list(ypos)); ax2.set_yticklabels(s.index, fontsize=8.5)
ax2.set_xlabel("Index membership churn since Jun 2021: names added or removed "
               "(% of 2021 sector count)", color=FG, fontsize=9)
ax2.set_title("Where the bias hides: churn by sector", color=FG, fontsize=10, pad=8)
for i, v in zip(ypos, s["churn_rate"] * 100):
    ax2.text(v + 0.6, i, f"{v:.0f}%", color=FG, fontsize=8, va="center")
ax2.set_xlim(0, s["churn_rate"].max() * 118)
plt.tight_layout()
plt.savefig("growth-screen-survivorship-bias-python.png", dpi=150, facecolor=BG)
