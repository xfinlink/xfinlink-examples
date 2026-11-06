# Full write-up: https://xfinlink.com/blog/january-effect-survivorship-bias-python
#
# Does the January effect in last year's losers survive survivorship bias? At each month end
# from December 1999 to November 2024, S&P 500 members are sorted into equal-weight quintiles on
# their trailing 12-month total return and held for the next month. The December sort ranks on
# the calendar year just ended, so its holding month is the January the tax-loss story is about;
# the other eleven months are the control. The test runs once on the roster in force at each
# month end (point-in-time) and once on today's roster backfilled (the survivor list). A member
# that stops trading earns any recorded delisting return in its exit month; a member with no
# return in the holding month drops from that month's average.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# SPY: month-end trading days (formation dates) and monthly total return
spy = xfl.prices("SPY", start="1999-01-01", end="2024-12-31", fields=["return_daily"])
spy["month"] = spy["date"].dt.to_period("M")
spy_m = spy.groupby("month")["return_daily"].apply(lambda r: (1 + r.fillna(0)).prod() - 1)
formation = spy.groupby("month")["date"].max().loc["1999-12":"2024-11"]

# Roster in force at each month end; a member leaving that day is not held
pit = {}
for m, d in formation.items():
    r = xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).dropna(subset=["entity_id"])
    r = r[pd.to_datetime(r["removed_date"]) != d]
    pit[m] = set(r["entity_id"].astype(int))
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))

# Monthly total returns by entity_id; 16 ids x 312 months fit on one 5,000-row page,
# so each batch's recorded exits (meta.delisting) arrive complete
ids = sorted(set().union(*pit.values()) | today)
parts, exits = [], {}
for i in range(0, len(ids), 16):
    p = xfl.prices(entity_id=ids[i:i + 16], start="1999-01-01", end="2024-12-31",
                   interval="1mo", fields=["return_daily"], max_rows=100_000)
    if p.empty:
        continue
    assert p.attrs["meta"].get("pages", 1) == 1, "batch spans more than one page"
    parts.append(p)
    exits.update(p.attrs["meta"].get("delisting") or {})
px = pd.concat(parts)
px["month"] = px["date"].dt.to_period("M")
months = pd.period_range("1999-01", "2024-12", freq="M")
ret = (px.groupby(["month", "entity_id"])["return_daily"]   # compounds a month split in two bars
       .apply(lambda r: np.nan if r.isna().all() else (1 + r.fillna(0)).prod() - 1)
       .unstack().reindex(months))
traded = ret.notna()

# Recorded delisting return, applied in the exit month (the month of the last bar, or the
# month after it when the exit falls before that month's first bar)
applied = 0
for eid, ev in exits.items():
    eid, dl, m = int(eid), ev.get("delisting_return"), pd.Period(ev["event_date"], "M")
    if dl is None or eid not in ret.columns or m not in ret.index:
        continue
    last = ret[eid].last_valid_index()
    if last == m:
        ret.at[m, eid] = (1 + ret.at[m, eid]) * (1 + dl) - 1
        applied += 1
    elif last == m - 1:
        ret.at[m, eid] = dl
        applied += 1

# Signal: compounded return over the 12 months ending at the formation month, all 12 present
signal = np.expm1(np.log1p(ret).rolling(12, min_periods=12).sum())
print(f"{len(pit)} month-end rosters, {len(ids)} securities, {len(exits)} recorded exits, "
      f"{applied} delisting returns applied in the exit month")


def backtest(universe):
    rows, picks = [], []
    for m in pit:
        u = [e for e in universe(m) if e in signal.columns]
        s = signal.loc[m, u].dropna()
        q = np.ceil(s.rank(pct=True) * 5)
        h = ret.loc[m + 1, s.index]
        row = {"month": m + 1, "n": len(s), "all": h.mean(), "dropped": int(h.isna().sum())}
        for k in range(1, 6):
            row[f"Q{k}"] = h[q == k].mean()
        rows.append(row)
        if m.month == 12:
            picks.append(pd.DataFrame({"formed": m, "entity_id": s.index[q == 1],
                                       "past_12m": s[q == 1].values, "january": h[q == 1].values}))
    b = pd.DataFrame(rows).set_index("month")
    b["spread"] = b["Q1"] - b["Q5"]
    b["spy"] = spy_m.reindex(b.index).values
    return b, pd.concat(picks, ignore_index=True)


def t_stat(x, lags):
    x = x.dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0] * 100, fit.tvalues[0]


books = {"Point-in-time rosters": backtest(lambda m: pit[m]),
         "Today's roster": backtest(lambda m: today)}
for name, (b, picks) in books.items():
    jan, rest = b[b.index.month == 1], b[b.index.month != 1]
    print(f"\n{name}: {b['n'].mean():.0f} members sorted a month; member-months with no "
          f"holding-month return: {b['dropped'].sum()}")
    for label, part, lags in [("January (25)", jan, 1), ("Feb-Dec (275)", rest, 6)]:
        mu, t = t_stat(part["spread"], lags)
        print(f"  {label}: losers {part['Q1'].mean():+.2%}, winners {part['Q5'].mean():+.2%}, "
              f"all members {part['all'].mean():+.2%}, SPY {part['spy'].mean():+.2%}; "
              f"losers minus winners {mu:+.2f}% a month (Newey-West t {t:.2f})")
    print("  Quintile means, January: " + " ".join(f"{jan[f'Q{k}'].mean():+.2%}" for k in range(1, 6)))
    print("  Quintile means, Feb-Dec: " + " ".join(f"{rest[f'Q{k}'].mean():+.2%}" for k in range(1, 6)))
    X = sm.add_constant((b.index.month == 1).astype(float))
    fit = sm.OLS(b["spread"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"  January minus other months, spread: {fit.params[1] * 100:+.2f} points "
          f"(Newey-West t {fit.tvalues[1]:.2f})")
    ex = jan["Q1"] - jan["spy"]
    print(f"  Losers vs SPY in January: {ex.mean():+.2%} on average, ahead in {(ex > 0).sum()} "
          f"of {len(ex)} Januaries; spread positive in {(jan['spread'] > 0).sum()} of {len(jan)}")
    gone = ~picks["entity_id"].isin(today)
    ended = picks["entity_id"].map(lambda e: ret[e].last_valid_index() < months[-1])
    print(f"  December losers: {len(picks)} name-years, {gone.mean():.1%} not in today's index, "
          f"{ended.mean():.1%} stopped trading before 2024-12")

pt, sv = books["Point-in-time rosters"][0], books["Today's roster"][0]
jan_pt, jan_sv = pt[pt.index.month == 1], sv[sv.index.month == 1]
print("\nJanuary by year, losers minus winners (%):")
print(pd.DataFrame({"point-in-time": jan_pt["spread"] * 100, "today's roster": jan_sv["spread"] * 100,
                    "PIT losers": jan_pt["Q1"] * 100, "SPY": jan_pt["spy"] * 100})
      .set_axis(jan_pt.index.year).round(2).to_string())
for a, z in [("2000", "2012"), ("2013", "2024")]:
    p, s = jan_pt.loc[a:z], jan_sv.loc[a:z]
    print(f"Januaries {a}-{z}: spread point-in-time {p['spread'].mean():+.2%}, "
          f"today's roster {s['spread'].mean():+.2%}")
no01 = lambda j: j.drop(pd.Period("2001-01", "M"))["spread"]
print(f"Januaries without 2001: spread point-in-time {no01(jan_pt).mean():+.2%}, today's roster "
      f"{no01(jan_sv).mean():+.2%}; median January spread {jan_pt['spread'].median():+.2%} and "
      f"{jan_sv['spread'].median():+.2%}")
e01 = lambda j: no01(j.loc["2000":"2012"]).mean()
print(f"Januaries 2000-2012 without 2001: spread point-in-time {e01(jan_pt):+.2%}, "
      f"today's roster {e01(jan_sv):+.2%}")

by_month = pd.DataFrame({"PIT spread": pt.groupby(pt.index.month)["spread"].mean(),
                         "survivor spread": sv.groupby(sv.index.month)["spread"].mean(),
                         "PIT losers": pt.groupby(pt.index.month)["Q1"].mean(),
                         "survivor losers": sv.groupby(sv.index.month)["Q1"].mean()}) * 100
print("\nAverage by holding month, 2000-2024 (% a month):")
print(by_month.round(2).to_string())
apr = pt[pt.index.month == 4]["spread"]
print(f"Largest April spread: {apr.idxmax()} {apr.max():+.2%}; April without it {apr.drop(apr.idxmax()).mean():+.2%}")

# Point-in-time December losers that are not in today's index, and their Januaries
picks = books["Point-in-time rosters"][1]
tick = px.groupby(["month", "entity_id"])["ticker"].last()  # symbol in force that month
gone = picks[~picks["entity_id"].isin(today)]
print(f"\nPoint-in-time December losers: in today's index {picks.loc[picks['entity_id'].isin(today), 'january'].mean():+.2%} "
      f"in January, not in today's index {gone['january'].mean():+.2%} (n={len(gone)})")
held_tick = lambda r: tick.get((r["formed"] + 1, r["entity_id"]), tick.get((r["formed"], r["entity_id"])))
show = picks.assign(ticker=picks.apply(held_tick, axis=1), formed=picks["formed"].astype(str),
                    in_index_today=picks["entity_id"].isin(today))
print("Largest January gains among December losers:")
print(show.nlargest(5, "january")[["formed", "ticker", "past_12m", "january", "in_index_today"]]
      .round(3).to_string(index=False))
print("Largest January losses among December losers:")
print(show.nsmallest(5, "january")[["formed", "ticker", "past_12m", "january", "in_index_today"]]
      .round(3).to_string(index=False))

# Chart: loser-minus-winner spread by holding month, and the loser quintile's own return
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(1, 13)
series = [("PIT", "Point-in-time rosters", "#3b82f6", -0.2), ("survivor", "Today's roster", "#f59e0b", 0.2)]
for key, label, color, off in series:
    ax1.bar(x + off, by_month[f"{key} spread"].values, width=0.4, color=color, label=label)
ax1.axhline(0, color="#666666", lw=1)
ax1.set_xticks(x, list("JFMAMJJASOND"))
ax1.set_xlabel("Holding month")
ax1.set_ylabel("Losers minus winners, average (% a month)")
ax1.set_title("Spread by month, 2000-2024", fontsize=11)
ax1.legend(frameon=False, loc="upper right")
groups = [jan_pt["Q1"].mean(), pt[pt.index.month != 1]["Q1"].mean()], \
         [jan_sv["Q1"].mean(), sv[sv.index.month != 1]["Q1"].mean()]
for (key, label, color, off), vals in zip(series, groups):
    bars = ax2.bar(np.arange(2) + off, np.array(vals) * 100, width=0.4, color=color, label=label)
    ax2.bar_label(bars, fmt="%.2f", padding=3, color="#e0e0e0", fontsize=9)
ax2.set_xticks([0, 1], ["January", "Feb-Dec"])
ax2.set_ylabel("Prior-year losers, average (% a month)")
ax2.set_title("Loser quintile return", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("January Effect Backtest: Survivorship Bias in the S&P 500", fontsize=13)
plt.tight_layout()
plt.savefig("january-effect-survivorship-bias-python.png", dpi=150)
