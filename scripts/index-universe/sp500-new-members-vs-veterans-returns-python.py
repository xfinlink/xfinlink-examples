# Full write-up: https://xfinlink.com/blog/sp500-new-members-vs-veterans-returns-python
#
# Do companies newly added to the S&P 500 earn more or less than long-standing members?
# At each month end from December 1999 to November 2024, the roster in force that day is
# split by tenure, the years since the start of each member's current membership spell
# (a re-added company starts again at zero). Each tenure bucket is held equal-weight for the
# following month, January 2000 to December 2024, and compared with SPY (total return).
# The same test is run on today's roster with today's join dates, the survivor-only shortcut.
# A robustness view follows every addition from 2000 to 2021 for 12 and 36 months.
#
# Holding rules: a member whose last day in the index is the formation date is not held;
# a recorded delisting return is added in the month of the exit; a member with no return in
# the holding month drops out of that month.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Month-end trading days from SPY's own sessions, and SPY's monthly total return
spy = xfl.prices("SPY", start="1999-12-01", end="2024-12-31", fields=["return_daily"])
spy["date"] = pd.to_datetime(spy["date"])
month_ends = spy.groupby(spy["date"].dt.to_period("M"))["date"].max()
spy_m = spy.groupby(spy["date"].dt.to_period("M"))["return_daily"].apply(
    lambda r: (1 + r.fillna(0)).prod() - 1)
formation = month_ends.loc[:"2024-11"]

# The roster in force at each month end, with each member's join date for its current spell
pit = pd.concat([xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).dropna(subset=["entity_id"])
                 .assign(month=m, asof=d) for m, d in formation.items()])
today = xfl.index("sp500").dropna(subset=["entity_id"])
for f in (pit, today):
    f["entity_id"] = f["entity_id"].astype(int)
    f["added_date"] = pd.to_datetime(f["added_date"])
pit["removed_date"] = pd.to_datetime(pit["removed_date"])
leaving = pit["removed_date"] == pit["asof"]          # last day in the index is the formation date
pit = pit[~leaving]

# Survivor-only version: today's members, counted from their join date onward
surv = pd.concat([today[today["added_date"] <= d].assign(month=m, asof=d)
                  for m, d in formation.items()])

# Monthly total returns for every security either roster holds, plus recorded exit returns
ids = sorted(set(pit.entity_id) | set(today.entity_id))
parts, exits = [], {}
for i in range(0, len(ids), 16):   # 16 ids x 302 months fit on one 5,000-row page, so each
    p = xfl.prices(entity_id=ids[i:i + 16], start="1999-11-01", end="2024-12-31",   # page's
                   interval="1mo", fields=["return_daily"], max_rows=100_000)       # exits are kept
    assert len(p) < 5000, "batch spans more than one page"
    parts.append(p)
    exits.update(p.attrs.get("meta", {}).get("delisting", {}) or {})
px = pd.concat(parts)
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot_table(index="month", columns="entity_id", values="return_daily", aggfunc="first")
n_exit = 0
for eid, ev in exits.items():
    eid, dl, m = int(eid), ev.get("delisting_return"), pd.Period(ev["event_date"], "M")
    if dl is None or eid not in ret.columns or m not in ret.index:
        continue
    r0 = ret.at[m, eid]
    ret.at[m, eid] = (1 + (0 if pd.isna(r0) else r0)) * (1 + dl) - 1
    n_exit += 1

# Tenure buckets at each month end, held for the next month
bins, labels = [0, 1, 3, 10, np.inf], ["first year", "years 2-3", "years 4-10", "10+ years"]
def bucket_returns(book):
    b = book.assign(hold=book["month"] + 1,
                    tenure=(book["asof"] - book["added_date"]).dt.days / 365.25)
    b["bucket"] = pd.cut(b["tenure"], bins, right=False, labels=labels)
    s = ret.stack()
    b["r"] = s.reindex(pd.MultiIndex.from_arrays([b["hold"], b["entity_id"]])).to_numpy()
    held = b.dropna(subset=["r"])
    out = held.groupby(["hold", "bucket"], observed=True)["r"].mean().unstack()
    out["all members"] = held.groupby("hold")["r"].mean()
    return out, b, held

def stats(r):
    ex = r - spy_m.reindex(r.index)
    return pd.Series({"% a year": ((1 + r).prod() ** (12 / len(r)) - 1) * 100,
                      "volatility %": r.std() * np.sqrt(12) * 100,
                      "months beating SPY %": (ex > 0).mean() * 100,
                      "excess vs SPY % a year": ex.mean() * 1200,
                      "t-stat": ex.mean() / ex.std() * np.sqrt(len(ex))})

res = {}
for name, book in [("Point-in-time rosters", pit), ("Today's roster, today's join dates", surv)]:
    out, b, held = bucket_returns(book)
    res[name] = out
    if name.startswith("Point"):
        print(f"{len(formation)} month ends, {len(ids)} securities, holding months "
              f"{out.index[0]} to {out.index[-1]}")
        print(f"Member-months held: {len(held):,}; not held because the member left on the "
              f"formation date: {leaving.sum()}; no return in the holding month: "
              f"{b['r'].isna().sum()}; recorded delisting returns applied: {n_exit}")
        print(f"Securities re-added after an earlier spell (tenure restarts): "
              f"{(book.drop_duplicates(['entity_id', 'added_date']).groupby('entity_id').size() > 1).sum()}\n")
    sizes = held.groupby(["hold", "bucket"], observed=True).size().unstack()
    print(f"{name}: average members per month "
          + ", ".join(f"{c} {sizes[c].mean():.0f}" for c in labels))
    print(pd.DataFrame({c: stats(out[c]) for c in labels + ["all members"]}).T.round(2)
          .to_string(), "\n")

print("Spread: first year minus 10+ years, equal weight")
for name, out in res.items():
    d = (out["first year"] - out["10+ years"]).dropna()
    print(f"  {name}: {d.mean() * 1200:+.2f} points a year, t-stat "
          f"{d.mean() / d.std() * np.sqrt(len(d)):.2f}, first year ahead in "
          f"{(d > 0).mean() * 100:.0f}% of {len(d)} months")
by5 = pd.DataFrame({name: (out["first year"] - out["10+ years"]).groupby(
    lambda m: f"{m.year // 5 * 5}-{m.year // 5 * 5 + 4}").mean() * 1200 for name, out in res.items()})
by5.index.name = "holding months"
print("  by five-year period, points a year:")
print(by5.round(2).to_string())
print(f"SPY: {((1 + spy_m.loc['2000-01':]).prod() ** (12 / 300) - 1) * 100:.2f}% a year\n")

# Cross-check: each addition's 12- and 36-month return from the end of its join month vs SPY
# A stock that stops trading is compared with SPY through its last return month only;
# a window with a gap in the stock's monthly returns is left out
def forward(eid, start, n):
    win = pd.period_range(start + 1, periods=n, freq="M")
    r = ret[eid].reindex(win)
    last = r.last_valid_index()
    if last is None or r.loc[:last].isna().any():
        return np.nan, np.nan
    return (1 + r.loc[:last]).prod() - 1, (1 + spy_m.loc[win[0]:last]).prod() - 1

rows = []
for name, book in [("Additions on point-in-time rosters", pit),
                   ("Additions still in today's roster", today)]:
    adds = book.drop_duplicates(["entity_id", "added_date"])
    adds = adds[(adds["added_date"] >= "2000-01-01") & (adds["added_date"] <= "2021-12-31")]
    for n in (12, 36):
        fw = pd.DataFrame([forward(e, pd.Period(a, "M"), n)
                           for e, a in zip(adds["entity_id"], adds["added_date"])],
                          columns=["stock", "spy"])
        ex = (fw["stock"] - fw["spy"]).dropna()
        rows.append({"group": name, "horizon": f"{n} months", "additions": len(adds),
                     "left out (gap)": len(adds) - len(ex),
                     "median excess %": ex.median() * 100, "mean excess %": ex.mean() * 100,
                     "beat SPY %": (ex > 0).mean() * 100})
print("Additions 2000-2021: return from the end of the join month, minus SPY over the same months\n"
      "(through the stock's last return month when it stopped trading sooner)")
print(pd.DataFrame(rows).set_index(["group", "horizon"]).round(1).to_string())

# The additions a survivor-only test cannot see: largest 36-month losses among those since removed
gone = pit.drop_duplicates(["entity_id", "added_date"])
gone = gone[gone["added_date"].between("2000-01-01", "2021-12-31") & ~gone["entity_id"].isin(today["entity_id"])]
gone = gone.assign(r36=[forward(e, pd.Period(a, "M"), 36)[0]
                        for e, a in zip(gone["entity_id"], gone["added_date"])]).dropna(subset=["r36"])
print(f"\nAdditions 2000-2021 no longer in today's roster, gap-free windows: {len(gone)}; "
      "five largest 36-month losses")
print(gone.nsmallest(5, "r36").assign(added=lambda g: g["added_date"].dt.strftime("%Y-%m-%d"),
      r36=lambda g: (g["r36"] * 100).round(1))[["ticker", "entity_name", "added", "r36"]]
      .rename(columns={"r36": "36-month return %"}).to_string(index=False))

# Chart: annualised return by tenure, point-in-time vs survivor-only
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
for k, (name, color) in enumerate(zip(res, ["#3b82f6", "#f59e0b"])):
    vals = [stats(res[name][c])["% a year"] for c in labels]
    bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, color=color, label=name)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#e0e0e0", fontsize=10)
spy_ann = ((1 + spy_m.loc["2000-01":]).prod() ** (12 / 300) - 1) * 100
ax.axhline(spy_ann, color="#888888", lw=1, ls="--")
ax.text(len(labels) - 0.05, spy_ann + 0.3, f"SPY {spy_ann:.1f}%", color="#aaaaaa",
        ha="right", fontsize=10)
ax.set_xlim(-0.6, len(labels))
ax.set_ylim(0, 17)
ax.set_xticks(x, ["First year", "Years 2-3", "Years 4-10", "10+ years"])
ax.set_xlabel("Time in the S&P 500 at the start of each month")
ax.set_ylabel("Annualised return, 2000-2024 (%)")
ax.set_title("New vs Veteran S&P 500 Members: Equal-Weight Return by Tenure, 2000-2024")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig("sp500-new-members-vs-veterans-returns-python.png", dpi=150)
