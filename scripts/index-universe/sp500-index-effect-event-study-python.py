# Full write-up: https://xfinlink.com/blog/sp500-index-effect-event-study-python
"""Event study of S&P 500 additions, 2016-2024.

Measures market-adjusted abnormal returns around the date membership took
effect, splits the sample by era, and tests whether any pre-inclusion
run-up reverses afterwards.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

PRE, POST = 40, 60          # trading days either side of the effective date
SLUG = "sp500-index-effect-event-study-python"
CHART_PATH = f"/home/user/xfinlink/worker/src/site/blog-images/{SLUG}.png"
WINDOWS = [("[-40,-11]", -40, -11), ("[-10,-1]", -10, -1), ("[-1] only", -1, -1),
           ("[0]", 0, 0), ("[+1,+5]", 1, 5), ("[+1,+20]", 1, 20),
           ("[+1,+60]", 1, 60)]

# ---------------------------------------------------------------- events
ev = xfl.index_events("sp500", event_type="added",
                      start="2016-01-01", end="2024-12-31")
ev["eff"] = pd.to_datetime(ev["effective_date"])

# ------------------------------------------------- benchmark + calendar
spy = xfl.prices("SPY", start="2015-08-01", end="2025-06-30",
                 fields=["return_daily"])
cal = pd.DatetimeIndex(sorted(spy["date"].unique()))
spy_ret = spy.set_index("date")["return_daily"].astype(float)

# ------------------------------------------------------- identity gate
# A symbol is not a company. Tickers get recycled between unrelated issuers
# (TWTR belonged to Tweeter Home Entertainment before Twitter), so validate
# that each event's symbol maps to that event's entity on the effective date.
# resolve() returns the identity window per entity for a symbol.
spans = {}
tk_all = sorted(ev["ticker"].unique().tolist())
for i in range(0, len(tk_all), 10):        # resolve() takes 10 tickers per call
    chunk = tk_all[i:i + 10]
    for t, payload in xfl.resolve(chunk)["data"].items():
        spans[t] = [(e["entity_id"], e.get("ticker_valid_from"),
                     e.get("ticker_valid_to")) for e in payload.get("entities", [])]

lapsed = []

def symbol_matches(ticker, entity_id, eff):
    hit = [s for s in spans.get(ticker, []) if s[0] == entity_id]
    if not hit:
        return "symbol identity window covers a different company"
    _, valid_from, valid_to = hit[0]
    if valid_from and pd.Timestamp(valid_from) > eff:
        return f"symbol identity window starts {valid_from}"
    if valid_to and pd.Timestamp(valid_to) < eff:
        # The recorded span closes before the event. Identity is then confirmed
        # from the price series itself: same entity, continuous window, no jump.
        lapsed.append(ticker)
    return None

# ------------------------------------------------------------ price panel
# Batch by quarter of the effective date, and key the panel on entity_id
# rather than ticker. entity_id is stable across renames and reassignments,
# so a recycled symbol (DD belonged to du Pont before DowDuPont carried it)
# cannot leak another company's prices into an event window.
ev["q"] = ev["eff"].dt.to_period("Q")
panel, no_rows = {}, []
for q, g in ev.groupby("q"):
    tickers = sorted(g["ticker"].unique().tolist())
    d = xfl.prices(tickers,
                   start=(g["eff"].min() - pd.Timedelta(days=95)).strftime("%Y-%m-%d"),
                   end=(g["eff"].max() + pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
                   fields=["return_daily", "volume"])
    if d.empty:
        continue
    for eid, de in d.groupby("entity_id"):
        panel[(q, eid)] = de.sort_values("date").set_index("date")

# --------------------------------------------------------- event windows
JUMP = 0.60          # a one-day move this large is screened out of the sample
paths, drops = [], []
for _, r in ev.iterrows():
    bad = symbol_matches(r["ticker"], r["entity_id"], r["eff"])
    if bad:
        drops.append((r["ticker"], r["entity_name"], r["effective_date"], bad))
        continue
    d = panel.get((r["q"], r["entity_id"]))
    if d is None:
        drops.append((r["ticker"], r["entity_name"], r["effective_date"],
                      "listing history does not span the event window"))
        continue
    pos = cal.searchsorted(r["eff"])
    if pos - PRE < 0 or pos + POST >= len(cal) or cal[pos] != r["eff"]:
        drops.append((r["ticker"], r["entity_name"], r["effective_date"],
                      "effective date not a trading day / calendar edge"))
        continue
    win = cal[pos - PRE: pos + POST + 1]
    ret = d["return_daily"].astype(float).reindex(win)
    if ret.isna().any():
        n_pre, n_post = ret.iloc[:PRE].isna().sum(), ret.iloc[PRE:].isna().sum()
        drops.append((r["ticker"], r["entity_name"], r["effective_date"],
                      f"did not trade on {n_pre} of {PRE} days before, "
                      f"{n_post} of {POST + 1} from day 0"))
        continue
    if ret.abs().max() > JUMP:
        day = int(np.argmax(ret.abs().values)) - PRE
        drops.append((r["ticker"], r["entity_name"], r["effective_date"],
                      f"one-day move of {ret.abs().max() * 100:.0f}% on day {day:+d}"))
        continue
    vol = d["volume"].astype(float).reindex(win)
    base = vol.iloc[:PRE - 1].median()          # days -40 to -2
    paths.append({
        "ticker": r["ticker"], "name": r["entity_name"], "eff": r["eff"],
        "era": "2016-2019" if r["eff"] < pd.Timestamp("2020-01-01") else "2020-2024",
        "ar": (ret - spy_ret.reindex(win)).values,
        "relvol_m1": vol.iloc[PRE - 1] / base,
        "peak_vol_day": int(np.argmax(vol.iloc[PRE - 5:PRE + 6].values)) - 5,
        "max_move": ret.abs().max(),
        "max_move_day": int(np.argmax(ret.abs().values)) - PRE,
    })

res = pd.DataFrame([p["ar"] for p in paths], columns=np.arange(-PRE, POST + 1))
meta = pd.DataFrame([{k: v for k, v in p.items() if k != "ar"} for p in paths])
assert not res.isna().any().any()

def car(df, a, b):
    return df.loc[:, [c for c in df.columns if a <= c <= b]].sum(axis=1) * 100

# ------------------------------------------------------------------ print
print(f"S&P 500 additions, effective 2016-01-01 to 2024-12-31: {len(ev)} events")
print(f"Complete return history over [-{PRE},+{POST}] trading days: "
      f"{len(res)} events ({len(drops)} excluded)")
ev["era"] = np.where(ev["eff"] < pd.Timestamp("2020-01-01"), "2016-2019", "2020-2024")
for era, n in meta["era"].value_counts().sort_index().items():
    raw = (ev["era"] == era).sum()
    print(f"  {era}: {n} of {raw} events kept ({n / raw * 100:.0f}%)")
print(f"\nExcluded events ({len(drops)}):")
for t, _, d, why in drops:
    print(f"  {t:<6} {d}  {why}")

print(f"\nIdentity re-checked against continuous price history for "
      f"{len(lapsed)} names: {', '.join(lapsed) or 'none'}")

print("\nLargest single-day moves left in the sample (checked by name)")
print(meta.nlargest(5, "max_move")[["ticker", "eff", "max_move", "max_move_day"]]
      .to_string(index=False, formatters={"eff": lambda d: d.strftime("%Y-%m-%d"),
                                          "max_move": lambda v: f"{v * 100:.1f}%"}))

print("\nDay-0 alignment check (effective date = first day of membership)")
peak = meta["peak_vol_day"].value_counts().sort_index()
print(f"  median volume on day -1 vs days -40..-2: "
      f"{meta['relvol_m1'].median():.1f}x")
print(f"  share of events whose heaviest volume day in [-5,+5] is day -1: "
      f"{(meta['peak_vol_day'] == -1).mean() * 100:.0f}%")
print(f"  peak-volume day distribution: "
      f"{ {int(k): int(v) for k, v in peak.items()} }")

print("\nCumulative abnormal return vs SPY, market-adjusted (beta = 1)")
print(f"{'window':<11}{'mean':>9}{'median':>9}{'t-stat':>9}{'p':>9}{'% > 0':>8}")
for name, a, b in WINDOWS:
    v = car(res, a, b)
    t = stats.ttest_1samp(v, 0)
    print(f"{name:<11}{v.mean():8.2f}%{v.median():8.2f}%{t.statistic:9.2f}"
          f"{t.pvalue:9.4f}{(v > 0).mean() * 100:7.0f}%")

print("\nBy era (median CAR, mean in brackets)")
print(f"{'window':<11}" + "".join(f"{e:>22}" for e in ["2016-2019", "2020-2024"]))
for name, a, b in WINDOWS:
    cells = []
    for era in ["2016-2019", "2020-2024"]:
        v = car(res[meta["era"] == era], a, b)
        cells.append(f"{v.median():7.2f}% ({v.mean():6.2f}%)")
    print(f"{name:<11}" + "".join(f"{c:>22}" for c in cells))

run_up, after = car(res, -10, -1), car(res, 1, 20)
print("\nDoes the run-up reverse? (medians)")
for era in ["2016-2019", "2020-2024"]:
    m = (meta["era"] == era).values
    ru, af = run_up[m].median(), after[m].median()
    tail = (f"gives back {-af / ru * 100:.0f}% of it" if ru > 0 > af
            else "no run-up to reverse" if ru <= 0 else "no give-back")
    print(f"  {era}: [-10,-1] = {ru:+.2f}%, [+1,+20] = {af:+.2f}%  ->  {tail}")
top = meta.assign(run_up=run_up.values, hold60=car(res, 1, 60).values)
print("\nLargest pre-inclusion run-ups")
print(top.nlargest(4, "run_up")[["ticker", "eff", "run_up", "hold60"]]
      .to_string(index=False, formatters={"eff": lambda d: d.strftime("%Y-%m-%d")}))
print("Widest 60-day outcomes after inclusion")
print(pd.concat([top.nlargest(2, "hold60"), top.nsmallest(2, "hold60")])
      [["ticker", "eff", "run_up", "hold60"]]
      .to_string(index=False, formatters={"eff": lambda d: d.strftime("%Y-%m-%d")}))

spy_20 = pd.Series([spy_ret.reindex(
    cal[cal.searchsorted(e) + 1: cal.searchsorted(e) + 21]).sum()
    for e in meta["eff"]]) * 100
print(f"\nMedian SPY return over the same [+1,+20] window: {spy_20.median():.2f}% "
      f"(a beta error of 0.3 moves the estimate by "
      f"{abs(0.3 * spy_20.median()):.2f}pp)")

# ------------------------------------------------------------------ chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5.5))
days = res.columns.to_numpy()
for era, colour in [("2016-2019", "#9ca3af"), ("2020-2024", "#3b82f6")]:
    sub = res[meta["era"] == era]
    ax.plot(days, sub.cumsum(axis=1).median() * 100, color=colour, lw=2,
            label=f"{era} (n={len(sub)})")
ax.axhline(0, color="#333333", lw=1)
ax.axvline(-1, color="#ef4444", lw=1, ls="--")
lo, hi = ax.get_ylim()
ax.text(2, lo + 0.04 * (hi - lo), "day -1: index funds trade at this close",
        color="#ef4444", fontsize=9, ha="left", va="bottom")
ax.axvline(0, color="#666666", lw=1)
ax.set_title("Median cumulative abnormal return around S&P 500 inclusion")
ax.set_xlabel("Trading days relative to the date membership took effect")
ax.set_ylabel("Cumulative return vs SPY (%)")
ax.legend(frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
