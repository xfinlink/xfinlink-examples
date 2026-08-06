# Full write-up: https://xfinlink.com/blog/sp500-replacement-pairs-deletion-returns-python
"""What happens to a stock after it is removed from the S&P 500?

Takes every date from 2015 to 2024 on which exactly one company entered the
S&P 500 and exactly one left, follows both legs by entity identifier rather
than by ticker, and splits the removals into companies that stopped trading
at the swap and companies that carried on as ordinary listed stocks. For the
second group it measures the twelve-month return of the removed company
against its own replacement, market-adjusted.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

HOLD = 252          # trading days held after the effective date
SLUG = "sp500-replacement-pairs-deletion-returns-python"
CHART = f"/home/user/xfinlink/worker/src/site/blog-images/{SLUG}.png"

# ------------------------------------------------------- replacement pairs
ev = xfl.index_events("sp500", start="2015-01-01", end="2024-12-31")
ev["eff"] = pd.to_datetime(ev["effective_date"])
ev = ev.dropna(subset=["entity_id"])
ev["entity_id"] = ev["entity_id"].astype(int)

counts = ev.groupby(["eff", "event_type"]).size().unstack(fill_value=0)
swap_dates = counts[(counts["added"] == 1) & (counts["removed"] == 1)].index
swaps = ev[ev["eff"].isin(swap_dates)]
pairs = (swaps.pivot_table(index="eff", columns="event_type", values="entity_id",
                           aggfunc="first").dropna().astype(int))
names = swaps.set_index(["eff", "event_type"])["entity_name"].unstack()
tick = swaps.set_index(["eff", "event_type"])["ticker"].unstack()

# ------------------------------------------------------- market benchmark
spy = xfl.prices("SPY", start="2014-12-01", end="2026-02-01", fields=["adj_close"])
spy = spy.set_index("date")["adj_close"].astype(float).sort_index()


def pair_prices(args):
    """Both legs of one swap, from three weeks before to fourteen months after."""
    eff, add_id, rem_id = args
    for attempt in range(3):
        try:
            return xfl.prices(entity_id=[add_id, rem_id],
                              start=(eff - pd.Timedelta(days=20)).date().isoformat(),
                              end=(eff + pd.Timedelta(days=430)).date().isoformat(),
                              fields=["close", "adj_close", "split_ratio", "return_daily"])
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(2 ** attempt)


with ThreadPoolExecutor(max_workers=3) as ex:
    frames = list(ex.map(pair_prices,
                         [(e, r["added"], r["removed"]) for e, r in pairs.iterrows()]))


def series(df, eid, eff):
    """Split-adjusted closes from the last bar on or before the effective date."""
    if df.empty:
        return None
    s = df[df["entity_id"] == eid].drop_duplicates("date").set_index("date").sort_index()
    s = s[s["adj_close"].notna() & (s["adj_close"] > 0)]
    if s.empty:
        return None
    prior = s.index[s.index <= eff]
    if len(prior) == 0 or (eff - prior[-1]).days > 7:
        return None
    return s[s.index >= prior[-1]]


def raw_return(s):
    """Same 12-month return from raw close and the cumulative split ratio.

    A plain split cancels out, so this ties to the adjusted-close return to
    machine precision. A spinoff or similar distribution does not cancel,
    and the two numbers separate.
    """
    factor = s["split_ratio"].fillna(1.0).values[1:HOLD + 1].prod()
    return s["close"].values[HOLD] * factor / s["close"].values[0] - 1.0


def widest_daily_break(s):
    """Largest one-day gap between the total return and the price return.

    An ordinary dividend opens a gap the size of that day's yield. A
    distribution the price series does not carry opens a much larger one,
    which is what this measures.
    """
    px = s["adj_close"].values
    px_ret = px[1:HOLD + 1] / px[:HOLD] - 1.0
    total = s["return_daily"].fillna(0.0).values[1:HOLD + 1]
    return float(np.max(np.abs(total - px_ret)))


# ----------------------------------------------- classify every removal
outcome, rows, paths, checks, dropped = [], [], [], [], []
for (eff, r), df in zip(pairs.iterrows(), frames):
    add, rem = series(df, r["added"], eff), series(df, r["removed"], eff)
    if rem is None:
        outcome.append({"eff": eff, "removed": names.loc[eff, "removed"],
                        "state": "no usable price history", "n_after": 0})
        continue
    n_after = len(rem) - 1
    state = ("stopped trading at the swap" if n_after <= 5 else
             "stopped trading within the year" if n_after < HOLD else
             "still trading a year later")
    outcome.append({"eff": eff, "removed": names.loc[eff, "removed"],
                    "state": state, "n_after": n_after,
                    "last": rem.index[-1].date()})
    if state != "still trading a year later" or add is None or len(add) < HOLD + 1:
        continue

    mkt = spy[spy.index >= add.index[0]].iloc[:HOLD + 1]
    if len(mkt) < HOLD + 1:
        continue
    mkt = mkt.values / mkt.values[0] - 1.0
    pa = add["adj_close"].values[:HOLD + 1] / add["adj_close"].values[0] - 1.0
    pr = rem["adj_close"].values[:HOLD + 1] / rem["adj_close"].values[0] - 1.0

    gap = max(abs(raw_return(add) - pa[-1]), abs(raw_return(rem) - pr[-1]))
    brk = max(widest_daily_break(add), widest_daily_break(rem))
    if gap > 0.005 or brk > 0.02:      # a distribution the price series cannot carry
        dropped.append((eff, names.loc[eff, "removed"], gap, brk))
        continue
    checks.append((gap, brk))

    tr_a = np.prod(1 + add["return_daily"].fillna(0.0).values[1:HOLD + 1]) - 1
    tr_r = np.prod(1 + rem["return_daily"].fillna(0.0).values[1:HOLD + 1]) - 1
    rows.append({"tr_spread": tr_r - tr_a,
                 "eff": eff, "added": names.loc[eff, "added"],
                 "removed": names.loc[eff, "removed"],
                 "add_tkr": tick.loc[eff, "added"], "rem_tkr": tick.loc[eff, "removed"],
                 "add_exc": pa[-1] - mkt[-1], "rem_exc": pr[-1] - mkt[-1],
                 "spread": (pr[-1] - mkt[-1]) - (pa[-1] - mkt[-1])})
    paths.append((pa - mkt, pr - mkt))

out = pd.DataFrame(outcome)
res = pd.DataFrame(rows)
P = np.array(paths)

t, p = stats.ttest_1samp(res["spread"], 0.0)
usable = out[out["state"] != "no usable price history"]
print(f"one-for-one replacement dates 2015-2024: {len(pairs)}, "
      f"of which {len(usable)} carry a usable price history for both legs")
print(usable["state"].value_counts().to_string())
print(f"\nremoved leg 12m excess: mean {res['rem_exc'].mean():+.2%}  "
      f"median {res['rem_exc'].median():+.2%}")
print(f"added   leg 12m excess: mean {res['add_exc'].mean():+.2%}  "
      f"median {res['add_exc'].median():+.2%}")
print(f"pair spread (removed minus added), n={len(res)}: mean {res['spread'].mean():+.2%}  "
      f"median {res['spread'].median():+.2%}  t={t:.2f}  p={p:.4f}  "
      f"removed leg wins {(res['spread'] > 0).mean():.1%}")
print(f"wilcoxon p={stats.wilcoxon(res['spread']).pvalue:.4f}; "
      f"sign test p={stats.binomtest((res['spread'] > 0).sum(), len(res), 0.5).pvalue:.4f}; "
      f"middle 80% of spreads {np.percentile(res['spread'], 10):+.1%} to "
      f"{np.percentile(res['spread'], 90):+.1%}")
print("removed leg vs the market along the way: "
      + ", ".join(f"{np.median(P[:, 1, d]):+.1%} at day {d}" for d in (21, 63, 126, 252)))
trim = res["spread"].sort_values().iloc[1:-1]
print(f"robustness: trimmed mean spread (widest each way removed, n={len(trim)}) "
      f"{trim.mean():+.2%}; same pairs measured on total return "
      f"mean {res['tr_spread'].mean():+.2%} median {res['tr_spread'].median():+.2%}")
print(f"\npairs set aside for a distribution inside the year: {len(dropped)} "
      f"({', '.join(d[1].title() for d in dropped)})")
print(f"on the {len(checks)} pairs kept, the adjusted-close return and the raw-close "
      f"recompute differ by at most {max(c[0] for c in checks):.2e}, and no single day's "
      f"total return parts from its price return by more than "
      f"{max(c[1] for c in checks):.2%}")
print("  set-aside margins: "
      + "; ".join(f"{d[1].title()} {d[2]:.1%} / {d[3]:.1%}" for d in dropped))

show = res.sort_values("spread")[["eff", "removed", "rem_tkr", "rem_exc",
                                  "added", "add_tkr", "add_exc", "spread"]]
print("\nwidest five pairs and narrowest three, 12-month return vs the S&P 500")
for _, x in pd.concat([show.head(5), show.tail(3)]).iterrows():
    print(f"  {x['eff'].date()}  {x['removed'][:24]:24s} ({x['rem_tkr']:5s}) {x['rem_exc']:+7.1%}"
          f"   vs  {x['added'][:22]:22s} ({x['add_tkr']:5s}) {x['add_exc']:+7.1%}"
          f"   spread {x['spread']:+7.1%}")

# -------------------------------------------------- point-in-time evidence
today = xfl.index("sp500")
snap = xfl.index("sp500", as_of="2016-01-01")
gone = set(snap["entity_id"]) - set(today["entity_id"])
print(f"members on 2016-01-01: {len(snap)}; today: {len(today)}; "
      f"in the 2016 roster and not in today's: {len(gone)}")

# ------------------------------------------------------------------ chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

tally = out[out["state"] != "no usable price history"].copy()
tally["year"] = tally["eff"].dt.year
grid = (tally.assign(k=np.where(tally["state"] == "still trading a year later",
                                "Still trading a year later",
                                "Stopped trading within the year"))
        .pivot_table(index="year", columns="k", aggfunc="size", fill_value=0))
ax1.bar(grid.index, grid["Stopped trading within the year"], color="#6b7280",
        label="Stopped trading within the year")
ax1.bar(grid.index, grid.get("Still trading a year later", 0),
        bottom=grid["Stopped trading within the year"], color="#3b82f6",
        label="Still trading a year later")
ax1.set_title("Companies removed from the S&P 500 in one-for-one swaps, 2015-2024")
ax1.set_ylabel("Removals")
ax1.legend(frameon=False, fontsize=9)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

x = np.arange(HOLD + 1)
ax2.plot(x, np.median(P[:, 0, :], axis=0) * 100, color="#3b82f6", lw=1.8,
         label="Company added")
ax2.plot(x, np.median(P[:, 1, :], axis=0) * 100, color="#f59e0b", lw=1.8,
         label="Company removed")
ax2.axhline(0, color="#555555", lw=0.8)
ax2.set_title(f"Median return after the swap, market-adjusted "
              f"({len(res)} pairs where the removed company kept trading)")
ax2.set_xlabel("Trading days after the change took effect")
ax2.set_ylabel("Cumulative return vs S&P 500 (%)")
ax2.set_xlim(0, HOLD)
ax2.legend(frameon=False, fontsize=9)
ax2.grid(axis="y", color="#222222", lw=0.6)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
