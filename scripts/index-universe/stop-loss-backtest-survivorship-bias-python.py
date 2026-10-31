# Full write-up: https://xfinlink.com/blog/stop-loss-backtest-survivorship-bias-python
#
# Does a stop-loss look worse on today's S&P 500 than on the index as it stood? At each
# year end from 1999 to 2019, every member of the roster in force that day (point-in-time)
# and every member of today's roster then trading is bought and held for five years, with
# and without a stop. Stops are checked on each daily close of the total-return index:
# trailing stops sell 15% or 30% below the running peak, fixed stops 15% or 30% below the
# entry price, and the proceeds sit in cash earning nothing. A holding that stops trading
# for 20 sessions or more is sold at its last close; the proceeds sit in cash.
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

H, GAP = 5, 20  # five-year holds; 20 sessions without a return ends a holding
RULES = {"Trailing 15%": ("trailing", 0.15), "Trailing 30%": ("trailing", 0.30),
         "Fixed 15%": ("fixed", 0.15), "Fixed 30%": ("fixed", 0.30)}
starts = pd.date_range("1999-12-01", "2019-12-31", freq="BYE")  # last business day of each year
pit = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).entity_id.dropna().astype(int).tolist()
       for d in starts}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

ids = sorted(set().union(*pit.values()) | today)
def fetch(i):
    return xfl.prices(entity_id=ids[i:i + 25], start="2000-01-01", end="2024-12-31",
                      fields=["return_daily"], max_rows=250_000)
with ThreadPoolExecutor(8) as pool:
    chunks = list(pool.map(fetch, range(0, len(ids), 25)))
px = pd.concat(chunks)
ret = px.pivot_table(index="date", columns="entity_id", values="return_daily", aggfunc="first").sort_index()

def held_path(r):
    """Total-return index from the first day of the window to the last close before a 20-session gap."""
    missing = np.isnan(r)
    run = np.convolve(missing, np.ones(GAP, int), "valid")  # missing sessions in each 20-session stretch
    end = np.flatnonzero(run == GAP)[0] if (run == GAP).any() else len(r)
    return np.cumprod(1 + np.nan_to_num(r[:end])), end < len(r)

def with_stop(P, kind, k):
    """Value at the end of the window (1 = entry) and whether the stop sold."""
    peak = np.maximum.accumulate(np.maximum(P, 1.0)) if kind == "trailing" else np.ones_like(P)
    hit = np.flatnonzero(P <= (1 - k) * peak)
    return (P[hit[0]], True) if len(hit) else (P[-1], False)

rows = []
for year, members in pit.items():
    window = ret.loc[f"{year + 1}-01-01":f"{year + H}-12-31"]
    trading = set(window.columns[window.iloc[0].notna()])
    for roster, u in {"Point-in-time": [e for e in members if e in trading],
                      "Today's roster": sorted(today & trading)}.items():
        for e in u:
            P, stopped_trading = held_path(window[e].to_numpy())
            row = {"start": year, "roster": roster, "entity_id": e, "member_today": e in today,
                   "stopped_trading": stopped_trading, "hold": P[-1]}
            for name, (kind, k) in RULES.items():
                row[name], row[name + " sold"] = with_stop(P, kind, k)
            rows.append(row)
res = pd.DataFrame(rows)

def summary(g):
    basket = g.groupby("start")  # one equal-weight basket per start date
    annual = lambda col: basket[col].mean() ** (1 / H) - 1
    out = {"positions": len(g), "hold, % a year": annual("hold").mean() * 100}
    for name in RULES:
        sold = g[g[name + " sold"]]
        out[f"{name}: cost, pts a year"] = (annual("hold") - annual(name)).mean() * 100
        out[f"{name}: stop beats hold"] = (g[name] > g["hold"]).mean() * 100
        out[f"{name}: sales where holding ended lower"] = (sold["hold"] < sold[name]).mean() * 100
    return pd.Series(out)

pit_res = res[res["roster"] == "Point-in-time"]
groups = {"Point-in-time": pit_res,
          "  still in index": pit_res[pit_res["member_today"]],
          "  since removed": pit_res[~pit_res["member_today"]],
          "Today's roster": res[res["roster"] == "Today's roster"]}
table = pd.DataFrame({k: summary(g) for k, g in groups.items()})

# Where the gap comes from: five-year return given up per position, in points, split into
# members later removed (in the point-in-time roster only) and later additions (in today's only)
gaps = pd.DataFrame({k: {name: (g["hold"] - g[name]).mean() * 100 for name in RULES} for k, g in groups.items()})
gaps["removed members"] = gaps["  still in index"] - gaps["Point-in-time"]
gaps["later additions"] = gaps["Today's roster"] - gaps["  still in index"]
gaps["removed share"] = gaps["removed members"] / (gaps["Today's roster"] - gaps["Point-in-time"])

print(f"{len(pit)} start dates, {len(ids)} securities, {len(res):,} five-year holdings")
print(res.groupby("roster").size().div(len(pit)).round(0).rename("members per start").to_string(), "\n")
print(f"Point-in-time holdings no longer in the index today: {1 - pit_res['member_today'].mean():.1%}; "
      f"stopped trading inside the window: {pit_res['stopped_trading'].mean():.1%}\n")
for metric in ["cost, pts a year", "stop beats hold", "sales where holding ended lower"]:
    t = table.loc[[f"{n}: {metric}" for n in RULES]].rename(index=lambda i: i.split(":")[0])
    print(f"{metric} (%)" if metric != "cost, pts a year" else "Cost of the stop, points of annual return")
    print(t.to_string(float_format=lambda x: f"{x:.2f}"), "\n")
print("Buy and hold, % a year:", table.loc["hold, % a year"].round(2).to_dict(), "\n")
print("Five-year return given up per position (points) and where the survivor roster's extra cost comes from")
print(gaps.drop(columns=["  still in index", "  since removed"]).to_string(
    float_format=lambda x: f"{x:.1%}" if abs(x) < 1 else f"{x:.1f}"), "\n")

# Examples: well-known exits held from a point-in-time roster
print("Examples (value of $1 after five years)")
for ticker, year in [("ENE", 1999), ("WCOM", 1999), ("LEH", 2003), ("WM", 2003), ("AIG", 2003)]:
    roster = xfl.index("sp500", as_of=starts[year - 1999].strftime("%Y-%m-%d")).set_index("ticker")
    r = pit_res[(pit_res["start"] == year) & (pit_res["entity_id"] == int(roster.loc[ticker, "entity_id"]))].iloc[0]
    print(f"  {roster.loc[ticker, 'entity_name'].title():34s} from {year}  hold {r['hold']:.3f}  "
          + "  ".join(f"{n} {r[n]:.3f}" for n in RULES))

# Sensitivity: add the recorded delisting return to holdings still open when their security left
# the exchange (exit records come with monthly bars, 16 securities per request)
def exits(i):
    m = xfl.prices(entity_id=ids[i:i + 16], start="2000-01-01", end="2024-12-31", interval="1mo",
                   fields=["return_daily"], max_rows=10_000)
    return m.attrs["meta"].get("delisting") or {}
with ThreadPoolExecutor(8) as pool:
    recorded = {int(k): v for d in pool.map(exits, range(0, len(ids), 16)) for k, v in d.items()}
def last_value(r, col):
    d = recorded.get(r["entity_id"]) or {}
    inside = r["stopped_trading"] and d.get("delisting_return") is not None and \
        str(r["start"] + 1) <= d["event_date"][:4] <= str(r["start"] + H)
    still_open = col == "hold" or not r[col + " sold"]
    return r[col] * (1 + d["delisting_return"]) if inside and still_open else r[col]
adj = res.copy()
for col in ["hold", *RULES]:
    adj[col] = res.apply(last_value, axis=1, col=col)
print(f"\nWith recorded delisting returns ({sum(1 for v in recorded.values() if v.get('delisting_return') is not None)} "
      "records), cost of the stop in points of annual return:")
adj_pit = adj[adj["roster"] == "Point-in-time"]
print(pd.DataFrame({k: summary(g) for k, g in {"Point-in-time": adj_pit,
                    "Today's roster": adj[adj["roster"] == "Today's roster"]}.items()})
      .loc[[f"{n}: cost, pts a year" for n in RULES]].rename(index=lambda i: i.split(":")[0])
      .to_string(float_format=lambda x: f"{x:.2f}"))

# Chart: cost of each stop, and how often holding on would have ended below the sale
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
x, rules = np.arange(len(RULES)), list(RULES)
series = {"Point-in-time rosters": ("Point-in-time", "#3b82f6"), "Today's roster": ("Today's roster", "#f59e0b")}
for i, (label, (col, c)) in enumerate(series.items()):
    v = [table.loc[f"{n}: cost, pts a year", col] for n in rules]
    a1.bar(x + (i - 0.5) * 0.38, v, 0.38, color=c, label=label)
    for xi, vi in zip(x + (i - 0.5) * 0.38, v):
        a1.text(xi, vi + 0.2, f"{vi:.1f}", ha="center", fontsize=8)
falls = {"Today's roster": ("Today's roster", "#f59e0b"), "Point-in-time, still in index": ("  still in index", "#93c5fd"),
         "Point-in-time, since removed": ("  since removed", "#3b82f6")}
for i, (label, (col, c)) in enumerate(falls.items()):
    v = [table.loc[f"{n}: sales where holding ended lower", col] for n in rules]
    a2.bar(x + (i - 1) * 0.27, v, 0.27, color=c, label=label)
for ax in (a1, a2):
    ax.set_xticks(x); ax.set_xticklabels([r.replace(" ", "\n") for r in rules])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
a1.set_ylabel("Return given up, points a year")
a1.set_title("Cost of the stop against buy and hold", fontsize=11)
a2.set_ylabel("Stop sales where holding on ended lower (%)")
a2.set_title("How often the stop sale was the better call", fontsize=11)
a2.set_ylim(0, 60)
fig.suptitle("Stop-Loss Backtest: Point-in-Time vs Today's S&P 500 Roster, Five-Year Holds 2000-2024", fontsize=12)
plt.tight_layout()
plt.savefig("stop-loss-backtest-survivorship-bias-python.png", dpi=150)
