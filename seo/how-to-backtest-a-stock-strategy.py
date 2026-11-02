# Full write-up: https://xfinlink.com/blog/how-to-backtest-a-stock-strategy
#
# A complete backtest on S&P 500 members, December 1999 to December 2024, monthly bars.
# The rule: at each month end, rank members on their price change over the six months that
# ended one month earlier, buy equal weights of the top 50 and hold for one month. The correct
# version uses the roster in force at each month end, total returns, recorded exit returns and
# a cost of 10 basis points on every dollar traded. Each variant then changes one thing.
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Step 2: the universe on each date, and today's roster for comparison
days = xfl.prices("SPY", start="1999-12-01", end="2024-12-31", fields=["close"])
month_ends = pd.to_datetime(days["date"]).groupby(pd.to_datetime(days["date"]).dt.to_period("M")).max()
def roster(d):
    return d.to_period("M"), set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).entity_id.dropna().astype(int))
with ThreadPoolExecutor(8) as pool:
    dated = dict(pool.map(roster, month_ends))
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

# Step 3: monthly bars keyed by entity_id; twelve ids a call keeps each response on one page
ids = sorted(set().union(*dated.values()) | today)
def fetch(i):
    df = xfl.prices(entity_id=ids[i:i + 12], start="1999-01-01", end="2024-12-31", interval="1mo",
                    fields=["adj_close", "return_daily"], max_rows=100_000)
    return df, df.attrs["meta"].get("delisting") or {}
with ThreadPoolExecutor(8) as pool:
    batches = list(pool.map(fetch, range(0, len(ids), 12)))
px = pd.concat(b[0] for b in batches).astype({"entity_id": int})
px["month"] = pd.to_datetime(px["date"]).dt.to_period("M")
close = px.pivot(index="month", columns="entity_id", values="adj_close")
total_ret = px.pivot(index="month", columns="entity_id", values="return_daily")
price_ret = close.pct_change(fill_method=None)

# Add each recorded exit return to the security's final month
with_exits = total_ret.copy()
applied = 0
for b in batches:
    for i, ev in b[1].items():
        month = pd.Period(ev["event_date"][:7], "M")
        if ev["delisting_return"] is not None and month in with_exits.index:
            last = with_exits.loc[month, int(i)]
            with_exits.loc[month, int(i)] = (1 + (0 if pd.isna(last) else last)) * (1 + ev["delisting_return"]) - 1
            applied += 1

months = sorted(dated)[:-1]  # the last roster has no following month to hold

def backtest(returns=with_exits, universe=None, lookback=6, top=50, cost_bp=10, gap=1):
    """Monthly returns of the rule. gap: months between the ranking's last close and the trade
    (0 fills at the close it ranked on; -1 ranks on the holding month itself)."""
    signal = close / close.shift(lookback) - 1
    out, weights, turnover, positions = {}, pd.Series(dtype=float), [], []
    for t in months:
        members = sorted(dated[t] if universe is None else universe)
        ranks = signal.loc[t - gap].reindex(members).dropna()
        ranks = ranks[close.loc[t].reindex(ranks.index).notna()]  # has a bar in the trade month
        picks = ranks.nlargest(top).index if top else ranks.index
        target = pd.Series(1 / len(picks), index=picks)
        all_ids = target.index.union(weights.index)
        traded = (target.reindex(all_ids, fill_value=0) - weights.reindex(all_ids, fill_value=0)).abs().sum()
        r = returns.loc[t + 1].reindex(picks).fillna(0)  # a position whose series has ended sits in cash
        out[t + 1] = (target * r).sum() - traded * cost_bp / 10_000
        weights = target * (1 + r) / (target * (1 + r)).sum()
        turnover.append(traded)
        positions += [(t + 1, i) for i in picks]
    return pd.Series(out), sum(turnover) / len(turnover), positions

def stats(r):
    wealth = (1 + r).cumprod()
    return {"% a year": (wealth.iloc[-1] ** (12 / len(r)) - 1) * 100, "volatility %": r.std() * 12 ** 0.5 * 100,
            "return/risk": r.mean() / r.std() * 12 ** 0.5, "max drawdown %": (wealth / wealth.cummax() - 1).min() * 100,
            "$1 becomes": wealth.iloc[-1]}

correct, avg_turnover, positions = backtest()
bench = xfl.prices("SPY", start="2000-01-01", end="2024-12-31", interval="1mo", fields=["return_daily"])
spy = bench.set_index(pd.to_datetime(bench["date"]).dt.to_period("M"))["return_daily"].reindex(correct.index)
members_ew = backtest(top=0)[0]
runs = {"Correct version": correct,
        "Today's roster": backtest(universe=today)[0],
        "Ranks on the holding month": backtest(gap=-1)[0],
        "Fills at the ranking close": backtest(gap=0)[0],
        "Price return, no dividends or exits": backtest(returns=price_ret)[0],
        "No trading costs": backtest(cost_bp=0)[0],
        "No exit returns": backtest(returns=total_ret)[0],
        "SPY, total return": spy,
        "All members, equal weight": members_ew}

print(f"{len(correct)} holding months, {correct.index[0]} to {correct.index[-1]}; {len(ids)} securities; "
      f"{applied} exit returns applied")
print(f"Holdings a month: 50; average turnover {avg_turnover * 100:.0f}% of the portfolio a month "
      f"(buys plus sells); members of the {months[0]} roster still on today's: {len(dated[months[0]] & today)}\n")
print(pd.DataFrame({k: stats(v) for k, v in runs.items()}).T.round(2).to_string(), "\n")

print("Split sample, % a year")
print(pd.DataFrame({f"{a}-{z}": {"Correct version": stats(correct[a:z])["% a year"], "SPY": stats(spy[a:z])["% a year"],
                                 "All members": stats(members_ew[a:z])["% a year"]}
                    for a, z in [("2000", "2012"), ("2013", "2024")]}).round(2).to_string(), "\n")

print("Parameter grid, % a year after costs (rows: holdings; columns: lookback months)")
print(pd.DataFrame({L: {n: stats(backtest(lookback=L, top=n)[0])["% a year"] for n in (25, 50, 100)}
                    for L in (3, 6, 9)}).round(2).to_string(), "\n")

for name, other in [("SPY", spy), ("all members", members_ew)]:
    gap = correct - other
    print(f"Against {name}: {gap.mean() * 1200:+.2f} points a year on average, "
          f"t-statistic {gap.mean() / gap.std() * len(gap) ** 0.5:.2f}")

exits = pd.DataFrame([ev for b in batches for ev in b[1].values() if ev["event_date"] < "2025"])
print("\nRecorded exit returns by type, 1999-2024")
print(exits.groupby("event_type")["delisting_return"].agg(["count", "median"]).round(4).to_string())
exit_month = {int(i): (pd.Period(ev["event_date"][:7], "M"), ev["event_type"]) for b in batches for i, ev in b[1].items()}
in_exit = pd.Series([exit_month[i][1] for m, i in positions if i in exit_month and exit_month[i][0] == m])
print(f"Positions of the correct version held in their exit month: {len(in_exit)} of {len(positions)}: "
      f"{in_exit.value_counts().to_dict()}")

print(f"\nMonths ahead of SPY: {(correct > spy).sum()} of {len(correct)}")

print("\nOne ticker, two companies in this price file:")
mir = px[px["ticker"] == "MIR"].groupby(["entity_id", "entity_name"])["month"].agg(["min", "max"])
print(mir.to_string())

# Chart: growth of $1
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444"})
fig, ax = plt.subplots(figsize=(10, 5))
for label, r, color in [("Same rule on today's roster", runs["Today's roster"], "#f59e0b"),
                        ("Correct version", correct, "#3b82f6"),
                        ("All members, equal weight", members_ew, "#a3a3a3"),
                        ("SPY, total return", spy, "#6ee7b7")]:
    w = (1 + r).cumprod()
    ax.plot(w.index.to_timestamp(), w, label=label, color=color, lw=1.5)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["$0.50", "$1", "$2", "$5", "$10", "$20", "$50", "$100"])
ax.set_ylabel("Growth of $1 (log scale)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("How to Backtest a Stock Strategy: One Rule, Two Rosters, 2000-2024", fontsize=12)
plt.tight_layout()
plt.savefig("how-to-backtest-a-stock-strategy.png", dpi=150)
