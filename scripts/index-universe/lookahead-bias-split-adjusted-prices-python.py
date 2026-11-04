# Full write-up: https://xfinlink.com/blog/lookahead-bias-split-adjusted-prices-python
#
# How much does sorting S&P 500 members on split-adjusted price levels inflate a backtest?
# At each month end from December 1999 to November 2024, the roster in force that day is
# sorted on two prices for the same date: the close that actually traded (as traded), and
# adj_close as served today, which is restated for every split, stock dividend and spin-off
# that came later (as served today). The lowest- and highest-priced fifths are held
# equal-weight for the following month, January 2000 to December 2024, on the same monthly
# total returns; only the selection differs. A "skip anything under $10" filter is run both
# ways too.
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

# The roster in force at each month end
pit = pd.concat([xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).dropna(subset=["entity_id"])
                 .assign(month=m, asof=d) for m, d in formation.items()])
pit["entity_id"] = pit["entity_id"].astype(int)
pit["removed_date"] = pd.to_datetime(pit["removed_date"])
leaving = pit["removed_date"] == pit["asof"]          # last day in the index is the formation date
pit = pit[~leaving]

# Monthly bars for every member: month-end close (as traded), adj_close (as served today),
# total return; plus recorded exit returns
ids = sorted(pit["entity_id"].unique())
parts, exits = [], {}
for i in range(0, len(ids), 16):   # 16 ids x 301 months fit on one 5,000-row page, so each
    p = xfl.prices(entity_id=ids[i:i + 16], start="1999-12-01", end="2024-12-31",   # page's
                   interval="1mo", fields=["close", "adj_close", "return_daily"],   # exits
                   max_rows=100_000)                                                # are kept
    assert len(p) < 5000, "batch spans more than one page"
    parts.append(p)
    exits.update(p.attrs.get("meta", {}).get("delisting", {}) or {})
px = pd.concat(parts)
px["month"] = pd.to_datetime(px["date"]).dt.to_period("M")
bars = px.drop_duplicates(["month", "entity_id"]).set_index(["month", "entity_id"])
ret = px.pivot_table(index="month", columns="entity_id", values="return_daily", aggfunc="first")
exit_cells = set()
for eid, ev in exits.items():
    eid, dl, m = int(eid), ev.get("delisting_return"), pd.Period(ev["event_date"], "M")
    if dl is None or eid not in ret.columns or m not in ret.index:
        continue
    r0 = ret.at[m, eid]
    ret.at[m, eid] = (1 + (0 if pd.isna(r0) else r0)) * (1 + dl) - 1
    exit_cells.add((m, eid))

# Both prices on the formation date, and the return of the following month
key = pd.MultiIndex.from_arrays([pit["month"], pit["entity_id"]])
pit["close"] = bars["close"].reindex(key).to_numpy()
pit["adj_close"] = bars["adj_close"].reindex(key).to_numpy()
pit["factor"] = pit["close"] / pit["adj_close"]       # later splits etc. folded into adj_close
pit["r"] = ret.stack().reindex(pd.MultiIndex.from_arrays(
    [pit["month"] + 1, pit["entity_id"]])).to_numpy()
no_price = pit["close"].isna() | pit["adj_close"].isna() | (pit["adj_close"] <= 0)
book = pit[~no_price].copy()
book["hold"] = book["month"] + 1
for col in ("close", "adj_close"):                    # quintile 1 = cheapest fifth that month
    book["q_" + col] = book.groupby("month")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
held = book.dropna(subset=["r"])
n_exit = sum((h, e) in exit_cells for h, e in zip(held["hold"], held["entity_id"]))

print(f"{len(formation)} month ends, {len(ids)} securities, holding months "
      f"{held['hold'].min()} to {held['hold'].max()}")
print(f"Member-months sorted: {len(book):,}; not held because the member left on the formation "
      f"date: {leaving.sum()}; no month-end price: {no_price.sum()}; no return in the holding "
      f"month: {book['r'].isna().sum()}; held member-months with a recorded delisting return: {n_exit}\n")

def ann(r):
    return ((1 + r).prod() ** (12 / len(r)) - 1) * 100

def tstat(d):
    return d.mean() / d.std() * np.sqrt(len(d))

port = {}
for col, name in [("close", "as traded"), ("adj_close", "as served today")]:
    g = held.groupby(["hold", "q_" + col])["r"].mean().unstack()
    port[name] = g
spy_h = spy_m.reindex(port["as traded"].index)
rows = []
for name, g in port.items():
    d = g[1] - g[5]
    rows.append({"sort on": name, "cheapest fifth % a year": ann(g[1]),
                 "dearest fifth % a year": ann(g[5]),
                 "cheap minus dear, points a year": d.mean() * 1200, "t-stat": tstat(d)})
print("Equal-weight quintiles sorted on month-end price, held the next month")
print(pd.DataFrame(rows).set_index("sort on").round(2).to_string())
print(f"SPY over the same {len(spy_h)} months: {ann(spy_h):.2f}% a year")
gap = port["as served today"][1] - port["as traded"][1]
print(f"Cheapest fifth, served minus traded: {gap.mean() * 1200:+.2f} points a year, "
      f"t-stat {tstat(gap):.2f}, served version ahead in {(gap > 0).mean() * 100:.0f}% of "
      f"{len(gap)} months")
over = (book["q_adj_close"] == 1) & (book["q_close"] != 1)
print(f"Cheapest-fifth picks that differ between the two sorts: "
      f"{over.groupby(book['month']).sum().mean():.0f} of "
      f"{(book['q_adj_close'] == 1).groupby(book['month']).sum().mean():.0f} a month\n")

# What moves a stock between fifths: the factor of later restatements, close / adj_close
# (above 1 after a later split, stock dividend or spin-off; below 1 after a reverse split)
mech = {}
for q, qname in [(1, "cheapest"), (5, "dearest")]:
    for col, name in [("close", "as traded"), ("adj_close", "as served")]:
        f = book.loc[book["q_" + col] == q, "factor"]
        mech[f"{qname} fifth, {name}"] = {"% restated up later": (f > 1.001).mean() * 100,
                                         "median factor if so": f[f > 1.001].median(),
                                         "% reverse split later": (f < 0.999).mean() * 100}
print(pd.DataFrame(mech).T.round(2).to_string())
print(f"by five-year period, cheapest fifth served minus traded, points a year: "
      + ", ".join(f"{k} {v:+.2f}" for k, v in (gap.groupby(
          lambda m: f"{m.year // 5 * 5}-{m.year // 5 * 5 + 4}").mean() * 1200).items()) + "\n")

# A "skip anything under $10" filter, applied to each price
excl = held[(held["adj_close"] < 10) & (held["close"] >= 10)]
incl = held[(held["adj_close"] >= 10) & (held["close"] < 10)]
both = held[(held["adj_close"] >= 10) & (held["close"] >= 10)]
print(f"Under-$10 filter, member-months: served price under $10 but traded at $10 or more: "
      f"{len(excl):,} ({excl['entity_id'].nunique()} securities); the reverse: {len(incl):,} "
      f"({incl['entity_id'].nunique()} securities); total held {len(held):,}")
print(f"  next-month return, mean: wrongly skipped {excl['r'].mean() * 100:.2f}%, wrongly kept "
      f"{incl['r'].mean() * 100:.2f}%, passing on both prices {both['r'].mean() * 100:.2f}%")
filt = {name: held[held[col] >= 10].groupby("hold")["r"].mean()
        for col, name in [("close", "as traded"), ("adj_close", "as served today")]}
fg = filt["as served today"] - filt["as traded"]
print("  equal-weight members priced $10 or more: "
      + ", ".join(f"{k} {ann(v):.2f}% a year" for k, v in filt.items())
      + f"; difference {fg.mean() * 1200:+.2f} points a year, t-stat {tstat(fg):.2f}")
print(f"  all members, no filter: {ann(held.groupby('hold')['r'].mean()):.2f}% a year\n")

# Two members, each on one date: the traded close against the adj_close served today
ex = pd.concat([book[(book["ticker"] == t) & (book["month"] == pd.Period(m, "M"))]
                for t, m in [("AAPL", "1999-12"), ("NVDA", "2001-11")]])
print("As traded vs as served today")
print(ex.assign(date=ex["asof"].dt.strftime("%Y-%m-%d"))[
    ["ticker", "date", "close", "adj_close", "factor", "q_close", "q_adj_close"]].rename(columns={
        "q_close": "fifth traded", "q_adj_close": "fifth served"}).round(4).to_string(index=False))
print(f"Months AAPL sat in the served cheapest fifth while in the traded top two fifths: "
      f"{((book['ticker'] == 'AAPL') & (book['q_adj_close'] == 1) & (book['q_close'] >= 4)).sum()}\n")

# Chart: growth of $1 in the cheapest fifth on each price, and SPY
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
x = port["as traded"].index.to_timestamp(how="end")
for series, label, color in [(port["as served today"][1], "Cheapest fifth on adj_close (as served today)", "#f59e0b"),
                             (port["as traded"][1], "Cheapest fifth on traded close", "#3b82f6"),
                             (spy_h, "SPY", "#888888")]:
    growth = (1 + series).cumprod()
    ax.plot(x, growth, color=color, lw=1.8 if label != "SPY" else 1.2, label=label)
    ax.text(x[-1], growth.iloc[-1], f"  ${growth.iloc[-1]:.1f}", color=color, va="center",
            fontsize=10)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["$0.5", "$1", "$2", "$5", "$10", "$20", "$50"])
ax.set_xlabel("Holding month")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("Look-Ahead Bias: Cheapest S&P 500 Fifth, Adjusted vs Traded Price, 2000-2024")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig("lookahead-bias-split-adjusted-prices-python.png", dpi=150)
