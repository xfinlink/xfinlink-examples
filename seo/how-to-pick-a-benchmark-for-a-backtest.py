"""Three benchmarks for one nine-year window.

Builds a cap-weighted benchmark (SPY), an equal-weighted benchmark from the
point-in-time S&P 500 roster, and an equal-weighted benchmark from today's
roster applied backwards. Returns are price returns from adj_close.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2017-01-01", "2025-12-31"
YEARS = list(range(2017, 2026))

# The roster as it stood at each prior year-end, plus the roster as it stands now.
rosters = {y: xfl.index("sp500", as_of=f"{y - 1}-12-31") for y in YEARS}
today = xfl.index("sp500")
ids = sorted({int(i) for r in rosters.values() for i in r["entity_id"].dropna()}
             | {int(i) for i in today["entity_id"].dropna()})
print(f"unique entities across 10 rosters: {len(ids)}")

# Monthly prices for every entity that appears in any of them.
frames = []
for i in range(0, len(ids), 100):
    frames.append(xfl.prices(entity_id=ids[i:i + 100], start=START, end=END,
                             interval="1mo", fields=["adj_close"], max_rows=200000))
px = pd.concat(frames, ignore_index=True)
px["month"] = px["date"].dt.to_period("M")
wide = px.pivot_table(index="month", columns="entity_id", values="adj_close")
ret = wide.pct_change()

# Names carrying a monthly move above +100% are excluded as suspected
# corporate-action artefacts.
bad = ret.columns[(ret > 1.0).any()]
ret = ret.drop(columns=bad)
print(f"priced entities: {wide.shape[1]}, excluded: {len(bad)}, kept: {ret.shape[1]}")
print(f"months: {ret.index.min()} to {ret.index.max()}")


def equal_weight(members_by_year):
    """Equal weight, rebalanced monthly, membership refreshed each January."""
    out = {}
    for m in ret.index[1:]:
        members = [c for c in members_by_year[m.year] if c in ret.columns]
        out[m] = ret.loc[m, members].dropna().mean()
    return pd.Series(out)


pit = equal_weight({y: [int(i) for i in rosters[y]["entity_id"].dropna()] for y in YEARS})
now = equal_weight({y: [int(i) for i in today["entity_id"].dropna()] for y in YEARS})

spy = xfl.prices("SPY", start=START, end=END, interval="1mo", fields=["adj_close"])
spy = spy.set_index(spy["date"].dt.to_period("M"))["adj_close"].pct_change().reindex(pit.index)

n_years = len(pit) / 12.0


def summarise(name, r):
    cum = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
    vol = r.std() * (12 ** 0.5)
    print(f"{name:<34} {cum.iloc[-1]:>8.2f}x {cagr * 100:>7.2f}% {vol * 100:>7.2f}%")
    return cum


print(f"\n{'benchmark':<34} {'growth':>9} {'ann.':>8} {'vol':>8}")
c_spy = summarise("SPY (cap-weighted)", spy)
c_pit = summarise("Equal weight, point-in-time", pit)
c_now = summarise("Equal weight, today's roster", now)

ann = lambda c: c.iloc[-1] ** (1 / n_years) - 1
print(f"\nweighting choice (SPY - PIT equal weight): {(ann(c_spy) - ann(c_pit)) * 100:.2f} pts/yr")
print(f"survivorship gap (today's roster - PIT):   {(ann(c_now) - ann(c_pit)) * 100:.2f} pts/yr")

# Chart
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
x = c_spy.index.to_timestamp()
ax.plot(x, c_now, color="#ef4444", lw=1.8, label="Equal weight, today's roster")
ax.plot(x, c_spy, color="#3b82f6", lw=2.2, label="SPY (cap-weighted)")
ax.plot(x, c_pit, color="#e0e0e0", lw=1.8, label="Equal weight, point-in-time roster")
ax.set_title("Three benchmarks for the same nine years (price return, $1 invested)",
             color="#e0e0e0", fontsize=13, pad=14)
ax.set_ylabel("Growth of $1", color="#e0e0e0")
ax.tick_params(colors="#e0e0e0")
ax.grid(alpha=0.15, color="#e0e0e0")
for s in ax.spines.values():
    s.set_color("#333333")
ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0", loc="upper left")
plt.tight_layout()
plt.savefig("how-to-pick-a-benchmark-for-a-backtest.png", dpi=150, facecolor="#0a0a0a")
