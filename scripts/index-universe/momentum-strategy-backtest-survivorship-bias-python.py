# Full write-up: https://xfinlink.com/blog/momentum-strategy-backtest-survivorship-bias-python
#
# Momentum strategy backtest on the S&P 500, 2000-2024: the same 12-1 month
# momentum deciles run on point-in-time rosters and on today's roster.
# Rebalanced monthly, equal weight. A holding that stops trading earns its
# return through its last trading day and sits in cash for the rest of the month.
import xfinlink as xfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Formation dates: last business day of each month, Dec 1999 to Nov 2024
dates = pd.date_range("1999-12-01", "2024-11-30", freq="BME")
pit = {d.to_period("M"): set(xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))
                             .entity_id.dropna().astype(int)) for d in dates}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

# Monthly total returns for every company either universe ever holds
ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="1999-01-01", end="2024-12-31",
                          interval="1mo", fields=["return_daily"], max_rows=100_000)
               for i in range(0, len(ids), 25))
px["month"] = px["date"].dt.to_period("M")
ret = px.pivot(index="month", columns="entity_id", values="return_daily")

# 12-1 momentum: compounded return over months t-11..t-1, skipping month t
mom = np.expm1(np.log1p(ret).shift(1).rolling(11, min_periods=11).sum())
nxt = ret.shift(-1)  # return in the holding month t+1

def backtest(universe):
    rows = []
    for m in pit:
        score = mom.loc[m, [i for i in universe(m) if i in mom.columns]].dropna()
        hold = nxt.loc[m, score.index].fillna(0.0)  # no bar after an exit: cash
        top, bottom = score >= score.quantile(0.9), score <= score.quantile(0.1)
        rows.append({"month": m + 1, "names": len(score), "winners": hold[top].mean(),
                     "losers": hold[bottom].mean(), "universe": hold.mean(),
                     "winners_gone": (~score.index[top].isin(today)).mean(),
                     "losers_gone": (~score.index[bottom].isin(today)).mean()})
    return pd.DataFrame(rows).set_index("month")

cagr = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
books = {"point-in-time": backtest(lambda m: pit[m]), "today's roster": backtest(lambda m: today)}
for name, b in books.items():
    print(f"{name}: {b.names.mean():.0f} names a month, " + ", ".join(
        f"{leg} {cagr(b[leg]):.2%}" for leg in ["winners", "losers", "universe"])
        + f", winners minus universe {cagr(b.winners) - cagr(b.universe):+.2%}")
    for a, z in [("2000", "2012"), ("2013", "2024")]:
        p = b.loc[a:z]
        print(f"  {a}-{z}: winners {cagr(p.winners):.2%}, losers {cagr(p.losers):.2%}, universe {cagr(p.universe):.2%}")
pt = books["point-in-time"]
print(f"point-in-time names no longer in the index today: winners {pt.winners_gone.mean():.1%}, losers {pt.losers_gone.mean():.1%}")

# Chart: growth of $1 in each decile, both universes
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
styles = {("point-in-time", "winners"): ("#3b82f6", "-"), ("point-in-time", "losers"): ("#3b82f6", "--"),
          ("today's roster", "winners"): ("#f59e0b", "-"), ("today's roster", "losers"): ("#f59e0b", "--")}
for (name, leg), (color, ls) in styles.items():
    growth = (1 + books[name][leg]).cumprod()
    ax.plot(growth.index.to_timestamp(), growth.values, color=color, ls=ls, lw=1.8,
            label=f"{leg.capitalize()}, {name} ({cagr(books[name][leg]):.1%} a year)")
ax.set_yscale("log")
ax.set_title("Momentum Strategy Backtest: Point-in-Time vs Today's S&P 500 Roster", color="#e0e0e0")
ax.set_ylabel("Growth of $1 (log scale)", color="#e0e0e0")
ax.set_xlabel("Holding month", color="#e0e0e0")
ax.tick_params(colors="#e0e0e0")
for s in ax.spines.values():
    s.set_color("#444444")
ax.legend(facecolor="#0a0a0a", edgecolor="#444444", labelcolor="#e0e0e0", fontsize=9)
plt.tight_layout()
plt.savefig("momentum-strategy-backtest-survivorship-bias-python.png", dpi=150, facecolor=fig.get_facecolor())
