# Full write-up: https://xfinlink.com/blog/how-many-stocks-to-diversify-survivorship-bias-python
#
# How many stocks does a portfolio need, and does the answer change when the universe
# is today's S&P 500 instead of the index as it stood? At each year end from 1999 to
# 2014, 2,000 random equal-weight portfolios of 1 to 100 stocks are drawn from the
# roster in force that day (point-in-time) and from today's roster, then held for ten
# years without rebalancing. A holding that stops trading is sold at its last price and
# the proceeds go to the remaining holdings (to cash if none remain).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SIZES, DRAWS, H = [1, 5, 10, 20, 30, 50, 100], 2000, 120  # H: ten years of months
starts = pd.date_range("1999-12-01", "2014-12-31", freq="BYE")  # last business day of each year
pit = {d.year: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")).entity_id.dropna().astype(int).tolist()
       for d in starts}
today = set(xfl.index("sp500").entity_id.dropna().astype(int))

ids = sorted(set().union(*pit.values()) | today)
px = pd.concat(xfl.prices(entity_id=ids[i:i + 25], start="2000-01-01", end="2024-12-31", interval="1mo",
                          fields=["return_daily"], max_rows=100_000) for i in range(0, len(ids), 25))
ret = px.assign(month=px["date"].dt.to_period("M")).pivot(index="month", columns="entity_id",
                                                           values="return_daily")

def hold(R):
    """Buy and hold equal weights. R: (portfolios, months, stocks), NaN once a stock stops trading."""
    v, cash = np.full((R.shape[0], R.shape[2]), 1 / R.shape[2]), np.zeros(R.shape[0])
    alive, monthly = np.ones(v.shape, bool), []
    for t in range(R.shape[1]):
        r = R[:, t, :]
        alive &= ~np.isnan(r)
        gone = np.where(alive, 0, v).sum(1)
        v = np.where(alive, v, 0)
        left = v.sum(1)
        v = v * np.where(left > 0, (left + gone) / np.where(left > 0, left, 1), 1)[:, None]
        cash = cash + np.where(left > 0, 0, gone)
        before = v.sum(1) + cash
        v = v * (1 + np.nan_to_num(r))
        monthly.append((v.sum(1) + cash) / before - 1)
    return v.sum(1) + cash, np.array(monthly).T

rng, rows = np.random.default_rng(7), []
for year, members in pit.items():
    window = ret.loc[pd.Period(f"{year + 1}-01", "M"):].iloc[:H]
    trading = window.columns[window.iloc[0].notna()]
    universes = {"Point-in-time roster": [e for e in members if e in trading],
                 "Today's roster": [e for e in trading if e in today]}
    for name, u in universes.items():
        A = window[u].to_numpy()
        for n in SIZES:
            picks = np.array([rng.choice(len(u), n, replace=False) for _ in range(DRAWS)])
            wealth, monthly = hold(A[:, picks].transpose(1, 0, 2))
            rows.append(pd.DataFrame({"start": year, "roster": name, "n": n, "pool": len(u), "wealth": wealth,
                                      "vol": monthly.std(axis=1) * np.sqrt(12)}))
sims = pd.concat(rows)
sims["annual"] = sims["wealth"] ** (1 / 10) - 1

print(f"{len(pit)} start dates, {len(ids)} securities, {len(sims):,} simulated portfolios")
print(sims.groupby("roster")["pool"].mean().round(0).to_string(), "\n")
summary = sims.groupby(["n", "roster"]).agg(
    lost_money=("wealth", lambda w: (w < 1).mean()),
    median_annual=("annual", "median"),
    worst_5pct_annual=("annual", lambda a: a.quantile(0.05)),
    volatility=("vol", "mean")).unstack("roster")
print(summary.to_string(float_format=lambda x: f"{x:.2%}"), "\n")

by_start = sims[sims["n"] == 5].groupby(["start", "roster"])["wealth"].apply(lambda w: (w < 1).mean()).unstack()
print("Five-stock portfolios that lost money over ten years, by start year")
print(by_start.to_string(float_format=lambda x: f"{x:.1%}"), "\n")

# Single members of each point-in-time roster, held alone: did they lose money, and are they still in the index?
singles = []
for year, members in pit.items():
    window = ret.loc[pd.Period(f"{year + 1}-01", "M"):].iloc[:H]
    u = [e for e in members if e in window.columns[window.iloc[0].notna()]]
    singles.append(pd.DataFrame({"wealth": hold(window[u].to_numpy().T[:, :, None])[0],
                                 "still_member": [e in today for e in u]}))
singles = pd.concat(singles)
losers = singles[singles["wealth"] < 1]
print(f"Point-in-time members held alone for ten years: {len(singles):,}, lost money {len(losers) / len(singles):.1%}, "
      f"lost 90% or more {(singles['wealth'] < 0.1).mean():.1%}")
print(f"Lost money: {singles.groupby('still_member')['wealth'].apply(lambda w: (w < 1).mean()).rename({True: 'still in index', False: 'no longer in index'}).map('{:.1%}'.format).to_dict()}")
print(f"Share of the losers no longer in the index today: {(~losers['still_member']).mean():.1%}")

# Chart: chance of a ten-year loss, and average volatility, by portfolio size
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
                     "ytick.color": "#e0e0e0", "axes.edgecolor": "#444444"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
colors = {"Point-in-time roster": "#3b82f6", "Today's roster": "#f59e0b"}
for name, c in colors.items():
    a1.plot(SIZES, summary["lost_money"][name] * 100, "o-", color=c, label=name)
    a2.plot(SIZES, summary["volatility"][name] * 100, "o-", color=c, label=name)
for ax in (a1, a2):
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
    ax.set_xlabel("Stocks in the portfolio")
    ax.spines[["top", "right"]].set_visible(False)
a1.set_ylabel("Portfolios that lost money over ten years (%)")
a2.set_ylabel("Average annualised volatility (%)")
a1.set_title("Chance of a ten-year loss", fontsize=11)
a2.set_title("Volatility", fontsize=11)
a1.legend(frameon=False)
fig.suptitle("Portfolio Diversification: Point-in-Time vs Today's S&P 500 Roster", fontsize=12)
plt.tight_layout()
plt.savefig("how-many-stocks-to-diversify-survivorship-bias-python.png", dpi=150)
