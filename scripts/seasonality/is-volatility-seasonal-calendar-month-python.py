# Full write-up: https://xfinlink.com/blog/is-volatility-seasonal-calendar-month-python
"""Does the calendar month predict how volatile a market will be?

Realized volatility is measured for every fund-month across 11 exchange-traded
funds from 2005 to 2025, compared against each fund's own average for that
year, and tested for stability between the two halves of the sample.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "DIA", "XLK", "XLE", "XLF", "XLP", "XLU"]
START, END = "2005-01-01", "2025-12-31"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

px = xfl.prices(TICKERS, start=START, end=END,
                fields=["close", "return_daily"], max_rows=200000)
px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
px["year"] = px["date"].dt.year
px["month"] = px["date"].dt.month

counts = px.groupby("ticker")["date"].agg(["count", "min", "max"])
assert counts["count"].nunique() == 1, "fund histories differ in length"
assert px["return_daily"].isna().sum() == 0
assert px.groupby("ticker")["date"].apply(lambda s: s.is_monotonic_increasing).all()

# Realized volatility: annualised standard deviation of daily returns in each
# calendar month, then compared with the same fund's average for that year.
rv = (px.groupby(["ticker", "year", "month"])["return_daily"]
        .agg(days="size", sd="std").reset_index())
rv = rv[rv["days"] >= 15].copy()
rv["vol"] = rv["sd"] * np.sqrt(252) * 100
rv["lv"] = np.log(rv["vol"])
rv["dev"] = rv["lv"] - rv.groupby(["ticker", "year"])["lv"].transform("mean")
rv["rank"] = rv.groupby(["ticker", "year"])["vol"].rank(ascending=False)
assert rv[["vol", "dev"]].notna().all().all()


def pct(series):
    """Average log deviation expressed as a percentage difference."""
    return 100 * (np.exp(series.mean()) - 1)


print(f"{' '.join(TICKERS)}")
print(f"daily returns {px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}, "
      f"{len(px):,} bars, {len(rv):,} fund-months, "
      f"{counts['count'].iloc[0]:,} bars per fund")
print("realized volatility = annualised standard deviation of daily returns "
      "inside the calendar month")
print("deviation = that month against the same fund's average for the same year")
print()

print("volatility by calendar month, 11 funds and 21 years pooled")
print(f"{'month':<6}{'avg vol':>10}{'deviation':>12}{'t':>8}{'top-3 share':>14}")
rows = []
for m in range(1, 13):
    sub = rv[rv["month"] == m]
    yearly = sub.groupby("year")["dev"].mean()
    t = yearly.mean() / (yearly.std(ddof=1) / np.sqrt(len(yearly)))
    top3 = 100 * (sub["rank"] <= 3).mean()
    rows.append((MONTHS[m - 1], sub["vol"].mean(), pct(sub["dev"]), t, top3))
    print(f"{MONTHS[m - 1]:<6}{sub['vol'].mean():>9.2f}%{pct(sub['dev']):>11.2f}%"
          f"{t:>8.2f}{top3:>13.1f}%")
print("t is computed on 21 yearly averages, so the 11 funds inside a year count once")
print("top-3 share = fund-years in which the month ranked among that year's three "
      "most volatile (25.0% if the calendar did not matter)")
print()

anova = stats.f_oneway(*[rv[rv["month"] == m]["dev"].values for m in range(1, 13)])
print(f"one-way test across the 12 months, treating every fund-month as an "
      f"independent draw: F {anova.statistic:.2f}, p {anova.pvalue:.2g}")
print()

h1 = rv[rv["year"] <= 2015].groupby("month")["dev"].mean()
h2 = rv[rv["year"] >= 2016].groupby("month")["dev"].mean()
rho = stats.spearmanr(h1, h2)
print("does the pattern repeat? deviation by half of the sample")
print(f"{'month':<6}{'2005-2015':>12}{'2016-2025':>12}")
for m in range(1, 13):
    print(f"{MONTHS[m - 1]:<6}{100 * (np.exp(h1[m]) - 1):>11.2f}%"
          f"{100 * (np.exp(h2[m]) - 1):>11.2f}%")
agree = int((np.sign(h1.values) == np.sign(h2.values)).sum())
print(f"rank correlation between the two halves {rho.statistic:.2f} "
      f"(p {rho.pvalue:.2f}); the sign agrees in {agree} of 12 months")
print()

ex = rv[~rv["year"].isin([2008, 2020])]
print("deviation excluding 2008 and 2020")
print("  ".join(f"{MONTHS[m - 1]} {pct(ex[ex['month'] == m]['dev']):.2f}%"
                for m in range(1, 13)))
mar20 = pct(rv[(rv["month"] == 3) & (rv["year"] == 2020)]["dev"])
mar_ex = pct(rv[(rv["month"] == 3) & (rv["year"] >= 2016) & (rv["year"] != 2020)]["dev"])
print(f"March 2020 alone deviates {mar20:.1f}%; the 2016-2025 March figure is "
      f"{100 * (np.exp(h2[3]) - 1):.2f}% with that year and {mar_ex:.2f}% without it")
print()

print("the two claims fund by fund: a dangerous autumn and a quiet July")
print(f"{'fund':<6}{'Sep-Oct':>10}{'Jul':>10}")
for ticker, g in rv.groupby("ticker"):
    autumn = pct(g[g["month"].isin([9, 10])]["dev"])
    july = pct(g[g["month"] == 7]["dev"])
    print(f"{ticker:<6}{autumn:>9.2f}%{july:>9.2f}%")
print()

# How much of monthly volatility does the calendar explain, against the simplest
# alternative: last month's volatility.
reg = rv.sort_values(["ticker", "year", "month"]).copy()
reg["idx"] = reg["year"] * 12 + reg["month"]
reg["lag"] = reg.groupby("ticker")["lv"].shift(1)
reg["lag_idx"] = reg.groupby("ticker")["idx"].shift(1)
reg = reg[reg["idx"] - reg["lag_idx"] == 1]
y = (reg["lv"] - reg.groupby("ticker")["lv"].transform("mean")).values
dummies = pd.get_dummies(reg["month"], drop_first=True).astype(float).values
lagged = (reg["lag"] - reg.groupby("ticker")["lag"].transform("mean")).values.reshape(-1, 1)


def r_squared(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


print(f"what explains a fund-month's volatility ({len(y):,} fund-months, "
      "each fund measured against its own average)")
print(f"{'calendar month':<28}R2 {r_squared(dummies, y):.4f}")
print(f"{'previous month volatility':<28}R2 {r_squared(lagged, y):.4f}")
print(f"{'both together':<28}R2 {r_squared(np.column_stack([dummies, lagged]), y):.4f}")

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(12)
ax.bar(x - 0.2, [100 * (np.exp(h1[m]) - 1) for m in range(1, 13)], 0.4,
       color="#3b82f6", label="2005-2015")
ax.bar(x + 0.2, [100 * (np.exp(h2[m]) - 1) for m in range(1, 13)], 0.4,
       color="#94a3b8", label="2016-2025")
ax.axhline(0, color="#3a3a3a", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(MONTHS)
ax.set_ylabel("Volatility versus the fund's own average\nfor that year (%)")
ax.set_title("Is volatility seasonal? Calendar month against the year's own average, "
             "11 funds", color="#e0e0e0")
ax.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("is-volatility-seasonal-calendar-month-python.png", dpi=150,
            facecolor="#0a0a0a")
