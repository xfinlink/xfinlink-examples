# Companion script for https://xfinlink.com/blog/why-beta-differs-between-sources
"""One stock, four beta conventions, four different numbers."""

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["AAPL", "KO", "NVDA", "JPM", "XOM", "PG"]
px = xfl.prices(TICKERS + ["SPY"], start="2021-06-30", end="2026-06-30",
                fields=["adj_close"], max_rows=200000)
px["date"] = pd.to_datetime(px["date"])
wide = px.pivot_table(index="date", columns="ticker", values="adj_close")


def beta(returns, market, window_years, freq):
    """Ordinary least squares slope of stock returns on market returns."""
    r = returns.resample(freq).last().pct_change() if freq != "D" else returns.pct_change()
    m = market.resample(freq).last().pct_change() if freq != "D" else market.pct_change()
    cut = r.index.max() - pd.DateOffset(years=window_years)
    r, m = r[r.index > cut].dropna(), m[m.index > cut].dropna()
    joined = pd.concat([r, m], axis=1).dropna()
    return joined.cov().iloc[0, 1] / joined.iloc[:, 1].var()


rows = []
for t in TICKERS:
    raw5m = beta(wide[t], wide["SPY"], 5, "ME")
    rows.append({
        "ticker": t,
        "5y monthly": raw5m,
        "3y weekly": beta(wide[t], wide["SPY"], 3, "W"),
        "1y daily": beta(wide[t], wide["SPY"], 1, "D"),
        "5y monthly, Blume": 0.67 * raw5m + 0.33,
    })
tab = pd.DataFrame(rows).set_index("ticker")
tab["high - low"] = tab.max(axis=1) - tab.min(axis=1)

print(f"Beta against SPY, window ending {wide.index.max().date()}")
print(tab.to_string(float_format=lambda v: f"{v:8.2f}"))
print()
print(f"Median spread across the four conventions: "
      f"{tab['high - low'].median():.2f}")
