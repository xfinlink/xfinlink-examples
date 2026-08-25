# Companion script for https://xfinlink.com/blog/broker-api-vs-data-api
"""Twenty years of daily bars and annual statements in two calls."""

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["KO", "JNJ", "PG", "MCD"]

px = xfl.prices(tickers, start="1996-01-02", end="2015-12-31", fields=["adj_close"])
fn = xfl.fundamentals(tickers, period_type="annual",
                      start="1996-01-01", end="2015-12-31", fields=["revenue"])

for t in tickers:
    bars = px[px["ticker"] == t]
    stmts = fn[fn["ticker"] == t]
    print(f"{t}: {len(bars):,} daily bars {bars['date'].min().date()} to "
          f"{bars['date'].max().date()} | {len(stmts)} annual statements")
