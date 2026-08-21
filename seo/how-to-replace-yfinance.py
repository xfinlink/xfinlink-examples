# Companion script for https://xfinlink.com/blog/how-to-replace-yfinance
"""Migration checks for a script moving off yfinance: which price column is which."""

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Check 1 — NVIDIA's 10-for-1 split week. close is the as-traded price and steps
# down on the split date; adj_close is split-adjusted and does not.
px = xfl.prices("NVDA", start="2024-06-05", end="2024-06-12",
                fields=["close", "adj_close", "split_ratio"]).sort_values("date")
px["split_ratio"] = px["split_ratio"].fillna(1.0)  # 1.0 = no split that day
print(px[["date", "close", "adj_close", "split_ratio"]].to_string(index=False))

# Check 2 — five years of Verizon. adj_close measures price change only;
# return_daily is total return and includes dividends.
vz = xfl.prices("VZ", start="2021-08-02", end="2026-07-31",
                fields=["adj_close", "return_daily"]).sort_values("date")
price_return = vz["adj_close"].iloc[-1] / vz["adj_close"].iloc[0] - 1
total_return = (1 + vz["return_daily"].iloc[1:]).prod() - 1
print(f"\nVZ 2021-08-02 to 2026-07-31: price {price_return:.1%}, total {total_return:.1%}")

# Check 3 — one long DataFrame for many tickers; pivot for a wide price matrix.
wide = (xfl.prices(["AAPL", "MSFT", "NVDA"], start="2024-06-05", end="2024-06-12",
                   fields=["adj_close"])
        .pivot(index="date", columns="ticker", values="adj_close"))
print("\n" + wide.to_string())
