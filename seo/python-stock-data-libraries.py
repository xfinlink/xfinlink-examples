"""Python Stock Data Libraries: What Each Gives You

What the xfinlink Python client does on the caller's behalf: DataFrame output
with parsed dtypes, cursor pagination, multi-ticker batching in one call, and a
named warning instead of a silent empty result.
"""
import warnings

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# 1. One call, 26 years. The internal page size is 5,000 rows, so the client
#    follows the cursor until the requested range is complete.
df = xfl.prices("AAPL", start="2000-01-01", end="2025-12-31")
print(f"rows returned: {len(df):,}")
print(f"date dtype:    {df['date'].dtype}")
print(df[["date", "close", "adj_close", "volume"]].head(3).to_string(index=False))

# 2. One call, five tickers, two price fields.
panel = xfl.prices(
    ["AAPL", "MSFT", "JNJ", "XOM", "KO"],
    start="2024-01-01",
    end="2024-12-31",
    fields=["close", "volume"],
)
print(f"\nrows returned: {len(panel):,}")
print(panel.groupby("ticker").size().to_string())

# 3. A misspelled ticker names itself instead of returning silence.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    empty = xfl.prices("APPL", start="2024-01-01", end="2024-01-31")
print(f"\nrows returned: {len(empty)}")
print(caught[0].message)
