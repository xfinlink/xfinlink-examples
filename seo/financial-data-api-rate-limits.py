# Full write-up: https://xfinlink.com/blog/financial-data-api-rate-limits
#
# Sizes a daily screener against a real index universe: how many ticker-pulls,
# rows and HTTP calls the job costs per day.

import math

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# The universe the job has to cover, taken from the index rather than guessed.
universe = len(xfl.index("sp500"))

endpoints = 2            # one prices pull, one metrics pull
refreshes_per_day = 1    # once after the close
history_years = 1
trading_days = 252
tickers_per_call = 100   # the plan's batch cap

ticker_pulls = universe * endpoints * refreshes_per_day
rows = universe * history_years * trading_days
http_calls = math.ceil(universe / tickers_per_call) * endpoints * refreshes_per_day

print(f"universe            {universe}")
print(f"ticker-pulls / day  {ticker_pulls:,}")
print(f"rows / day          {rows:,}")
print(f"HTTP calls / day    {http_calls}")
