# Do Grain Prices Predict Food Inflation? Granger Causality Test in Python

## What's the question?

When grain commodity prices spike, food prices at the grocery store eventually follow. But how long is the lag, and is the relationship statistically causal in the Granger sense? Granger causality tests whether past values of one series improve the prediction of another beyond what the other series' own past values provide. If grain returns at lag k improve food CPI forecasts, grain prices Granger-cause food inflation. If the reverse also holds, there is a feedback loop. If only one direction is significant, the transmission is unidirectional.

## The approach

Pull the Food CPI sub-index (CPIUFDSL) from FRED and DBA (grain commodity ETF) returns from xfinlink. 10 years of monthly data. Run Granger causality in both directions at lags 1-6 months.

## Code

```python
import xfinlink as xfl
import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.stattools import grangercausalitytests

xfl.api_key = "YOUR_API_KEY"
fred = Fred(api_key="YOUR_FRED_KEY")

# Food CPI from FRED
food_cpi = fred.get_series("CPIUFDSL", observation_start="2015-01-01")
food_mom = food_cpi.pct_change().dropna().rename("food_cpi")
food_mom.index = food_mom.index.to_period("M")

# DBA monthly returns from xfinlink
df = xfl.prices("DBA", start="2015-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["close"].last()
grain_ret = monthly.pct_change().dropna().rename("grain_ret")

merged = pd.concat([food_mom, grain_ret], axis=1, join="inner").dropna()

# Granger causality: grain -> food CPI
print("Test 1: Grain returns -> Food CPI")
gc1 = grangercausalitytests(merged[["food_cpi", "grain_ret"]], maxlag=6, verbose=True)

print("\nTest 2: Food CPI -> Grain returns")
gc2 = grangercausalitytests(merged[["grain_ret", "food_cpi"]], maxlag=6, verbose=True)
```

Full script with formatting and visualisation: [granger-grains-food-cpi-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/granger-grains-food-cpi-python.py)

## Output

![Granger causality F-statistics at lags 1-6 for grain prices and food CPI](/blog-images/granger-grains-food-cpi-python.png)

```
Period: 2015-02 to 2026-04 (133 months)

Test 1: Grain returns -> Food CPI
  Lag 1: F=0.39  p=0.5308
  Lag 2: F=2.09  p=0.1278
  Lag 3: F=1.42  p=0.2407
  Lag 6: F=2.61  p=0.0209 *

Test 2: Food CPI -> Grain returns
  Lag 1: F=0.06  p=0.8004
  Lag 2: F=0.68  p=0.5075
  Lag 3: F=0.58  p=0.6303
  Lag 6: F=0.39  p=0.8825

Result: UNIDIRECTIONAL -- grain prices Granger-cause food CPI
```

## What this tells us

Grain prices Granger-cause food CPI at lag 6 (F=2.61, p=0.021). The effect is not instantaneous -- lags 1-3 are insignificant. This 6-month transmission lag reflects the time it takes for commodity price changes to flow through processing, distribution, and retail pricing. Food CPI does not Granger-cause grain returns at any lag (all p > 0.50), confirming the causal direction runs from commodities to consumer prices, not the reverse. This is consistent with a supply-chain cost-push model.

## So what?

For inflation forecasting, a 6-month spike in grain commodity prices is a leading indicator of food CPI acceleration. Central bank watchers and macro traders can use this lag to anticipate food inflation prints. The unidirectional finding also has portfolio implications: grain commodity exposure (via DBA or futures) provides a structural lead over food inflation, making it useful for inflation-hedging strategies that need to be positioned before CPI data confirms the trend.

---

Built with [xfinlink](https://xfinlink.com) -- free financial data API for US equities. No credit card, no rate limits.
