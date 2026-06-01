# Could Shorter AI Asset Lives Hit Earnings? Depreciation Stress Test in Python

June 1, 2026 · FUNDAMENTAL-ANALYSIS

## What's the question?

An AI data center is not only a growth asset. It is also a depreciating asset. Depreciation is the accounting expense that spreads the cost of a long-lived asset over its useful life. If useful-life assumptions prove too generous, companies may need to recognize higher depreciation expense or write down assets sooner than investors expected.

This matters because AI infrastructure is unusually exposed to technological obsolescence. Chips, servers, power systems, networking hardware, and buildings do not all age at the same speed. If the economic life of the fast-moving components is shorter than the accounting life, earnings can remain strong for a period and then reset lower when depreciation policy catches up.

The practical question is how much operating income would be at risk if annual depreciation and amortization rose materially.

## The approach

The test covers AMZN, MSFT, GOOG, META, ORCL, NVDA, AVGO, and TSM. The group mixes infrastructure buyers with semiconductor and hardware suppliers, which helps separate who carries the capital burden from who earns the profit pool. Built from SEC EDGAR public filings and market data, the screen uses each company's latest annual statement.

1. Pull revenue, operating income, capital expenditure, depreciation and amortization, and net property, plant, and equipment
2. Calculate capex divided by depreciation, a rough measure of how much new investment exceeds current depreciation expense
3. Estimate implied PP&E life as net PP&E divided by annual depreciation and amortization
4. Apply a 25% depreciation stress and measure the reduction in operating income

The 25% stress is not a prediction. It is a sensitivity case for shorter useful lives.

## Code

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["AMZN", "MSFT", "GOOG", "META", "ORCL", "NVDA", "AVGO", "TSM"]
fields = ["revenue", "operating_income", "capital_expenditures",
          "depreciation_amortization", "property_plant_equipment_net"]

df = xfl.fundamentals(tickers, period_type="annual", period="5y", fields=fields)
latest = df.sort_values(["ticker", "period_end"]).groupby("ticker").tail(1).set_index("ticker")

latest["capex_abs"] = latest["capital_expenditures"].abs()
latest["capex_to_depreciation"] = latest["capex_abs"] / latest["depreciation_amortization"]
latest["implied_ppe_life"] = latest["property_plant_equipment_net"] / latest["depreciation_amortization"]
latest["extra_depreciation"] = latest["depreciation_amortization"] * 0.25
latest["operating_income_hit"] = latest["extra_depreciation"] / latest["operating_income"]

print(latest[["capex_to_depreciation", "implied_ppe_life", "operating_income_hit"]])
```

Full script with formatting and visualisation: [ai-depreciation-policy-stress-test-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/ai-depreciation-policy-stress-test-python.py)

## Output

<img src="/blog-images/ai-depreciation-policy-stress-test-python.png" alt="Depreciation stress test showing operating income sensitivity for AI infrastructure companies" style="width:100%;border-radius:8px;margin:16px 0;" />

```text
=== AI Depreciation Policy Stress Test ===
Stress case: depreciation and amortization rises 25%
Latest annual periods: 2024-12-31 to 2026-01-25

Combined capex / depreciation: 2.7x
Aggregate operating-income hit: 6.1%

Company-level depreciation sensitivity:
AMZN  capex/depr= 2.0x  implied_PPE_life= 5.4y  depr=$ 65.8B  op_income_hit=20.6%  stressed_op_income=$  63.5B
TSM   capex/depr= 1.4x  implied_PPE_life= 4.9y  depr=$ 20.2B  op_income_hit=12.5%  stressed_op_income=$  35.3B
META  capex/depr= 3.7x  implied_PPE_life= 9.5y  depr=$ 18.6B  op_income_hit= 5.6%  stressed_op_income=$  78.6B
ORCL  capex/depr= 5.5x  implied_PPE_life=11.3y  depr=$  3.9B  op_income_hit= 5.5%  stressed_op_income=$  16.7B
MSFT  capex/depr= 2.9x  implied_PPE_life= 9.3y  depr=$ 22.0B  op_income_hit= 4.3%  stressed_op_income=$ 123.0B
GOOG  capex/depr= 4.3x  implied_PPE_life=11.7y  depr=$ 21.1B  op_income_hit= 4.1%  stressed_op_income=$ 123.8B
AVGO  capex/depr= 1.1x  implied_PPE_life= 4.4y  depr=$  0.6B  op_income_hit= 0.6%  stressed_op_income=$  25.3B
NVDA  capex/depr= 2.1x  implied_PPE_life= 3.7y  depr=$  2.8B  op_income_hit= 0.5%  stressed_op_income=$ 129.7B
```

## What this tells us

The aggregate picture is not catastrophic, but it is uneven. Across the group, capital expenditure is 2.7 times depreciation and amortization. A 25% depreciation stress would reduce aggregate operating income by 6.1%. That is manageable at the group level, but the average hides large company differences.

AMZN is the most sensitive company in this screen. A 25% increase in depreciation and amortization would reduce operating income by 20.6%. TSM is next at 12.5%. Both companies have large physical asset bases relative to current operating income, so a shorter useful-life assumption has a visible earnings effect.

The suppliers with the strongest current economics are much less exposed on this measure. NVDA's stress impact is 0.5% of operating income, and AVGO's is 0.6%. Their earnings risk is more tied to demand, pricing, and margins than to depreciation on their own asset base.

## So what?

Depreciation is a useful early-warning variable in the AI cycle. If companies begin shortening useful lives or discussing faster replacement cycles, the risk is not only lower accounting earnings. It is evidence that the economic life of the infrastructure may be shorter than the original investment case assumed.

For research and risk management, the companies to monitor are not necessarily the largest AI winners. They are the firms where depreciation expense is already large relative to operating income. In this screen, AMZN and TSM deserve closer earnings-quality monitoring than NVDA or AVGO for depreciation-specific risk.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
