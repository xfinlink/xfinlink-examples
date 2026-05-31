# Is Apollo More Balance-Sheet Sensitive Than Peers? Leverage Screen in Python

May 31, 2026 · FUNDAMENTAL-ANALYSIS

## What's the question?

Apollo is often grouped with alternative asset managers such as Blackstone, KKR, Ares, Blue Owl, and traditional public asset managers. That comparison can be misleading if one company carries a much larger balance sheet relative to equity. A fee-light asset manager earns management and performance fees on third-party assets. A balance-sheet-intensive financial company is more exposed to asset values, liabilities, funding costs, and capital rules.

This distinction matters in debates about private credit and reinsurance. Insurance and reinsurance structures can make an asset manager look less like a pure fee business and more like a financial balance sheet. Public filings cannot settle every structural question, but they can show whether Apollo's public company financials are unusually asset-heavy compared with peers.

## The approach

The screen compares APO, KKR, BX, BLK, ARES, OWL, BEN, and TROW. The core metric is total assets divided by total equity. This is not a regulatory capital ratio. It is a simple accounting leverage measure: how many dollars of assets sit on the balance sheet for each dollar of book equity.

The analysis also computes revenue-to-assets and operating-cash-flow-to-assets. These measures show how efficiently the asset base produces revenue and cash flow. A high asset multiple with low cash flow productivity is more sensitive to asset marks and financing conditions than a low asset multiple with high cash flow productivity.

## Code

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["APO", "KKR", "BX", "BLK", "ARES", "OWL", "BEN", "TROW"]
fields = ["revenue", "net_income", "operating_cash_flow",
          "total_assets", "total_equity"]

df = xfl.fundamentals(tickers, period_type="annual", period="3y", fields=fields)
latest = df.sort_values("period_end").groupby("ticker").tail(1).copy()

latest["assets_to_equity"] = latest["total_assets"] / latest["total_equity"]
latest["revenue_to_assets"] = latest["revenue"] / latest["total_assets"]
latest["ocf_to_assets"] = latest["operating_cash_flow"] / latest["total_assets"]

print(latest.sort_values("assets_to_equity", ascending=False)[
    ["ticker", "assets_to_equity", "revenue_to_assets", "ocf_to_assets"]
])
```

Full script with formatting and visualisation: [apollo-peer-balance-sheet-risk-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/apollo-peer-balance-sheet-risk-python.py)

## Output

<img src="/blog-images/apollo-peer-balance-sheet-risk-python.png" alt="Apollo balance-sheet intensity compared with alternative asset manager peers" style="width:100%;border-radius:8px;margin:16px 0;" />

```
=== Asset-Manager Balance-Sheet Sensitivity Screen ===
Latest annual period per company (max period_end 2025-12-31)

APO assets/equity: 19.7x
Peer median assets/equity: 5.5x
APO multiple vs peer median: 3.6x

Balance-sheet ranking:
APO   assets=$  460.9B  equity=$  23.3B  assets/equity= 19.7x  revenue/assets= 7.0%  OCF/assets= 1.6%
KKR   assets=$  410.1B  equity=$  30.9B  assets/equity= 13.3x  revenue/assets= 4.7%  OCF/assets= 0.1%
ARES  assets=$   28.6B  equity=$   4.3B  assets/equity=  6.7x  revenue/assets=19.6%  OCF/assets=11.4%
OWL   assets=$   12.5B  equity=$   2.2B  assets/equity=  5.7x  revenue/assets=23.0%  OCF/assets=10.1%
BX    assets=$   47.7B  equity=$   8.7B  assets/equity=  5.5x  revenue/assets=30.3%  OCF/assets= 9.8%
BLK   assets=$  170.0B  equity=$  55.9B  assets/equity=  3.0x  revenue/assets=14.2%  OCF/assets= 2.3%
BEN   assets=$   32.4B  equity=$  12.1B  assets/equity=  2.7x  revenue/assets=27.1%  OCF/assets= 3.3%
TROW  assets=$   14.3B  equity=$  10.9B  assets/equity=  1.3x  revenue/assets=51.0%  OCF/assets=12.2%
```

## What this tells us

Apollo is materially more balance-sheet intensive than the median peer in this sample. APO reports $460.9B of total assets against $23.3B of equity, an assets-to-equity multiple of 19.7x. The peer median is 5.5x. Apollo's multiple is 3.6 times the peer median.

KKR is also balance-sheet intensive at 13.3x, so Apollo is not the only asset-heavy alternative manager. The contrast is clearest against TROW, BEN, BLK, and BX. TROW has only 1.3x assets to equity and produces revenue equal to 51.0% of assets. Apollo produces revenue equal to 7.0% of assets and operating cash flow equal to 1.6% of assets.

This does not prove that Apollo's asset quality is weak. It does show that Apollo should not be analyzed as a simple fee manager. A large asset base means changes in credit spreads, asset marks, insurance liabilities, and funding costs can matter more than they would for a lighter asset manager.

## So what?

The reinsurance and private-credit debate should start with balance-sheet classification. If Apollo is treated as a fee-light manager, the risk model understates its sensitivity to assets and liabilities. If it is treated as a balance-sheet financial company with an asset-management overlay, the risk model is more realistic.

For portfolio construction, this changes peer grouping. APO belongs in a stress test with KKR and other balance-sheet-sensitive financials, not only with traditional asset managers. The most useful next step is to track asset growth, equity growth, and operating cash flow together. A rising assets-to-equity multiple without improving cash flow productivity would be the warning sign.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
