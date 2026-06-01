# How Much AI Capex Risk Can a Portfolio Remove? Constrained Optimization in Python

June 1, 2026 · PORTFOLIO-CONSTRUCTION

## What's the question?

AI concentration risk is usually discussed as a stock-selection problem. A portfolio owns too much NVDA, MSFT, AMZN, META, or related infrastructure exposure, so the obvious response is to cut those positions. That is incomplete. The better question is how much exposure can be reduced while still keeping some participation in the theme.

A constrained optimization provides a practical answer. It starts with a universe of candidate stocks and chooses weights that minimize portfolio volatility subject to explicit limits. In this case, the limits target AI capex risk: portfolio capex intensity must fall to no more than 60% of the equal-weight portfolio, total AI basket exposure must stay below 45%, and no single stock can exceed 25%.

Capex intensity is capital expenditure divided by revenue. It is a simple proxy for how much current revenue must support long-lived investment.

## The approach

The candidate universe has two groups. The AI group is MSFT, AMZN, META, GOOG, ORCL, NVDA, and AVGO. The defensive group is PG, KO, JNJ, WMT, and MCD. Built from SEC EDGAR public filings and market data, the analysis combines two-year daily returns with latest annual fundamentals.

1. Build split-adjusted daily returns from closing prices and split ratios
2. Estimate the annualized covariance matrix from daily returns
3. Compute each company's latest capex intensity and free-cash-flow margin
4. Minimize annualized volatility under capex, AI-exposure, and single-name constraints
5. Compare the optimized portfolio with an equal-weight portfolio

The objective is not to forecast alpha. It is to show how much risk can be removed mechanically when capex exposure is treated as a portfolio constraint.

## Code

```python
import xfinlink as xfl
import pandas as pd
import numpy as np
from scipy.optimize import minimize

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ai = ["MSFT", "AMZN", "META", "GOOG", "ORCL", "NVDA", "AVGO"]
defensive = ["PG", "KO", "JNJ", "WMT", "MCD"]
tickers = ai + defensive

prices = xfl.prices(tickers, period="2y", fields=["close", "split_ratio"])

def split_adjusted_prices(df):
    pieces = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker"):
        ratio = group["split_ratio"].fillna(1.0).replace(0, 1.0)
        factor = ratio.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
        pieces.append(group.assign(adj_close=group["close"] / factor))
    adjusted = pd.concat(pieces)
    return adjusted.pivot_table(index="date", columns="ticker", values="adj_close")

adjusted = split_adjusted_prices(prices).dropna(subset=tickers)
returns = adjusted[tickers].pct_change().dropna()

fund = xfl.fundamentals(tickers, period_type="annual", period="4y",
                        fields=["revenue", "capital_expenditures", "free_cash_flow"])
latest = fund.sort_values(["ticker", "period_end"]).groupby("ticker").tail(1).set_index("ticker")
latest["capex_intensity"] = latest["capital_expenditures"].abs() / latest["revenue"]

cov = returns[tickers].cov() * 252
capex = latest.loc[tickers, "capex_intensity"].to_numpy()
equal = np.ones(len(tickers)) / len(tickers)

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    {"type": "ineq", "fun": lambda w: 0.60 * (equal @ capex) - w @ capex},
    {"type": "ineq", "fun": lambda w: 0.45 - sum(w[tickers.index(t)] for t in ai)},
]
result = minimize(lambda w: np.sqrt(w @ cov.to_numpy() @ w), equal,
                  bounds=[(0, 0.25)] * len(tickers), constraints=constraints)
print(result.x)
```

Full script with formatting and visualisation: [ai-capex-risk-portfolio-optimization-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/cross-endpoint/ai-capex-risk-portfolio-optimization-python.py)

## Output

<img src="/blog-images/ai-capex-risk-portfolio-optimization-python.png" alt="Optimized portfolio weights after reducing AI capex exposure" style="width:100%;border-radius:8px;margin:16px 0;" />

```text
=== AI Capex-Risk Portfolio Optimization ===
Return sample: 2024-06-04 to 2026-05-29 (497 trading days)
Constraints: capex intensity <= 60% of equal-weight portfolio; AI basket weight <= 45%; single name <= 25%

Equal-weight annualized volatility: 17.2%
Optimized annualized volatility:    10.5%
Equal-weight capex intensity:       13.2%
Optimized capex intensity:          7.9%
Equal-weight AI exposure:           58.3%
Optimized AI exposure:              25.6%

Optimized weights:
KO    weight=25.0%  capex_intensity= 4.4%  FCF_margin=11.0%
JNJ   weight=25.0%  capex_intensity= 5.1%  FCF_margin=20.9%
MSFT  weight=13.8%  capex_intensity=22.9%  FCF_margin=25.4%
MCD   weight=12.3%  capex_intensity= 1.3%  FCF_margin=37.9%
PG    weight= 9.4%  capex_intensity= 4.5%  FCF_margin=16.7%
GOOG  weight= 3.5%  capex_intensity=22.7%  FCF_margin=18.2%
NVDA  weight= 3.4%  capex_intensity= 2.8%  FCF_margin=44.8%
WMT   weight= 2.8%  capex_intensity= 3.8%  FCF_margin= 2.1%
AMZN  weight= 1.9%  capex_intensity=18.4%  FCF_margin= 1.1%
AVGO  weight= 1.8%  capex_intensity= 1.0%  FCF_margin=42.1%
ORCL  weight= 1.2%  capex_intensity=37.0%  FCF_margin=-0.7%
```

## What this tells us

The optimizer materially changes the risk profile. Equal weight produces 17.2% annualized volatility, 13.2% capex intensity, and 58.3% AI exposure. The optimized portfolio reduces annualized volatility to 10.5%, capex intensity to 7.9%, and AI exposure to 25.6%.

The weights explain the result. KO and JNJ reach the 25% single-name cap because they have low capex intensity and historically lower volatility in the sample. MSFT remains the largest AI-related holding at 13.8%, which reflects its high free-cash-flow margin and lower volatility relative to the more capital-intensive buyers. META receives effectively no weight because its capex intensity is high at 34.7% in this setup, even though its free-cash-flow margin is strong.

The optimizer does not remove AI exposure entirely. It keeps small weights in GOOG, NVDA, AMZN, AVGO, and ORCL, but it makes the exposure subordinate to the risk constraint.

## So what?

A portfolio does not need to make an all-or-nothing AI bubble call. The practical decision is how much capex-linked risk is acceptable. By turning that decision into a constraint, investors can preserve some exposure while forcing the portfolio away from companies whose current spending is hardest to fund.

This is useful for risk committees and portfolio managers because it converts a narrative concern into a measurable allocation rule. If the AI thesis strengthens, the capex constraint can be relaxed. If depreciation, utilization, or financing stress worsens, the same framework can tighten exposure without rewriting the portfolio process.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
