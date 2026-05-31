# Can Defensive Stocks Hedge AI Drawdowns? Basket Regime Test in Python

May 31, 2026 · PRICE-ANALYSIS

## What's the question?

AI exposure has become a large part of many growth portfolios. The risk is concentration. If the same stocks drive return, momentum, earnings revisions, and index weight, a single drawdown can affect portfolios that appear diversified on paper.

A defensive hedge should do two things during stress: lose less than the growth basket and ideally hold up better than the broad market. Defensive stocks are companies whose products are less sensitive to the economic cycle, such as household staples, beverages, healthcare, discount retail, and restaurants. They are not risk-free. The question is whether they have recently provided useful protection during AI-led selloffs.

## The approach

The AI basket is NVDA, MSFT, ORCL, AMZN, and META. The defensive basket is PG, JNJ, KO, WMT, and MCD. SPY is the market benchmark.

The test uses monthly returns because hedge decisions are usually made at portfolio-rebalancing frequency, not daily noise frequency. AI stress months are defined as the bottom decile of AI basket monthly returns. This avoids choosing an arbitrary crisis date and lets the data define the worst AI regimes.

Although five years of prices are requested, the complete common panel begins in July 2022 after requiring every ticker to have data in the same monthly return series. The output reports the actual sample used.

The analysis then compares average monthly returns for the AI basket, defensive basket, and SPY during stress months and all other months.

## Code

```python
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ai = ["NVDA", "MSFT", "ORCL", "AMZN", "META"]
defensive = ["PG", "JNJ", "KO", "WMT", "MCD"]
tickers = ai + defensive + ["SPY"]

df = xfl.prices(tickers, period="5y", fields=["close"])
prices = df.pivot_table(index="date", columns="ticker", values="close").dropna()

monthly = prices.resample("ME").last().pct_change().dropna()
monthly["AI basket"] = monthly[ai].mean(axis=1)
monthly["Defensive basket"] = monthly[defensive].mean(axis=1)
monthly["SPY"] = monthly["SPY"]

threshold = monthly["AI basket"].quantile(0.10)
stress = monthly[monthly["AI basket"] <= threshold]

print(stress[["AI basket", "Defensive basket", "SPY"]].mean())
```

Full script with formatting and visualisation: [defensive-stocks-ai-drawdown-hedge-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/defensive-stocks-ai-drawdown-hedge-python.py)

## Output

<img src="/blog-images/defensive-stocks-ai-drawdown-hedge-python.png" alt="Defensive stock basket returns during AI stress months compared with other months" style="width:100%;border-radius:8px;margin:16px 0;" />

```
=== Defensive Stocks During AI Drawdowns ===
Sample: 2022-07 to 2026-05
AI stress threshold (bottom decile): -7.7%
Stress months: 5 of 47

Average monthly returns:
AI basket, stress months:        -10.9%
Defensive basket, stress months: +1.5%
SPY, stress months:              -2.6%
Protection spread:               +12.4%

Defensive basket beat AI in 100% of stress months
Defensive basket beat SPY in 80% of stress months
Worst AI month: 2022-09  AI=-15.1%  Defensive=-5.4%  SPY=-9.6%
```

## What this tells us

The defensive basket worked in the recent sample. During the bottom decile of AI basket returns, AI stocks fell 10.9% on average. The defensive basket gained 1.5% on average. SPY fell 2.6%. The protection spread between defensive stocks and the AI basket was 12.4 percentage points.

The hit rate is also strong. The defensive basket beat the AI basket in 100% of stress months and beat SPY in 80% of stress months. The worst AI month was September 2022, when the AI basket fell 15.1%. Defensive stocks still fell 5.4%, but they lost much less than both the AI basket and SPY.

The test does not mean defensive stocks always hedge growth risk. It shows that in this recent AI-heavy market regime, a simple equal-weight basket of defensive companies has provided material drawdown protection when AI exposure was under pressure.

## So what?

Defensive stocks are a practical hedge for AI concentration risk, but they are not a perfect hedge. They reduce drawdowns by changing the economic exposure of the portfolio. They do not offset losses mechanically like a put option or short position.

For portfolio construction, the result supports a barbell approach: keep AI exposure where fundamentals justify it, but pair it with companies whose cash flows are less tied to the AI spending cycle. The hedge should be sized against stress-month behavior, not average full-period volatility. In this sample, the defensive basket earned its place because it helped exactly when AI exposure hurt.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
