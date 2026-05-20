# Does the Oil-to-Gold Ratio Signal Recessions? XLE/GLD Backtest in Python

## What's the question?

Oil prices rise with economic activity. Gold prices rise with fear, uncertainty, and inflation hedging. The ratio of oil to gold (proxied by XLE/GLD) captures the market's relative assessment of growth vs risk aversion. When the ratio falls sharply -- oil underperforming gold -- the market is pricing in weakening growth and rising risk premiums simultaneously. If this ratio carries forward-looking information, its rate of change should predict subsequent equity market performance.

## The approach

10 years of XLE, GLD, and SPY. Compute the 12-month rate of change in XLE/GLD, sort into quintiles, measure mean forward 6-month SPY total return per quintile.

## Code

```python
import xfinlink as xfl
import pandas as pd
import numpy as np

xfl.api_key = "YOUR_API_KEY"

# 10 years of XLE, GLD, SPY
tickers = ["XLE", "GLD", "SPY"]
df = xfl.prices(tickers, start="2016-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])

# Pivot to wide format
wide = df.pivot_table(index="date", columns="ticker", values="close")
wide = wide.dropna()

# Compute XLE/GLD ratio and 12-month rate of change
wide["ratio"] = wide["XLE"] / wide["GLD"]
wide["ratio_roc_12m"] = wide["ratio"].pct_change(252)

# Forward 6-month SPY return
wide["fwd_6m"] = wide["SPY"].pct_change(126).shift(-126)

# Drop NaN and bin into quintiles
analysis = wide.dropna(subset=["ratio_roc_12m", "fwd_6m"]).copy()
analysis["quintile"] = pd.qcut(analysis["ratio_roc_12m"], 5,
                                labels=["Q1 (Falling)", "Q2", "Q3", "Q4", "Q5 (Rising)"])

print(f"Current XLE/GLD ratio: {wide['ratio'].iloc[-1]:.3f}")
print(f"12-month change: {wide['ratio_roc_12m'].iloc[-1]*100:+.1f}%\n")

for q in ["Q1 (Falling)", "Q2", "Q3", "Q4", "Q5 (Rising)"]:
    sub = analysis[analysis["quintile"] == q]
    print(f"{q:14s}  fwd_ret={sub['fwd_6m'].mean()*100:+.1f}%  n={len(sub)}")
```

Full script with formatting and visualisation: [oil-gold-ratio-recession-signal-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/cross-endpoint/oil-gold-ratio-recession-signal-python.py)

## Output

![XLE/GLD ratio quintiles vs forward 6-month SPY return](/blog-images/oil-gold-ratio-recession-signal-python.png)

```
Current XLE/GLD ratio: 0.139
12-month change: -47.5%

Q1 (Falling)   fwd_ret=+13.5%  n=444
Q2             fwd_ret=+6.6%   n=444
Q3             fwd_ret=+8.1%   n=443
Q4             fwd_ret=+5.1%   n=444
Q5 (Rising)    fwd_ret=-0.1%   n=444
```

## What this tells us

The result is contrarian. When the oil-to-gold ratio is falling fastest (Q1) -- indicating gold outperforming oil, a risk-off environment -- forward 6-month SPY returns average +13.5%. This is the strongest quintile by far. When the ratio is rising fastest (Q5) -- oil outperforming gold, a risk-on environment -- forward returns are essentially zero (-0.1%). The interpretation: extreme risk-off conditions (falling ratio) tend to precede recoveries, while extreme risk-on conditions (rising ratio) tend to precede mean reversion. The current 12-month change of -47.5% places the market deep in Q1 territory.

## So what?

The XLE/GLD ratio is a contrarian macro indicator, not a trend-following one. A sharply falling ratio (gold outperforming oil) historically precedes strong equity returns over the next 6 months. A sharply rising ratio historically precedes flat-to-negative returns. For tactical allocation, a 12-month decline exceeding -30% in the ratio is a historically favorable entry point for equities. A 12-month increase exceeding +30% is a signal to reduce risk.

---

Built with [xfinlink](https://xfinlink.com) -- free financial data API for US equities. No credit card, no rate limits.
