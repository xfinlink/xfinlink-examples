# Do Technical Indicators Say the Same Thing? Signal Agreement Analysis in Python

August 15, 2026 · TECHNICAL-ANALYSIS

## What's the question?

A common habit in discretionary trading is to require confirmation. The 50-day average crosses above the 200-day, the price sits above its 200-day average, the twelve-month return is positive, and the position feels safer for it.

All three come from one closing price series over overlapping windows. When two rules read the same data through similar arithmetic, agreement is a property of the formulas rather than evidence about the market, and confirmation means something only if the confirming signal could plausibly have disagreed.

That is measurable. Reduce each indicator to the thing a trader acts on, a long or flat state for the next session, count how often two rules sit in the same state, and compare that against what two unrelated rules would produce. The result counts the distinct opinions in a six-indicator toolkit.

## The approach

Six price-based rules run on six exchange traded funds over 5,033 sessions, from January 2005 to December 2024: SPY, IWM, EFA, EEM, TLT and GLD, spanning US large and small caps, markets abroad, long Treasuries and gold.

1. Pull split-adjusted daily closes and daily total returns, built from SEC EDGAR public filings and market data.
2. Build the states. The 50/200 cross is long while the 50-day average exceeds the 200-day, the second rule while the close sits above its 200-day average, the third while the trailing twelve-month return is positive. The 20-day breakout turns long on a close above the prior 20-day high, flat on a close below the prior 20-day low, and holds between. MACD is long while the 12/26 difference exceeds its 9-day signal line, and RSI(14) is long above 50.
3. For each pair, compute the share of sessions spent in the same state.
4. Compute what that pair would produce if the two rules were unrelated, from how often each is long on its own. Two rules each long 80% of the time agree two thirds of the time by construction.
5. Repeat on the highest-volatility decile of sessions, ranked by 21-day realised volatility.

A backtest closes the loop: the 200-day rule against a 4-of-6 vote, both acting on the previous session's state and earning nothing on flat days.

## Code

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

px = xfl.prices(["SPY", "IWM", "EFA", "EEM", "TLT", "GLD"],
                start="2003-06-01", end="2024-12-31",
                fields=["adj_close", "return_daily"], max_rows=200000)

def states(c):
    s = pd.DataFrame(index=c.index)
    s["50/200 cross"] = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)
    s["price vs 200d"] = (c > c.rolling(200).mean()).astype(float)
    s["12m momentum"] = (c / c.shift(252) - 1 > 0).astype(float)

    hi, lo = c.shift(1).rolling(20).max(), c.shift(1).rolling(20).min()
    s["20d breakout"] = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, 0.0, np.nan)),
                                  index=c.index).ffill()

    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    s["MACD"] = (macd > macd.ewm(span=9).mean()).astype(float)

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    s["RSI(14)"] = (100 - 100 / (1 + up / dn) > 50).astype(float)
    return s

c = px[px["ticker"] == "SPY"].set_index("date")["adj_close"]
s = states(c).loc["2005-01-01":].dropna()
rules, share = list(s.columns), s.mean()

for i, a in enumerate(rules):              # observed agreement against chance
    for b in rules[i + 1:]:
        chance = share[a] * share[b] + (1 - share[a]) * (1 - share[b])
        print(f"{a:>14} vs {b:<14} {(s[a] == s[b]).mean():6.1%}   chance {chance:5.1%}")
```

Full script with formatting and visualisation: [technical-indicator-signal-correlation-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/technical-indicator-signal-correlation-python.py)

## Output

<img src="/blog-images/technical-indicator-signal-correlation-python.png" alt="Heatmap of pairwise agreement between six technical rules on SPY, and a bar chart of mean agreement on all sessions versus the most volatile decile for six ETFs" style="width:100%;border-radius:8px;margin:16px 0;" />

```
Six price-based rules, 2005-01-01 to 2024-12-31, daily long or flat states

ticker   sessions   mean pairwise    all six      same, top-decile   days
                       agreement     agree        volatility         long
SPY         5033          69.9%        34.8%             62.2%          69.6%
IWM         5033          66.5%        28.6%             60.5%          62.9%
EFA         5033          66.4%        28.9%             63.4%          59.9%
EEM         5033          65.2%        26.0%             62.6%          57.2%
TLT         5033          63.5%        24.4%             65.7%          51.7%
GLD         5033          65.9%        28.5%             61.4%          59.4%

SPY pairwise agreement, share of sessions in the same state
                   50/200 cros  price vs 20  12m momentu  20d breakou         MACD      RSI(14)
50/200 cross            100.0%        89.8%        87.8%        63.7%        47.8%        64.9%
price vs 200d            89.8%       100.0%        87.3%        72.3%        52.6%        74.6%
12m momentum             87.8%        87.3%       100.0%        65.8%        51.4%        68.9%
20d breakout             63.7%        72.3%        65.8%       100.0%        64.9%        84.5%
MACD                     47.8%        52.6%        51.4%        64.9%       100.0%        71.8%
RSI(14)                  64.9%        74.6%        68.9%        84.5%        71.8%       100.0%

SPY share of sessions each rule is long: 50/200 cross 77.6%, price vs 200d 77.4%, 12m momentum 80.6%, 20d breakout 63.7%, MACD 51.2%, RSI(14) 67.2%
mean pairwise agreement if the six rules were independent: 57.3%

SPY agreement within the three trend rules 88.3%, within the three faster rules 73.7%, between the two groups 62.5%

SPY: the 200-day rule is long on 77.4% of sessions, a 4-of-6 vote on 65.8%
the two hold the same position on 85.5% of sessions
  200-day rule   return   7.20%  volatility 11.27%  Sharpe 0.64
  4-of-6 vote    return   4.56%  volatility  9.70%  Sharpe 0.47
  buy and hold   return  10.32%  volatility 19.03%  Sharpe 0.54
```

## What this tells us

The six rules are not six opinions. They are close to two.

The three trend rules agree on 88.3% of sessions, against the 57.3% independence would produce across the toolkit. Adding a 50/200 cross to a 200-day filter changes the position on one session in ten, and those sessions cluster around turns, when the faster average and the price cross the slow average at slightly different moments. That is a lag difference rather than a second opinion.

The faster rules cluster too, at 73.7% among themselves and 84.5% between the breakout and RSI, one observation counted twice: both turn long when recent closes are high against the last few weeks. Across the two groups agreement falls to 62.5%. MACD alone behaves close to independently of the trend filters, agreeing on 47.8% to 52.6% of sessions, near what its own 51.2% long rate produces by chance.

Agreement is worst when a decision matters most. On the top volatility decile the SPY mean falls from 69.9% to 62.2%, and it falls for five of the six funds, TLT excepted at 65.7%. The indicators converge in quiet markets and scatter in violent ones.

The backtest prices the confirmation habit. Requiring four of six rules keeps SPY long on 65.8% of sessions instead of 77.4%, and the two hold the same position on 85.5% of sessions anyway, so the vote is largely the 200-day rule with extra delay. It returned 4.56% a year against 7.20%, and Sharpe fell from 0.64 to 0.47.

## So what?

Count the inputs before trusting a stack of confirmations. Two rules built from the same closes over similar windows agree most of the time whatever the market does, so stacking them raises confidence without raising information. The test is in the code above: compare observed agreement against what each rule's own long rate implies, and treat anything near that baseline as a genuine second opinion.

For a systematic version, the shape is a fast reading and a slow reading, with parameters inside each group chosen for cost rather than for signal. Choosing between a 200-day filter and a 50/200 cross decides turnover and a few weeks of lag, not which reads the market better, and the 89.8% agreement rate says so plainly. Spend the effort on information the price series does not already contain.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
