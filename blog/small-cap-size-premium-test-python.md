**Do Small Caps Actually Beat Large Caps? Size Premium Test in Python**

August 7, 2026 · PRICE-ANALYSIS

**What's the question?**

The size premium is the claim that small companies earn higher returns than large ones over long horizons. It has been part of standard asset pricing since Banz documented it in 1981, and it is one of the three factors in the Fama-French model that most quantitative equity research still starts from. Portfolios are tilted toward small caps on this basis, and the tilt is usually justified as compensation for bearing extra risk.

Whether the premium is still there is worth checking on its own. There is a second question hiding underneath it, though, and it turns out to matter more. "Small cap" is not a fact about the world; it is whatever a particular index provider decides to include. If two small-cap indices built from the same universe disagree about the premium, then the premium is partly a property of index construction rather than of company size.

**The approach**

Four funds cover the size range, from a common start in June 2000 through the end of 2024:

1. Take SPY for large caps, MDY for mid caps, and two different small-cap definitions: IJR, which tracks the S&P SmallCap 600, and IWM, which tracks the Russell 2000.
2. Compute annual compound return, volatility, return per unit of volatility, and worst drawdown for each on daily total returns.
3. Build rolling five-year annualised gaps against SPY and count how often each small-cap fund is ahead, which tests whether any advantage is persistent or concentrated.
4. Split the record at 2010, since the premium's disappearance is usually dated to the post-crisis period.

The two small-cap indices are the heart of the test. They target the same part of the market, but the S&P SmallCap 600 requires positive earnings in the most recent quarter and across the trailing four quarters before a company can be added, while the Russell 2000 applies no profitability screen at all. Any gap between IJR and IWM is a measure of what that one rule is worth.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = ["SPY", "MDY", "IJR", "IWM"]
rets = pd.DataFrame({
    t: (xfl.prices(t, start="2000-06-01", end="2024-12-31",
                   fields=["close", "return_daily"])
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values("date").set_index("date")["return_daily"])
    for t in FUNDS}).dropna()

yrs = (rets.index[-1] - rets.index[0]).days / 365.25
for t in FUNDS:
    r = rets[t]
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    curve = (1 + r).cumprod()
    print(f"{t}: CAGR {cagr*100:5.2f}%  vol {vol*100:5.2f}%  "
          f"ret/vol {cagr/vol:.2f}  maxDD {(curve/curve.cummax()-1).min()*100:.2f}%")

# rolling five-year annualised gap against large caps
W = 252 * 5
cum = (1 + rets).cumprod()
for t in ["IJR", "IWM"]:
    gap = ((cum[t] / cum[t].shift(W)) ** (252 / W) -
           (cum["SPY"] / cum["SPY"].shift(W)) ** (252 / W)).dropna() * 100
    print(f"{t} minus SPY: ahead in {(gap > 0).mean()*100:.1f}% of windows, "
          f"median {gap.median():.2f}pp")
```

Full script with formatting and visualisation: [small-cap-size-premium-test-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/small-cap-size-premium-test-python.py)

**Output**

![Growth of large, mid and two small-cap indices on a log scale, and the rolling five-year annualised gap of each small-cap index over the S&P 500, 2000 to 2024](/blog-images/small-cap-size-premium-test-python.png)

```
6185 common sessions, 2000-06-01 to 2024-12-31 (24.6 years)

     index                     CAGR     Vol  Ret/Vol    MaxDD
SPY  S&P 500 (large)          7.86%  19.21%     0.41  -55.20%
MDY  S&P 400 (mid)            9.20%  21.96%     0.42  -55.33%
IJR  S&P 600 (small)          9.55%  23.16%     0.41  -58.15%
IWM  Russell 2000 (small)     7.85%  24.00%     0.33  -59.03%

IJR minus SPY, rolling 5y: positive in  63.5% of 4925 windows  median   1.13pp  best  13.78pp  worst   -7.80pp
IWM minus SPY, rolling 5y: positive in  48.3% of 4925 windows  median  -0.11pp  best   9.54pp  worst   -8.72pp
MDY minus SPY, rolling 5y: positive in  53.4% of 4925 windows  median   0.49pp  best  10.70pp  worst   -7.12pp

2000-2010 (10.6y): SPY   0.61%  MDY   7.31%  IJR   8.13%  IWM   6.09%
2011-2024 (14.0y): SPY  13.70%  MDY  10.66%  IJR  10.64%  IWM   9.21%

2014-2024 only: SPY 13.06%  IJR 8.63%  IWM 7.51%
```

**What this tells us**

The answer depends on which small-cap index is used, and the difference is larger than the premium being argued about. IJR returned 9.55% a year against 7.85% for IWM. Both hold small US companies over the same 24.6 years, and they disagree by 1.70 percentage points annually, which compounds to 47% more terminal wealth over the period.

Against large caps, IJR shows a premium of 1.69 points a year and IWM shows none whatsoever, finishing 0.01 points behind SPY. The rolling windows tell the same story with more resolution: IJR is ahead of SPY in 63.5% of five-year windows with a median gap of 1.13 points, while IWM is ahead in 48.3%, a coin flip, with a median of -0.11. The profitability screen is doing the work that the size factor is usually credited for.

Risk-adjusted, even IJR's advantage disappears. Return per unit of volatility is 0.41 for IJR and 0.41 for SPY, identical to two decimal places, and IWM sits well behind at 0.33. Small caps carried about four points more annual volatility and roughly three points more drawdown at the worst moment. An investor was paid for that risk in raw return only by choosing the screened index, and was not paid at all on a risk-adjusted basis.

The premium is also concentrated in one stretch. From 2000 to 2010 large caps returned 0.61% a year while IJR returned 8.13%, a gap of seven and a half points that reflects the S&P 500 absorbing two crashes from a stretched starting valuation. Since 2011 the ordering reverses: SPY at 13.70% against 10.64% for IJR. Taking only the last eleven years, SPY returned 13.06% against 8.63%. The rolling chart shows the gap crossing below zero around 2016 and staying there.

**So what?**

Specify the index before debating the factor. A portfolio committee that approves a small-cap allocation and leaves the benchmark to the implementation team has decided less than it thinks, because the choice between the S&P 600 and the Russell 2000 mattered more here than the choice between small and large.

Prefer the screened index where a small-cap allocation is wanted. The earnings requirement is a crude quality filter, and crude quality filters have held up better than most factor refinements. It costs nothing to implement, since both funds are liquid and cheap.

Do not expect the size premium to pay on a risk-adjusted basis. Nothing in the last quarter century supports adding small caps to raise return per unit of risk; the honest case for the allocation is broader opportunity and different sector exposure, not compensation for bearing size risk. If a backtest of any size-tilted strategy shows a strong premium, check the sub-periods before trusting it, because a test that starts in 2000 inherits a decade in which large caps went nowhere.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
