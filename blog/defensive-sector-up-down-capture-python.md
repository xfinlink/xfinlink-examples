**Do Defensive Sectors Actually Defend? Up and Down Capture in Python**

August 8, 2026 · SECTOR-ROTATION

**What's the question?**

Rotating into utilities, staples, or health care is standard practice when an allocator turns cautious. The premise is that these sectors fall less than the market without giving up a matching share of the recovery, which would make the rotation close to free. If instead the reduced downside comes with proportionally reduced upside, the rotation is just a smaller position in equities wearing a sector label, and holding less of the market outright would achieve the same thing at lower cost.

Two measures separate those cases. Up capture is the average return of a sector in months when the market rose, divided by the market's own average return in those months; down capture is the same ratio over months when the market fell. A sector with 80% up capture and 50% down capture is genuinely asymmetric, keeping four fifths of the gains while taking half the losses. A sector at 80% and 80% is not asymmetric at all, whatever its reputation. Asymmetry is worth paying for, and reduced exposure is available for nothing.

**The approach**

Nine sector funds cover the S&P 500 as it was originally divided, all trading continuously since December 1998, measured against SPY from January 1999 through June 2026.

1. Pull daily closes and returns for the nine sector SPDRs and SPY, then compound to monthly returns.
2. Split the 330 months into those where SPY rose and those where it fell, and compute up and down capture for each sector.
3. Fit a separate regression of sector return on market return within each group, giving an up-market beta and a down-market beta. Capture ratios compare averages; these slopes measure how hard a sector is pulled as the market moves further.
4. Identify the four deepest peak-to-trough declines in SPY on daily data, then measure what each sector returned over those exact windows.

Step 4 exists because capture ratios weight a mild negative month and a crash equally, and protection that holds in ordinary weakness while failing in a real decline is not protection at all.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = ["XLP", "XLU", "XLV", "XLE", "XLF", "XLI", "XLK", "XLY", "XLB"]

px = xfl.prices(["SPY"] + SECTORS, start="1999-01-01", end="2026-06-30",
                fields=["close", "return_daily"], max_rows=200000)
px["date"] = pd.to_datetime(px["date"])
daily = px.pivot_table(index="date", columns="ticker", values="return_daily").dropna()
monthly = (1 + daily).resample("ME").prod() - 1

up = monthly["SPY"] > 0
dn = monthly["SPY"] < 0

for t in SECTORS:
    r = monthly[t]
    uc = r[up].mean() / monthly["SPY"][up].mean() * 100
    dc = r[dn].mean() / monthly["SPY"][dn].mean() * 100
    bu = np.polyfit(monthly["SPY"][up], r[up], 1)[0]
    bd = np.polyfit(monthly["SPY"][dn], r[dn], 1)[0]
    print(f"{t}: up {uc:.1f}%  down {dc:.1f}%  beta_up {bu:.2f}  beta_dn {bd:.2f}")

# participation in the market's four deepest daily drawdowns
curve = (1 + daily["SPY"]).cumprod()
dd = curve / curve.cummax() - 1
```

Full script with formatting and visualisation: [defensive-sector-up-down-capture-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/portfolio-construction/defensive-sector-up-down-capture-python.py)

**Output**

<CHART>

```
330 common months, 1999-01 to 2026-06
6907 daily sessions

up months: 209   down months: 121

      sector                 Up cap  Dn cap  Spread   B up   B dn    CAGR
XLU   Utilities               55.7%   34.8%   20.9pp   0.30   0.69   7.74%
XLP   Consumer Staples        56.0%   44.4%   11.6pp   0.48   0.56   6.59%
XLV   Health Care             80.3%   71.3%    9.0pp   0.59   0.70   8.35%
XLE   Energy                 100.5%   89.6%   10.9pp   0.92   1.03   8.46%
XLI   Industrials            106.1%  101.1%    5.0pp   1.16   1.24   9.51%
XLY   Cons. Discretionary    110.4%  107.2%    3.2pp   1.23   1.07   9.54%
XLB   Materials              106.4%  107.4%   -1.0pp   1.18   1.10   8.05%
XLF   Financials             102.4%  113.8%  -11.5pp   1.19   1.24   5.77%
XLK   Technology             133.4%  132.3%    1.1pp   1.33   1.22  10.60%

SPY   S&P 500                100.0%  100.0%    0.0pp   1.00   1.00   8.66%

Sector return over the market's four deepest peak-to-trough declines
          2000    2007    2020    2022
        -47.5%  -55.2%  -33.7%  -24.5%   <- SPY
XLU     -35.7%  -42.5%  -35.3%  -11.3%
XLP       1.2%  -28.5%  -24.1%  -10.6%
XLV     -17.0%  -38.3%  -27.9%  -11.6%
XLE     -23.8%  -47.9%  -56.0%   44.4%
XLI     -36.1%  -62.3%  -41.6%  -18.2%
XLY     -23.2%  -56.6%  -33.5%  -33.5%
XLB     -20.1%  -56.7%  -36.2%  -22.1%
XLF     -22.2%  -81.6%  -42.8%  -22.3%
XLK     -81.9%  -51.4%  -31.1%  -33.1%

cross-sector correlation between up capture and down capture: 0.978
lowest down capture: XLU at 34.8%, giving up 44.3pp of upside
```

**What this tells us**

Across the nine sectors, up capture and down capture have a correlation of 0.978, so a sector that takes less of the decline takes almost exactly proportionally less of the advance. Every sector sits close to the diagonal in the scatter plot. Sector defensiveness is therefore mostly a statement about how much market exposure a sector carries, not about the shape of that exposure.

Utilities come closest to genuine asymmetry, with a spread of 20.9 percentage points. The rest is thinner: staples at 11.6, energy at 10.9, health care at 9.0. Financials run the other way, capturing 113.8% of down months against 102.4% of up months, and the compounding cost shows in a 5.77% annual return against the market's 8.66% over the same 27 years.

The conditional betas complicate the utilities result. XLU moves 0.30 for each point the market gains in an up month, but 0.69 for each point it loses in a down month. Its low down capture comes from a level effect, meaning it tends to sit above the market in weak months generally, rather than from muted sensitivity to how bad a month becomes. Once a decline is severe, utilities track it at more than twice the slope they track rallies.

The drawdown table confirms this from a second direction. In 2020, when SPY fell 33.7%, utilities fell 35.3% despite having the lowest down capture of any sector, while staples held at 24.1% and health care at 27.9%. Energy is a separate case entirely: worst of all nine in 2020 at 56.0%, then up 44.4% in 2022 while the market fell 24.5%. Its defensiveness is tied to inflation rather than to market direction, which the monthly capture ratio of 89.6% averages away completely.

**So what?**

Before funding a defensive rotation, compare it against the honest alternative: holding less of the index and more cash. Utilities at 34.8% down capture and 55.7% up capture behave close to a 45% position in SPY. Sizing the index position directly delivers that exposure without concentrating in one regulated, rate-sensitive industry.

For protection against a severe decline specifically, the drawdown columns matter more than the capture ratios, and they point somewhere different. Staples and health care held up best in three of the four episodes; utilities did not hold up in 2020 at all. Run this table on the candidate sectors before sizing a tilt, and check whether the protection being bought showed up in the episodes that actually worry you, rather than in the average month.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
