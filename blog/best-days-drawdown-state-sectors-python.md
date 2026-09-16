**Are the Market's Best Days Always Rebounds? Drawdown-State Analysis Across Sectors in Python**

September 16, 2026 · PRICE-ANALYSIS

**What's the question?**

The best days in the market are rebounds inside crashes. The claim is common, and the arithmetic usually offered for it says nothing about when those gains happened, so it cannot support the claim.

Testing it needs a measure of conditions when each session arrived. Drawdown serves: the percentage distance between the current level of an investment and its highest previous level, so -40% means the fund has lost 40% of its peak and has not recovered it.

It also needs a base rate, or the answer is an illusion. Equity funds spend a great deal of time below their peak, and if a fund sits more than 10% under its high on half of all sessions, finding that its best days happened in that state is close to what chance delivers. The question is whether the best sessions concentrate far beyond the base rate, and whether that survives fund by fund rather than only on the index.

**The approach**

SPY plus the nine Select Sector SPDR funds, trading since December 1998 and splitting the S&P 500 into its sectors: ten funds, 6,791 sessions each, from 4 January 1999 to 31 December 2025.

1. Derive daily returns from split-adjusted closing prices. These are price returns, so every level below sits under a total-return figure by roughly the dividend yield.
2. Exclude any fund with a single-session move above 50% as a suspected corporate-action artefact. None here triggers it; the largest single session belongs to XLE at 20.14%.
3. Build each fund's drawdown series and read it at the previous close, so the state recorded is what an investor saw before the session happened, not after.
4. Take each fund's 20 largest gains and 20 largest losses, record the drawdown state on each, and compare against two controls: the fund's own median drawdown, and the share of its sessions spent more than 10% and 20% below the peak.
5. Check whether the extreme dates are sector-specific or shared across the ten funds.

Step 4 is the test. Without those controls the result would only restate how often these funds trade below a high.

**Code**

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

ret, draw = {}, {}
for t in FUNDS:
    px = xfl.prices(t, start="1998-12-01", end="2025-12-31", fields=["adj_close"])
    s = px.sort_values("date").set_index("date")["adj_close"]
    r = s.pct_change().dropna()
    ret[t] = r[r.index >= "1999-01-01"]
    level = (1 + ret[t]).cumprod()
    draw[t] = (level / level.cummax() - 1).shift(1)

screened = [t for t in FUNDS if ret[t].abs().max() > 0.50]

for t in ret:
    best = ret[t].nlargest(20).index
    print(t,
          round(draw[t].loc[best].median() * 100, 1),   # state on the 20 best sessions
          round(draw[t].median() * 100, 1),             # state on a typical session
          int((draw[t].loc[best] < -0.10).sum()),       # best sessions deep in a drawdown
          round((draw[t] < -0.10).mean() * 100, 1))     # base rate for that state
```

Full script with formatting and visualisation: [best-days-drawdown-state-sectors-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/best-days-drawdown-state-sectors-python.py)

**Output**

![SPY drawdown from 1999 to 2025 with its twenty best and twenty worst sessions marked, and the median drawdown on each fund's best sessions against its median on a typical session](/blog-images/best-days-drawdown-state-sectors-python.png)

```
SPY and the nine Select Sector SPDRs, price returns from split-adjusted closes
Window 1999-01-04 to 2025-12-31   funds: 10   sessions per fund: 6,791
Sessions per calendar year: min 248, median 252, max 253   every fund priced on every session: True
Largest single session in the sample: 20.14% (XLE)   funds screened out above 50%: 0

Drawdown at the previous close, by session type
fund     20 best  20 worst  all days   best below -10%
SPY       -37.3%    -32.8%     -8.3%           19 of 20
XLB       -35.5%    -28.0%    -10.8%           20 of 20
XLE       -57.3%    -46.0%    -20.2%           20 of 20
XLF       -70.7%    -62.2%    -16.1%           20 of 20
XLI       -34.8%    -25.7%     -6.9%           20 of 20
XLK       -49.1%    -50.1%    -41.4%           20 of 20
XLP       -21.1%     -9.8%     -6.5%           19 of 20
XLU       -36.7%    -27.9%    -10.8%           19 of 20
XLV       -21.2%    -15.8%     -6.7%           17 of 20
XLY       -32.0%    -28.8%     -6.0%           20 of 20

Across the ten funds: 194 of 200 best sessions and 176 of 200 worst sessions landed while the fund
was already more than 10% below its trailing peak, against 50.1% of all sessions. 167 of 200
best sessions landed more than 20% below it, against 29.3% of all sessions.

Top-20 best sessions shared by all ten funds: 2008-10-28, 2020-03-13, 2020-03-24
Of the 200 best sessions, 116 fall in 2008 (56) or 2020 (60).

Supporting: annualised price return with ten sessions removed
fund  sector                 all in  -10 best  -10 worst  -10 both
SPY   US large cap            6.54%     3.30%      9.99%     6.64%
XLB   Materials               5.40%     2.05%      9.07%     5.60%
XLE   Energy                  5.10%     0.60%     10.88%     6.13%
XLF   Financials              4.00%    -1.05%      9.28%     3.98%
XLI   Industrials             7.07%     3.87%     10.69%     7.39%
XLK   Technology              8.40%     4.07%     11.96%     7.49%
XLP   Consumer Staples        3.97%     1.64%      6.42%     4.04%
XLU   Utilities               3.92%     0.41%      7.13%     3.50%
XLV   Health Care             6.83%     3.97%      9.80%     6.85%
XLY   Cons Discretionary      8.54%     5.36%     12.29%     9.00%
```

**What this tells us**

The claim survives the base-rate control by a wide margin. Across the ten funds, 194 of the 200 best sessions landed while the fund was already more than 10% below its peak, against a base rate of 50.1%, and 167 landed more than 20% below against 29.3%. Chance would put roughly 100 and 59 sessions in those states.

The medians say it in a different unit. SPY's best sessions arrived at a median drawdown of -37.3% against -8.3% on a typical session, and financials at -70.7% against -16.1%, so the best days in XLF happened while the fund had lost roughly seven tenths of its value. Defensive sectors are milder but not exempt: consumer staples reads -21.1% against a typical -6.5%, health care -21.2% against -6.7%, the latter with 17 of 20 deep in a drawdown, the weakest count here.

XLK sets the limit of the measure. Its best sessions came at a median drawdown of -49.1%, but so did its typical session at -41.4%, because the fund spent 2000 to 2017 below its bubble peak. For technology, the base rate explains almost the whole result.

The worst sessions sit in the same place, 176 of 200, which is why removing one tail tends to remove the other. The [SPY version of that removal test is worked through in an earlier piece](/blog/missing-best-worst-days-market-timing-python); the table above extends it across sectors, where dropping the ten best sessions costs 2.33 points in consumer staples and 5.05 in financials, enough to turn XLF negative.

Sector choice dilutes none of it: 28 October 2008, 13 March 2020 and 24 March 2020 rank among the 20 best for all ten funds, and 116 of the 200 best sessions fall in two calendar years.

**So what?**

Treat the largest gains as conditional on being invested through a deep drawdown rather than as a toll paid for being invested at all. A stop that exits at -10%, or a trend rule that goes to cash after a decline, moves to safety at the threshold where 194 of these 200 sessions occur, so its cost is not a small chance of missing a scattered good day but a high chance of standing outside the window that produces them.

The base-rate comparison is the part to reuse. Any claim that the extreme days happen during X needs the share of ordinary days that also satisfy X before it means anything, and XLK shows what omitting that control does: a fund whose best days look crisis-bound until its own history is accounted for.

For allocation, the result argues against treating sector mix as a way to reduce dependence on a handful of sessions, since the dates are common to all ten funds. Position size and time horizon are the levers that move the exposure.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
