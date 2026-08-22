**Which Sectors Lead Out of a Market Bottom? Sector Recovery Analysis in Python**

August 22, 2026 · SECTOR-ROTATION

**What's the question?**

Defensive sectors earn their name on the way down. Consumer staples and health care fall less than the index in a selloff, which is the entire argument for holding them through one. The question that follows is rarely tested with the same care: what do they do afterwards.

An investor who rotates into defensives during a decline has made two decisions, not one. The first is to lose less while prices fall. The second, implicit and usually unexamined, is to hold that position through the recovery. If defensive sectors lag badly once the market turns, the protection bought on the way down is paid for on the way up, and the round trip can end behind a portfolio that did nothing.

This measures the second half. Sector performance is compared against the index over the year following each market bottom, using the sector funds that most investors would actually have traded.

**The approach**

The index is SPY on a total-return basis, from 1999 to August 2026, and the sectors are the eleven State Street sector funds.

1. Compound daily total returns into a cumulative index level and track its running maximum.
2. Mark an episode whenever the level falls 15 per cent or more below that maximum, and close the episode when a new high is reached.
3. Date the trough as the lowest point between the prior peak and the recovery.
4. From the day after each trough, compound forward returns over 63, 126 and 252 trading days for every sector fund and for SPY.
5. Subtract the index return to leave excess return, and require a fund to have traded through the entire forward window to enter an episode.

Step 5 governs the sample. Real estate and communication services became separate funds in October 2015 and June 2018, so they contribute four episodes against six for the rest, and the tables report the count alongside the average.

Averages across six episodes are fragile, so the hit rate is reported next to them: the share of episodes in which a sector beat the index, which a single extreme recovery cannot move.

**Code**

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

px = xfl.prices(["SPY"] + SECTORS, start="1999-01-01", end="2026-08-21",
                fields=["return_daily"], max_rows=300000)
px["date"] = pd.to_datetime(px["date"])
ret = px.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()

level = (1 + ret["SPY"].dropna()).cumprod()
drawdown = level / level.cummax() - 1

episodes, in_ep = [], False
for d, dd in drawdown.items():
    if not in_ep and dd <= -0.15:
        in_ep, start = True, level.loc[:d].idxmax()
    elif in_ep and dd >= 0:
        episodes.append(drawdown.loc[start:d].idxmin())
        in_ep = False

pos = {d: i for i, d in enumerate(ret.index)}
rows = []
for trough in episodes:
    i = pos[trough]
    window = ret.iloc[i + 1:i + 1 + 252]
    fwd = (1 + window).prod() - 1
    for tk in SECTORS:
        if window[tk].notna().all():
            rows.append({"trough": trough, "ticker": tk, "excess": (fwd[tk] - fwd["SPY"]) * 100})

panel = pd.DataFrame(rows)
print(panel.groupby("ticker")["excess"].agg(["mean", lambda s: (s > 0).mean()]))
```

Full script with formatting and visualisation: [sector-leadership-out-of-market-bottoms-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/sector-leadership-out-of-market-bottoms-python.py)

**Output**

```
S&P 500 drawdowns of 15% or more since 1999
        peak       trough    depth
  2000-03-24   2002-10-09   -47.5%
  2007-10-09   2009-03-09   -55.2%
  2018-09-20   2018-12-24   -19.3%
  2020-02-19   2020-03-23   -33.7%
  2022-01-03   2022-10-12   -24.5%
  2025-02-19   2025-04-08   -19.0%

Mean return against SPY from the trough (percentage points)
horizon                 3m     6m    12m  episodes_12m
sector
Technology            9.14  12.69  18.40             6
Financials           10.05   8.58  12.12             6
Industrials           5.89   4.59   9.93             6
Communication Svs    -1.42   0.74   6.23             4
Materials             4.72   3.67   5.62             6
Cons. Discretionary   0.63   3.42   4.53             6
Energy                1.67  -9.78  -0.95             6
Utilities            -4.52  -7.91 -15.46             6
Real Estate          -1.86 -10.84 -16.75             4
Health Care         -11.82 -13.18 -20.07             6
Cons. Staples       -12.90 -16.36 -24.36             6

Hit rate: share of episodes each sector beat SPY over 12 months
                      beat  n
sector
Technology           100.0  6
Financials            66.7  6
Materials             66.7  6
Industrials           66.7  6
Communication Svs     50.0  4
Cons. Discretionary   50.0  6
Energy                33.3  6
Utilities             16.7  6
Cons. Staples          0.0  6
Health Care            0.0  6
Real Estate            0.0  4

12-month excess by episode (percentage points)
sector      Communication Svs  Cons. Discretionary  Cons. Staples  Energy  Financials  Health Care  Industrials  Materials  Real Estate  Technology  Utilities
trough
2002-10-09                NaN                  2.4          -29.4   -12.4        10.3        -24.8          1.2        5.1          NaN        32.1       14.9
2009-03-09                NaN                 29.2          -26.2   -19.5        78.0        -23.5         29.0       13.2          NaN         1.3      -34.4
2018-12-24               -0.5                 -2.4           -7.0   -20.4         0.7        -10.0         -1.3       -7.9         -8.0        20.8      -11.5
2020-03-23               10.8                 13.3          -35.9    36.6        14.4        -21.6         19.2       24.8        -19.4        10.4      -33.6
2022-10-12               19.2                 -7.5          -21.6    -6.9       -11.1        -14.4         -0.9       -7.3        -18.2        23.1      -23.5
2025-04-08               -4.6                 -7.8          -26.0    16.9       -19.5        -26.1         12.5        5.8        -21.4        22.8       -4.7
```

**What this tells us**

The cost of defence is large and it is consistent. Consumer staples trailed the index by 24.36 percentage points on average over the year after a bottom, health care by 20.07, and neither beat the index in a single one of the six episodes. Utilities managed one out of six. A zero hit rate across six independent recoveries spanning the dot-com unwind, the financial crisis and the pandemic is not a statistical accident; it is the same mechanism appearing every time.

Technology is the mirror image, ahead of the index in all six episodes with an average of 18.40 points. Its consistency matters more than its size, because the per-episode column shows the lead was not built in one recovery: 32.1 points after 2002, 20.8 after 2018, 23.1 after 2022, 22.8 after 2025, with a still-positive 1.3 after 2009.

Financials require a caveat that the average hides. The 12.12-point mean rests heavily on 2009, when the sector rebounded 78.0 points ahead of the index after being the epicentre of the crash. Strip that episode and the remaining five contain two negatives, including -19.5 after the 2025 bottom. The hit rate of 66.7 per cent is the more honest summary, and it says financials usually lead, not that they lead by anything like twelve points.

The horizon columns show the effect building rather than fading. Consumer staples trail by 12.90 points at three months and 24.36 at twelve, so this is not a short reflex that reverses. Energy is the exception to every pattern here, ranging from +36.6 after 2020 to -20.4 after 2018, driven by the oil price rather than by the equity cycle.

**So what?**

Rotating into defensives during a decline requires a plan for getting out, and the exit matters more than the entry. A portfolio that moved to staples and health care near a bottom and held for a year gave back roughly 20 to 24 points against simply holding the index, which is larger than the drawdown protection those sectors typically provide.

The practical form is a rule written in advance rather than a judgement made in the moment. Setting the rotation back to the index or toward cyclicals on a mechanical trigger, such as a fixed number of weeks after the low or a recovery of a set fraction of the decline, avoids the position that feels safest precisely when it is most expensive.

Anyone building this into a strategy should test on episode counts rather than on pooled months. Six recoveries is a small sample, and the honest reading of these tables is that the direction is reliable while the magnitude is not.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
