**What Happens to Stocks Removed From the S&P 500? Replacement Pair Analysis in Python**

August 6, 2026 · INDEX-RESEARCH

**What's the question?**

When a company leaves the S&P 500, every fund tracking the index has to sell it on the same day. The standard account of that moment, running from Harris and Gurel (1986) through Chen, Noronha and Singal (2004), treats the selling as mechanical: price falls because supply arrives at once, not because anything changed at the company, and it reverses once the flow is absorbed. If that still holds, a deleted stock is a buy, best expressed against the company that replaced it.

The trade assumes the deleted company is still there to buy the next morning.

Two different events are both called a deletion. In one the company disappears, through a merger, a take-private, or failure. In the other it is demoted while it carries on as a listed stock. Only the second can be bought, and the event log does not separate them.

**The approach**

The sample is every S&P 500 membership change from 2015 to 2024, narrowed to the dates on which exactly one company entered and exactly one left: 96 one-for-one swaps. Pairing fixes the comparison, since the deleted company is measured against a replacement the same committee chose that day.

1. Follow both companies by entity identifier, not by ticker. Symbols get reassigned: FTI belonged to FMC Technologies, then to TechnipFMC, and both were removed inside this sample.
2. Pull daily prices for both legs, from three weeks before the effective date to fourteen months after.
3. Classify each removal by how far its price history runs past that date: stopped at the swap (five trading days or fewer), stopped later in the year, or still trading after 252 days.
4. For that third group, take the 252-trading-day return from the close on the effective date and subtract the S&P 500 over the same days. The spread is the removed company's figure minus the added one's.
5. Keep only the pairs whose year holds no corporate action beyond a plain split. The return recomputed from raw close and cumulative split ratio must match the adjusted-close return, which a split cancels out of and nothing else does; and no single day's total return may part from its price return by more than 2%, the mark of a distribution the price series does not carry. Six pairs fail and are set aside.

**Code**

```python
import numpy as np
import pandas as pd
from scipy import stats
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

HOLD = 252

ev = xfl.index_events("sp500", start="2015-01-01", end="2024-12-31")
ev["eff"] = pd.to_datetime(ev["effective_date"])
ev = ev.dropna(subset=["entity_id"])
ev["entity_id"] = ev["entity_id"].astype(int)

counts = ev.groupby(["eff", "event_type"]).size().unstack(fill_value=0)
swaps = ev[ev["eff"].isin(counts[(counts["added"] == 1) & (counts["removed"] == 1)].index)]
pairs = (swaps.pivot_table(index="eff", columns="event_type", values="entity_id",
                           aggfunc="first").dropna().astype(int))

spy = xfl.prices("SPY", start="2014-12-01", end="2026-02-01", fields=["adj_close"])
spy = spy.set_index("date")["adj_close"].sort_index()

def series(df, eid, eff):
    s = df[df["entity_id"] == eid].drop_duplicates("date").set_index("date").sort_index()
    s = s[s["adj_close"] > 0]
    prior = s.index[s.index <= eff]
    if len(prior) == 0 or (eff - prior[-1]).days > 7:
        return None
    return s[s.index >= prior[-1]]

def raw_return(s):                      # same window, from close and split ratio
    factor = s["split_ratio"].fillna(1.0).values[1:HOLD + 1].prod()
    return s["close"].values[HOLD] * factor / s["close"].values[0] - 1.0

state, rows = [], []
for eff, r in pairs.iterrows():
    df = xfl.prices(entity_id=[r["added"], r["removed"]],
                    start=(eff - pd.Timedelta(days=20)).date().isoformat(),
                    end=(eff + pd.Timedelta(days=430)).date().isoformat(),
                    fields=["close", "adj_close", "split_ratio"])
    add, rem = series(df, r["added"], eff), series(df, r["removed"], eff)
    if rem is None:
        continue
    n_after = len(rem) - 1
    state.append("stopped trading at the swap" if n_after <= 5 else
                 "stopped trading within the year" if n_after < HOLD else
                 "still trading a year later")
    if state[-1] != "still trading a year later" or add is None or len(add) < HOLD + 1:
        continue

    mkt = spy[spy.index >= add.index[0]].values[:HOLD + 1]
    mkt = mkt / mkt[0] - 1.0
    pa = add["adj_close"].values[:HOLD + 1] / add["adj_close"].values[0] - 1.0
    pr = rem["adj_close"].values[:HOLD + 1] / rem["adj_close"].values[0] - 1.0
    if max(abs(raw_return(add) - pa[-1]), abs(raw_return(rem) - pr[-1])) > 0.005:
        continue                        # a distribution the price series cannot carry
    rows.append({"add_exc": pa[-1] - mkt[-1], "rem_exc": pr[-1] - mkt[-1]})

res = pd.DataFrame(rows)
res["spread"] = res["rem_exc"] - res["add_exc"]
t, p = stats.ttest_1samp(res["spread"], 0.0)
print(pd.Series(state).value_counts().to_string())
print(f"n={len(res)}  removed {res['rem_exc'].median():+.2%}  "
      f"added {res['add_exc'].median():+.2%}  spread {res['spread'].mean():+.2%}  "
      f"t={t:.2f}  p={p:.4f}")
```

Full script with formatting and visualisation: [sp500-replacement-pairs-deletion-returns-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-replacement-pairs-deletion-returns-python.py)

**Output**

![Stacked bars showing that most S&P 500 removals in one-for-one swaps stopped trading within the year, and a line chart of median market-adjusted returns after the swap for added and removed companies](/blog-images/sp500-replacement-pairs-deletion-returns-python.png)

```
one-for-one replacement dates 2015-2024: 96, of which 92 carry a usable price history for both legs
stopped trading at the swap        62
still trading a year later         28
stopped trading within the year     2

removed leg 12m excess: mean -18.01%  median -28.45%
added   leg 12m excess: mean +6.63%  median -3.33%
pair spread (removed minus added), n=20: mean -24.64%  median -25.05%  t=-2.55  p=0.0197  removed leg wins 30.0%
wilcoxon p=0.0240; sign test p=0.1153; middle 80% of spreads -88.3% to +34.2%
removed leg vs the market along the way: -4.6% at day 21, -6.8% at day 63, -16.1% at day 126, -28.4% at day 252
robustness: trimmed mean spread (widest each way removed, n=18) -24.06%; same pairs measured on total return mean -23.44% median -27.87%

pairs set aside for a distribution inside the year: 6 (Windstream Holdings Inc, Noble Corp Plc, Johnson Controls Inc, Wyndham Worldwide Corp, Jefferies Financial Group Inc, Technipfmc Plc)
on the 20 pairs kept, the adjusted-close return and the raw-close recompute differ by at most 3.11e-15, and no single day's total return parts from its price return by more than 1.96%
  set-aside margins: Windstream Holdings Inc 50.9% / 2.2%; Noble Corp Plc 0.0% / 2.8%; Johnson Controls Inc 0.0% / 10.4%; Wyndham Worldwide Corp 45.9% / 1.1%; Jefferies Financial Group Inc 0.0% / 6.7%; Technipfmc Plc 25.5% / 0.2%

widest five pairs and narrowest three, 12-month return vs the S&P 500
  2023-03-20  LUMEN TECHOLOGIES INC    (LUMN )  -61.1%   vs  FAIR ISAAC CORP        (FICO )  +50.3%   spread -111.4%
  2017-04-04  SOUTHWESTERN ENERGY CO   (SWN  )  -61.5%   vs  D X C TECHNOLOGY CO    (DXC  )  +34.4%   spread  -95.8%
  2016-04-18  TENET HEALTHCARE CORP    (THC  )  -62.5%   vs  ULTA BEAUTY INC        (ULTA )  +24.9%   spread  -87.4%
  2018-11-13  E Q T CORP               (EQT  )  -58.2%   vs  HENRY JACK & ASSOC INC (JKHY )   -7.0%   spread  -51.2%
  2017-03-01  PITNEY BOWES INC         (PBI  )  -21.2%   vs  C B O E GLOBAL MARKETS (CBOE )  +28.8%   spread  -50.0%
  2019-01-18  P G & E CORP             (PCG  )  +58.7%   vs  TELEFLEX INC           (TFX  )  +24.5%   spread  +34.1%
  2016-03-04  CONSOL ENERGY INC        (CNX  )  +29.8%   vs  AMERICAN WATER WORKS C (AWK  )   -4.7%   spread  +34.5%
  2020-03-03  CIMAREX ENERGY CO        (XEC  )  +72.4%   vs  INGERSOLL RAND INC     (IR   )  +20.6%   spread  +51.8%
members on 2016-01-01: 501; today: 504; in the 2016 roster and not in today's: 159
```

**What this tells us**

Deletion is usually not a demotion. Of the 92 usable swaps, 64 removed companies stopped trading inside the year, most on the swap day itself. The list reads as a decade of takeovers: Whole Foods bought by Amazon in 2017, Monsanto by Bayer in 2018, Xilinx by AMD in 2022, Twitter taken private that year. First Republic is the exception that was not a sale; the bank failed in May 2023. In 2022 and 2024 every removal in a swap was a company that ceased to trade.

For the 20 pairs that clear both checks, the rebound does not appear. The removed company's median market-adjusted return over the next 252 days was -28.45%, against -3.33% for its replacement. The mean spread of -24.64% carries a t-statistic of -2.55 and p = 0.0197, and the signed-rank test agrees at p = 0.0240. Counting winners does not: the removed leg is ahead in 6 of 20 pairs, a sign test p of 0.12, so the finding rests on the size of the gaps rather than how often they fall one way. Removing the widest pair on each side leaves -24.06%; on total return the figure is -23.44%.

The path rules out the price-pressure explanation, which would put the trough in the first weeks and a recovery after it. The median removed company is 4.6% behind the market after a month and 6.8% behind after a quarter, then loses most of the ground later, reaching -28.4% at twelve months. That is a business in decline the committee noticed late: Lumen, Southwestern Energy, Tenet, EQT, Goodyear.

**So what?**

Buying deletions is not the trade. A committee that removes a listed company is confirming a decline the market has already priced and, on this evidence, has not finished pricing. Sizing deserves restraint: 20 observations across ten years, a tenth-to-ninetieth percentile spread of -88.3% to +34.2%, and a win count that proves nothing on its own. Read it as a reason not to hold a deleted name, and a caution against buying forced-sale flow.

The wider point applies to any research keyed on index membership. Two thirds of removals are corporate events, not portfolio decisions, so a panel that follows tickers will splice an acquirer onto a dead series and call the result a return. Following entity identifiers, and classifying each removal first, decides whether the sample is what it claims to be.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
