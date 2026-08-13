**What Growth Is Already Priced In? Reverse DCF on the S&P 500 in Python**

August 13, 2026 · FUNDAMENTAL-ANALYSIS

**What's the question?**

A discounted cash flow model asks what a company is worth given a forecast. Forecasts are the weak part. Change the growth assumption by two percentage points and the valuation moves by half, which means a DCF mostly reports the opinion of whoever built it.

A reverse DCF turns the model around. Rather than forecasting growth and solving for value, it takes the market price as given and solves for the growth rate that would justify it. The output is a statement about what the market has already agreed to, expressed in a unit that can be argued with: a growth rate, checkable against the economy, the sector, or the company's own history.

The question here is what that number looks like across the whole S&P 500. If the typical company is priced for growth close to the economy's rate, the index is making an ordinary assumption. If the typical company needs to outgrow the economy forever, the index relies on something that cannot be true of all its members at once.

**The approach**

The simplest version of the model treats a company as a perpetuity: a stream of free cash flow growing at a constant rate `g`, discounted at a required return `r`. Price equals next year's cash flow divided by the difference between the two, which rearranges into a direct solution for `g`.

1. Take the current S&P 500 by entity identifier, then pull the latest annual free cash flow and market capitalisation for each member.
2. Exclude Financials and Real Estate, whose cash flow statements are not comparable to an operating business on this measure, and require positive free cash flow, since a perpetuity cannot be inverted through a negative number.
3. Solve the perpetuity for `g` at a 9% cost of equity.
4. Compare the distribution against two reference points: inflation alone, and long-run nominal growth in the wider economy.
5. Re-run the whole calculation at other discount rates to see how much of the answer is the market's and how much is the analyst's.

Step 5 is not a footnote. The discount rate is an assumption, and a reverse DCF that reports a single number without showing its sensitivity has hidden the most important input.

**Code**

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

idx = xfl.index("sp500", as_of="2026-06-30", limit=1000)
ids = [int(i) for i in idx["entity_id"].dropna().unique()]

fun = xfl.fundamentals(entity_id=ids, period_type="annual",
                       start="2024-06-01", end="2026-06-30",
                       fields=["free_cash_flow", "revenue", "gics_sector"])
met = xfl.metrics(entity_id=ids, period_type="annual",
                  start="2024-06-01", end="2026-06-30",
                  fields=["market_cap"])

latest = lambda d: d.sort_values("period_end").groupby("entity_id").tail(1)
df = (latest(fun).set_index("entity_id")
      .join(latest(met).set_index("entity_id")[["market_cap"]], how="inner"))

df = df[~df["gics_sector"].isin(["Financials", "Real Estate"])]
df = df[(df["free_cash_flow"] > 0) & (df["market_cap"] > 0)]

# P = FCF * (1 + g) / (r - g)   =>   g = (P*r - FCF) / (P + FCF)
def implied_growth(price, fcf, r):
    return (price * r - fcf) / (price + fcf)

df["implied_g"] = implied_growth(df["market_cap"], df["free_cash_flow"], 0.09)
print(df["implied_g"].median())
```

Full script with formatting and visualisation: [reverse-dcf-implied-growth-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/reverse-dcf-implied-growth-python.py)

**Output**

```
Current members with a recent annual record: 503
  excluding Financials and Real Estate: 396
  with positive free cash flow:        342

Implied perpetual free cash flow growth at a 9% cost of equity
  10th   0.04%   25th   2.81%   median   4.46%   75th   6.04%   90th   7.07%
  above  2.0% (inflation alone):  80.7% of companies
  above  4.5% (long-run nominal GDP growth):  49.1% of companies

Median implied growth by sector (10 or more companies)
  Materials                  5.51%   n= 22
  Information Technology     5.06%   n= 68
  Industrials                4.72%   n= 75
  Consumer Discretionary     4.58%   n= 49
  Health Care                4.35%   n= 54
  Communication Services     4.13%   n= 17
  Consumer Staples           3.87%   n= 35
  Energy                     1.93%   n= 15

Sensitivity: the same prices under a different cost of equity
  r = 8%   median   3.50%   share above 4.5%:  34.2%
  r = 9%   median   4.46%   share above 4.5%:  49.1%
  r = 10%   median   5.42%   share above 4.5%:  66.1%
  r = 11%   median   6.38%   share above 4.5%:  76.3%
```

**What this tells us**

The median company is priced for 4.46% perpetual free cash flow growth, landing almost exactly on long-run nominal growth in the wider economy. At a 9% required return, the typical large American company is priced to grow with the economy and no faster.

The distribution around that median carries the information. A quarter of the sample sits below 2.81%, close to inflation and implying no real growth at all, while the top decile needs better than 7.07% forever. Just under half the companies, 49.1%, require growth above the economy's rate in perpetuity, and each can only deliver that by taking share from something else.

Sector medians line up with how these businesses are usually described. Energy sits lowest at 1.93%, priced roughly for inflation, consistent with a market treating the cash flow as real but the volumes as finite. Materials at 5.51% and Information Technology at 5.06% sit at the other end. That gap of about three and a half points compounds over twenty years into a factor of two in cash flow.

The sensitivity table deserves more attention than the headline. Moving the cost of equity from 8% to 11% moves the median implied growth from 3.50% to 6.38%, so each percentage point added to the discount rate adds roughly one point to the growth the same prices appear to demand. The share of the index needing above-economy growth swings from 34.2% to 76.3% across that range. The market supplies the price; the analyst supplies the discount rate, and it moves the answer about as much as the price does.

**So what?**

Use reverse DCF as a translation tool, not a valuation. The useful output is a sentence of the form "at this price and this required return, the company must grow cash flow at x% forever", which is a claim that can be checked against the company's history, its addressable market and its competitive position. It replaces a debate about whether a stock is expensive with a debate about a specific number.

Fix the discount rate before comparing companies, and keep it fixed. The sensitivity above shows that a cross-sectional ranking is only meaningful under a common `r`, and that a single-stock implied growth figure quoted without its discount rate carries almost no information.

Treat the 49.1% figure as a portfolio-level warning rather than a stock-level one. Any individual company can outgrow the economy permanently; half the index cannot do it at once without the economy itself growing faster. For a screen meant to find prices embedding modest expectations, the practical filter is implied growth below nominal GDP, which narrows the field to about half the index before any judgement about quality applies.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
