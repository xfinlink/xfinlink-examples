# Did the AI Boom Come From Existing S&P 500 Members? Point-in-Time Momentum Test in Python

June 1, 2026 · INDEX-RESEARCH

## What's the question?

AI performance can look obvious in hindsight. The hard question is whether the trade was investable before the winners became obvious. A survivorship-biased backtest answers the wrong question because it starts with today's winners and assumes they were all available in the same benchmark at the beginning of the test.

Point-in-time index membership fixes that problem. It asks which stocks were actually in the S&P 500 on a specific historical date, then measures performance only from that eligible set. This matters for institutional investors because benchmark-relative portfolios can usually own constituents more easily than future additions or non-index names.

The question is whether the AI boom since early 2023 was driven mainly by existing S&P 500 members or by stocks outside the index at the start.

## The approach

The test uses January 3, 2023 as the index date. The AI candidate list includes NVDA, AVGO, MSFT, AMZN, META, GOOG, ORCL, AMD, ANET, VRT, SMCI, TSM, and PLTR. Built from SEC EDGAR public filings and market data, the analysis combines historical S&P 500 constituents with `adj_close` price returns.

1. Pull S&P 500 membership as of January 3, 2023
2. Split the AI candidate list into point-in-time members and non-members
3. Pull daily prices from that date to the latest available trading day
4. Use `adj_close` before calculating cumulative returns
5. Compare equal-weight returns for member AI stocks, non-member AI stocks, and SPY

The adjusted close field is essential because NVDA and AVGO both had major stock splits during the sample.

## Code

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

as_of = "2023-01-03"
ai = ["NVDA", "AVGO", "MSFT", "AMZN", "META", "GOOG", "ORCL",
      "AMD", "ANET", "VRT", "SMCI", "TSM", "PLTR"]

constituents = xfl.index("sp500", as_of=as_of, limit=700).drop_duplicates("ticker")
members = [ticker for ticker in ai if ticker in set(constituents["ticker"])]
non_members = [ticker for ticker in ai if ticker not in set(constituents["ticker"])]

prices = xfl.prices(ai + ["SPY"], start=as_of, fields=["adj_close"])
adjusted = prices.pivot_table(index="date", columns="ticker", values="adj_close").ffill()
returns = adjusted / adjusted.iloc[0] - 1

print(returns.iloc[-1][members].mean())
print(returns.iloc[-1][non_members].mean())
print(returns.iloc[-1]["SPY"])
```

Full script with formatting and visualisation: [sp500-ai-momentum-survivorship-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/sp500-ai-momentum-survivorship-python.py)

## Output

<img src="/blog-images/sp500-ai-momentum-survivorship-python.png" alt="Point-in-time AI momentum returns for S&P 500 members and non-members" style="width:100%;border-radius:8px;margin:16px 0;" />

```text
=== Point-in-Time AI Momentum Test ===
Index date: 2023-01-03
Price sample: 2023-01-03 to 2026-06-02 (856 trading days)

AI tickers already in the S&P 500: NVDA, AVGO, MSFT, AMZN, META, GOOG, ORCL, AMD, ANET
AI tickers outside the S&P 500 then: VRT, SMCI, TSM, PLTR

Equal-weight return, point-in-time S&P 500 AI members: +509%
Equal-weight return, non-member AI basket:             +1009%
SPY return over same window:                           +99%

AI ticker returns since the index date:
PLTR   +2281%  non-member
NVDA   +1457%  member
AVGO    +770%  member
VRT     +754%  non-member
AMD     +715%  member
TSM     +503%  non-member
SMCI    +498%  non-member
ANET    +480%  member
META    +379%  member
GOOG    +303%  member
AMZN    +199%  member
ORCL    +192%  member
MSFT     +84%  member

Top contributors within the point-in-time member basket:
NVDA  contribution=+162%
AVGO  contribution=+86%
AMD   contribution=+79%
ANET  contribution=+53%
META  contribution=+42%
```

## What this tells us

The AI boom was not only an ex post small-cap or non-index phenomenon. Nine of the selected AI-related tickers were already in the S&P 500 on January 3, 2023: NVDA, AVGO, MSFT, AMZN, META, GOOG, ORCL, AMD, and ANET. Their equal-weight return was +509%, compared with +99% for SPY.

The non-member basket was even stronger at +1009%, led by PLTR at +2281% and VRT at +754%. That result shows why the AI story felt broader than the index. The most explosive returns were outside the point-in-time S&P 500 set.

Within the eligible S&P 500 member basket, NVDA alone contributed +162 percentage points to equal-weight performance. AVGO added +86 percentage points, and AMD added +79 percentage points. The boom was therefore both investable inside the benchmark and highly concentrated within a few names.

## So what?

The point-in-time result weakens one common objection to the AI trade. It was not necessary to discover every non-index winner to participate. A benchmark-aware investor could have captured large gains from companies already in the S&P 500 at the start of 2023.

The risk is concentration. A +509% equal-weight return from the member basket is exceptional, but the contribution table shows that a small number of names carried much of the result. Future AI exposure should therefore be evaluated name by name. Index membership made the trade accessible; it did not make it diversified.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
