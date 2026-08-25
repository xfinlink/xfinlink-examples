# Broker API vs Data API for Historical Stock Data

A broker's API is built to place orders and report account state, and its market data service exists to support that. A data API is built to hand back history. The documented limits say so plainly: Alpaca's market data plans list historical stock data going back to 2016 and cap the free plan at 200 API calls per minute, and Interactive Brokers paces historical requests at no more than 60 in any ten-minute period while requiring a funded account with market data subscriptions (both as of August 2026). Neither limit troubles a live strategy pulling this morning's bars. Both bind hard on a backtest that wants twenty years of daily prices across several hundred names, which is the work most people are actually doing when they reach for a broker's API and find it fighting back.

## What does a broker's API give you?

Alpaca sets the Basic plan as the default for paper and live accounts at no cost. For equities that plan carries real-time data from the IEX exchange only, restricts access to the latest 15 minutes, allows 200 API calls per minute, and limits websocket subscriptions to 30 symbols. Historical stock data on both the Basic and the Algo Trader Plus plan is listed as available since 2016, and Algo Trader Plus is priced at $99 per month (verified against the Alpaca documentation and pricing page, August 2026). The free tier is genuinely usable, which is a real point in its favour: a paper-trading bot on recent data costs nothing to run.

Interactive Brokers works differently. Its TWS API is described in the IBKR documentation as an interface to Trader Workstation or IB Gateway that requires network connectivity to a running instance of one of those programs, so the API is not a plain HTTP endpoint but a socket into desktop software the user keeps alive. Receiving market data through it requires trading permissions for the instruments, a funded account (except for forex and bonds), and market data subscriptions for the username (as of August 2026). Historical requests are then paced: identical requests inside 15 seconds are rejected, six or more requests for the same contract, exchange and tick type inside two seconds are rejected, and more than 60 requests in any ten-minute period is over the line, with bid-ask requests counted twice (as of August 2026).

Sixty requests per ten minutes is six per minute. Pulling one ticker's history per request, an S&P 500 universe takes about an hour and a half of continuous, carefully throttled requesting, assuming nothing drops.

## Why are the limits shaped that way?

Exchange licensing explains most of it. A broker redistributes quotes under agreements that meter who receives what, so entitlement checks and subscriptions sit in front of the data. Pacing explains the rest: the same gateway process that answers a historical bar request is the one carrying orders to the market, and a research script hammering it is competing with execution. Both constraints are reasonable engineering for a trading system. They are simply not designed around the shape of a research pull, which is one large, cold, infrequent request for a lot of history at once.

## How far back does the history reach?

A 2016 floor removes the 2008 crisis, the 2011 drawdown, the taper tantrum and the entire prior rate cycle from anything tested against it. That is not a detail. A momentum or value signal measured only across 2016 to 2026 has seen one broad regime and one short crash, so the estimate it produces carries far less information than the number of observations suggests. Depth of history is the first thing to check before anything else about a source, as covered in our notes on [data requirements for backtesting](https://xfinlink.com/blog/data-requirements-for-backtesting) and [survivorship bias](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting).

xfinlink serves end-of-day US equity and ETF data over plain HTTP with a Python client, and its paid plans carry daily prices back to 1996 and financial statements back to 1950; the free tier covers a rolling one-year window. One call reaches the window a 2016 floor cannot express:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["KO", "JNJ", "PG", "MCD"]

px = xfl.prices(tickers, start="1996-01-02", end="2015-12-31", fields=["adj_close"])
fn = xfl.fundamentals(tickers, period_type="annual",
                      start="1996-01-01", end="2015-12-31", fields=["revenue"])

for t in tickers:
    bars = px[px["ticker"] == t]
    stmts = fn[fn["ticker"] == t]
    print(f"{t}: {len(bars):,} daily bars {bars['date'].min().date()} to "
          f"{bars['date'].max().date()} | {len(stmts)} annual statements")
```

```
KO: 5,036 daily bars 1996-01-02 to 2015-12-31 | 20 annual statements
JNJ: 5,036 daily bars 1996-01-02 to 2015-12-31 | 20 annual statements
PG: 5,036 daily bars 1996-01-02 to 2015-12-31 | 20 annual statements
MCD: 5,036 daily bars 1996-01-02 to 2015-12-31 | 20 annual statements
```

Two calls, four tickers, no desktop process and no funded account. The per-call ticker cap runs 1 on the free tier, 100 on Pro, 500 on Max and 5,000 on Redistribution, and the daily request allowance is 100 on the free tier (with a 40 per hour burst cap) against 10,000 on Pro. Details sit on the [pricing page](https://xfinlink.com/pricing).

## What about company financials?

Alpaca's documentation index for US stock market data lists historical and latest bars, quotes and trades, along with snapshots, condition codes and exchange codes (as of August 2026). Interactive Brokers documents fundamental data as the Wall Street Horizon event calendar, covering earnings dates, dividend dates, options expiration, splits, spinoffs and investor conferences, and states that the Wall Street Horizon Enchilada Pro research subscription must be activated in Account Management first (as of August 2026). A corporate event calendar is a useful thing to have, and for event-driven work it is the right shape of data.

It is not a set of financial statements. Any screen ranking on margin, leverage or cash conversion needs the statements themselves, which is the other half of what a data API carries: xfinlink returns income statement, balance sheet and cash flow fields built from SEC EDGAR public filings, plus computed metrics, index membership, insider transactions and institutional holdings from the same key. The [docs](https://xfinlink.com/docs) list the field set.

## How do the three compare?

| | Alpaca (Basic) | Interactive Brokers TWS API | xfinlink |
|---|---|---|---|
| To start | Alpaca account, API key | Funded account, market data subscriptions, TWS or IB Gateway running | API key, free at signup |
| Historical stock data | Since 2016 | Paced per request | Daily prices from 1996 on paid plans; rolling year free |
| Documented request limit | 200 API calls/min | 60 historical requests per 10 minutes | 100/day free, 10,000/day Pro |
| Real-time quotes | IEX only; all US exchanges at $99/mo | Yes, with subscriptions | End-of-day |
| Financial statements | Bars, quotes, trades, snapshots documented | Wall Street Horizon event calendar (separate subscription) | Income, balance sheet, cash flow |

All competitor figures verified against each vendor's own documentation, August 2026.

## When is the broker's API the right answer?

Whenever the data has to match the venue where the order lands. Live quotes, the fill you actually received, current positions and buying power come from the broker and nowhere else, and a paper-trading loop on Alpaca's free plan is a cheap way to prove a strategy runs before it trades. Anyone reconciling executions to their own records needs the broker's record, not a third party's.

The split that works is a plain one. The broker's API drives the live loop; a data API supplies the history the strategy was built and tested on. Keeping them separate means the research pull never competes with order flow for the same connection, and the backtest is not silently truncated to whatever depth the trading connection happened to offer.

## FAQ

**Can I backtest on data from my broker's API?**
For a recent window, yes. Alpaca lists historical stock data since 2016 and Interactive Brokers serves history subject to its pacing rules, so a short backtest is workable. A multi-decade test across a wide universe runs into the depth floor or the request pacing well before it finishes.

**Do I need a funded brokerage account to get historical prices?**
With Interactive Brokers, the documentation names a funded account and market data subscriptions among the requirements for API market data. Alpaca's Basic plan is available on paper accounts. A data API needs neither, only a key.

**Which prices should a backtest use?**
End-of-day bars with a split adjustment are the standard basis for daily research. Reserve the broker's tick and quote data for execution modelling, where the venue and the timestamp genuinely matter.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
