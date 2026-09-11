# How to Run an Event Study in Python

An event study measures what a dated event did to a share price by comparing the return a stock actually produced against the return a benchmark model says it should have produced over the same sessions. Four inputs make it work: a list of events with exact dates, daily prices covering an estimation window before each event and a short window around it, a benchmark series on the same trading calendar, and an identifier that ties each event to the right company. The regression is four lines of Python. The event list is where the effort goes, and where most published results break.

## What Data Does an Event Study Need?

Each input has a failure mode, and none of them raise an error.

| Input | Requirement | How it fails |
| --- | --- | --- |
| Event list with dates | One event definition, one date convention | Announcement dates mixed with effective dates |
| Estimation-window returns | 100 to 250 sessions, ending before the event | Window absorbs the run-up and inflates the expected return |
| Event-window returns | Typically 5 sessions before to 20 after | Missing sessions silently shorten the window |
| Benchmark returns | Same calendar, same adjustment basis | Benchmark holds the event stock at a large weight |

The estimation window is what makes the data bill. Twenty-six sessions around the event is nothing; 120 sessions before it, for every event in the sample, is the requirement that decides which provider can supply the study.

## Where Do Event Dates Come From?

Index membership changes, insider transactions filed on Form 4, earnings releases and merger announcements all carry usable dates. Index changes are the cleanest starting point because the date is administrative rather than reported, and because the population is finite and known.

One distinction governs the whole design: the announcement date is when the market learns, and the effective date is when the index actually changes. The gap between them is weeks, not hours. S&P Dow Jones Indices dated its March 2025 release 7 March 2025 and stated that DoorDash, TKO Group Holdings, Williams-Sonoma and Expand Energy would join the S&P 500 "effective prior to the open of trading on Monday, March 24" (press.spglobal.com, as of September 2026). A window centred on 24 March therefore has the news sitting 11 sessions to its left. Either convention works as long as the analysis states which one it uses and the window is wide enough to hold the news.

For the benchmark, the Kenneth French data library publishes a Fama/French 3 Factors [Daily] file free of charge, current through July 2026 as of September 2026, and the market excess return in it is a defensible proxy for anyone who prefers a factor model to an ETF. An index ETF pulled from the same source as the stock prices has the advantage of sharing a calendar and an adjustment basis with them, which removes one alignment step from the code.

For the events themselves, `index_events("sp500", event_type="added")` returns one row per addition with an effective date, a company name and a stable `entity_id`, covering S&P 500 changes from 1957, Nasdaq 100 from 1995 and Russell 2000 from 1979. The identifier matters more than it looks: a ticker is a label that gets reassigned, and a study running back through several years of index changes will meet symbols that changed hands. Field definitions are in the [docs](https://xfinlink.com/docs).

## How Long Should the Windows Be?

The estimation window ends before the market can anticipate the event. This study uses sessions -120 to -21 relative to the effective date, leaving a 20-session gap so that pre-event drift does not contaminate the model of normal returns. The event window runs -5 to +20, which spans the sessions when index funds trade into the name and the month that follows.

That choice fixes what the study can answer. With the announcement roughly 11 sessions before a scheduled effective date, a -5 window sits entirely after the news, so the numbers below measure the inclusion trade and its aftermath rather than the market's reaction to the announcement. Measuring the announcement reaction is the same code with announcement dates and a window that reaches further left.

Short histories cut the sample. A company that joins an index on its first day as a separate listing has no pre-event price history, so no estimation window can be built for it. Ten of the 51 S&P 500 additions between 2023 and 2025 drop out on that test: nine began trading at or after the effective date, and one had listed too recently to fill even a 60-session minimum. The constraint belongs to the events rather than to any data source, since those companies did not exist as separate listings before the date being studied.

## What Does the Code Look Like?

Fetch the events, fetch a benchmark once, then loop: estimate a market model on the clean window, apply it to the event window, and keep the abnormal returns.

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ev = xfl.index_events("sp500", event_type="added", start="2023-01-01", end="2025-12-31")
mkt = xfl.prices("SPY", start="2022-01-01", end="2026-03-31", fields=["close"])
mkt = mkt[["date", "close"]].rename(columns={"close": "mkt"}).set_index("date")
mkt["rm"] = mkt["mkt"].pct_change()

rows = []
for _, e in ev.iterrows():
    d = pd.Timestamp(e["effective_date"])
    px = xfl.prices(
        entity_id=int(e["entity_id"]),
        start=(d - pd.Timedelta(days=260)).strftime("%Y-%m-%d"),
        end=(d + pd.Timedelta(days=50)).strftime("%Y-%m-%d"),
        fields=["close"],
    )
    if px.empty:
        continue
    df = px[["date", "close"]].set_index("date").join(mkt[["rm"]], how="inner")
    df["ri"] = df["close"].pct_change()
    df = df.dropna()

    pos = df.index.searchsorted(d)
    est = df.iloc[max(0, pos - 120):pos - 20]
    win = df.iloc[pos - 5:pos + 21]
    if len(est) < 60 or len(win) < 26:
        continue

    beta, alpha = np.polyfit(est["rm"], est["ri"], 1)
    ar = win["ri"] - (alpha + beta * win["rm"])
    rows.append({
        "ticker": e["ticker"], "date": d.date(), "beta": beta,
        "car_incl": ar.iloc[0:7].sum() * 100,
        "ar_event": ar.iloc[5] * 100,
        "car_post": ar.iloc[7:26].sum() * 100,
    })

r = pd.DataFrame(rows).round(2)
print(f"events={len(ev)}  usable={len(r)}  no usable estimation window={len(ev) - len(r)}")
print()
print(r.head(6).to_string(index=False))
print()

labels = {"car_incl": "CAR(-5,+1)", "ar_event": "AR(0)", "car_post": "CAR(+2,+20)"}
for col, lab in labels.items():
    x = r[col]
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{lab:<12} mean={x.mean():+6.2f}%  median={x.median():+6.2f}%  "
          f"share>0={(x > 0).mean() * 100:4.1f}%  t={t:+5.2f}")
```

Output from the run above:

```
events=51  usable=41  no usable estimation window=10

ticker       date  beta  car_incl  ar_event  car_post
    BG 2023-03-15  0.62      0.75     -7.15     -7.00
  PODD 2023-03-15  1.08      8.76     -1.00     -5.20
  FICO 2023-03-20  1.68     -7.61      2.25    -12.76
  AXON 2023-05-04  0.70     -2.28     -0.69    -21.04
  PANW 2023-06-20  0.85      5.85     -1.78     -4.52
  ABNB 2023-09-18  1.69     -1.36     -0.19     -8.60

CAR(-5,+1)   mean= +1.56%  median= +0.75%  share>0=56.1%  t=+1.18
AR(0)        mean= -0.76%  median= -0.83%  share>0=29.3%  t=-1.70
CAR(+2,+20)  mean= -6.33%  median= -5.12%  share>0=34.1%  t=-3.37
```

Forty-one events carry an average abnormal return of 1.56% across the inclusion window, which a t-statistic of 1.18 does not separate from zero, and an average of -6.33% over the following 19 sessions, which a t-statistic of -3.37 does. Read that second number carefully before treating it as a strategy: 41 events is a small sample, the dispersion runs from -43% to +13%, and additions between 2023 and 2025 skewed toward high-beta names that had already run a long way.

## What Makes an Event Study Wrong?

Survivorship in the event list is the first problem, and it is invisible in the output. An event list assembled from companies that are still listed today drops every addition that was later acquired or removed, and those are systematically the disappointing ones. The same mechanism that distorts backtests is at work here, described in [what is survivorship bias in backtesting](/blog/what-is-survivorship-bias-in-backtesting).

Joining on tickers is the second. Symbols get reassigned to unrelated companies, so a five-year study that keys on a ticker will occasionally attach the wrong estimation window to an event. Keying on an entity identifier removes the question, and dropping rows with a null identifier before the merge keeps the join honest.

Confounding events are the third, and they are unavoidable rather than fixable. A 26-session window covers roughly 40% of a quarter, so something close to two in five windows contain an earnings release. Report the median alongside the mean: a single 10% earnings collapse shifts the average of 41 events by a quarter of a percentage point, and the median barely notices it.

Date discipline is the fourth. Using information that was not public on the date it is attached to inverts the result rather than weakening it, which is the failure covered in [what is look-ahead bias in backtesting](/blog/what-is-look-ahead-bias-in-backtesting).

## FAQ

**How many events do I need?** Daily abnormal returns have a standard deviation of roughly 2% for a typical large cap, so a sample of 40 can resolve an effect of about 1% and no smaller. Effects under half a percent need several hundred events.

**Can I do this on free data?** Partly. Alpha Vantage's standard limit is 25 API requests per day as of September 2026, and the study above makes 53 requests, so a free key there spreads one run across three days. The yfinance library will hand back the price series for nothing, and for a handful of events that is a fair trade; its own README states the project is "intended for research and educational purposes" and that "the Yahoo! finance API is intended for personal use only", which settles the question for anything published or commercial. An xfinlink free key covers a rolling twelve months at one ticker per call, which is enough to build and test the loop before paying for the history it needs; the plans are on the [pricing](https://xfinlink.com/pricing) page.

**What is the minimum field list?** A date, a close and an identifier, for the stock and the benchmark. Everything else in an event study is arithmetic on those columns, which is why the choice of provider comes down to how far back the daily prices go and whether the event dates arrive with them instead of from a second source that has to be joined by hand. The full field reference is in the [docs](https://xfinlink.com/docs), and the fields a backtest needs on top of these are set out in [data requirements for backtesting](/blog/data-requirements-for-backtesting).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
