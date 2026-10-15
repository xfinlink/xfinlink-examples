# Survivorship Bias Examples in Investing: 7 Real Cases

Survivorship bias in investing is the error of judging funds, stocks or strategies by the cases still standing, after the failures have dropped out of the record. Seven examples recur: mutual fund databases that lose closed funds, fund families that advertise only their winners, hedge fund databases whose failing managers stop reporting, stock indexes read from today's member list, backtests built on that list, "best stocks" lists, and famous winners told without the companies that died beside them. Each makes past returns look better than investors received. The stock-index case is the easiest to measure: of the 500 companies in the S&P 500 on 3 January 2000, 312 are no longer in the index.

## What Is Survivorship Bias in Finance?

A sample suffers from survivorship bias when getting into it depends on the outcome being studied. In finance the outcome is usually performance and the filter is silent: a closed fund leaves the fund list, a bankrupt company leaves the stock list, a losing manager stops reporting. What remains leans towards the ones that did well, so averages are too high and measured risk too low.

The seven cases, what each one loses, and the correction:

| Example | What drops out | Effect on the numbers | Correction |
|---|---|---|---|
| Mutual fund databases | Merged and liquidated funds | Category averages too high | Count every fund alive at the start |
| Fund-family records | Funds closed or never launched | Advertised records look skilful | Judge the family, not its best fund |
| Hedge fund databases | Managers who stop reporting | Index returns too high | Keep dead funds; drop backfilled months |
| Stock indexes | Removed members | History reads as a list of winners | Use the roster on the past date |
| Backtests | Stocks that left before today | Returns nobody could earn | Pick the universe as of each rebalance |
| "Best stocks" lists | Delisted tickers | Winners look typical | Rank the full starting population |
| Famous winners | The failed peers | Odds of picking a winner look good | Name the cohort before the outcome |

## 1. Mutual Fund Databases That Lose Closed Funds

Fund companies merge weak funds into stronger siblings or liquidate them, and a database of funds open today has lost them. Elton, Gruber and Blake (*Review of Financial Studies*, 1996) followed every fund alive at the end of 1976 for this reason: funds that disappear tend to do so because of poor performance.

The attrition is large. Morningstar's Active/Passive Barometer for year-end 2025 started its 20-year window with 485 active US large-blend funds; 36.5% were still alive at the end. Of the active large-growth funds that existed two decades earlier, nearly 66% had closed, and just 1% outperformed their average indexed peer.

Carhart, Carpenter, Lynch and Musto (*Review of Financial Studies*, 2002) measured what dropping dead funds does to the average: the bias rises with sample length, from 0.07% a year for one-year samples to 1% a year beyond 15 years.

## 2. Fund-Family Track Records and Incubated Funds

A fund family can show a page of strong ten-year records without any skilled manager, provided the weaker funds were closed along the way; the advertisement is accurate fund by fund and misleading as a whole.

A sharper version happens before a fund is sold. Some families start several funds privately and open only the ones that did well. Richard Evans (*Journal of Finance*, 2010) found that funds in incubation outperformed non-incubated funds by 3.5% risk-adjusted, an edge that disappeared once they opened to the public. Filtering on fund age and on the ticker's creation date removes this bias; a fund size filter does not.

## 3. Hedge Fund Databases and Indexes

Hedge funds are private, and reporting to a commercial database is a choice. Fung and Hsieh (*Financial Analysts Journal*, 2009) trace the return biases to two events: a fund electing to enter a database, bringing its earlier history (backfill bias), and a fund leaving one (survivorship bias).

Bing Liang (*Journal of Financial and Quantitative Analysis*, 2000) put the survivorship bias in hedge fund returns at more than 2% a year, found poor performance to be the main reason funds disappear, and showed that two major databases held different numbers of dissolved funds. An index built from the funds currently reporting inherits all of it.

## 4. The S&P 500 Today Versus the S&P 500 of 2000

An index's member list is itself a survivor list: a company stays in the S&P 500 while it is large and listed, and leaves when it is acquired, shrinks or fails. The fix is the roster as it stood on the date that matters, which the [historical S&P 500 constituents lookup](/historical-sp500-constituents) shows for any date back to 1925, companies later removed included. The same roster is available in Python:

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

then = xfl.index("sp500", as_of="2000-01-03")  # the roster on that day
now = xfl.index("sp500")                       # the roster today

gone = then[~then["entity_id"].isin(now["entity_id"])]
print(len(then), len(now), len(gone))

exits = ["ENE", "WCOM", "LEH", "CPQ", "YHOO"]
print(gone[gone["ticker"].isin(exits)][["ticker", "entity_name", "removed_date"]].to_string(index=False))

late = now[now["ticker"].isin(["NVDA", "AMZN", "GOOGL", "META", "TSLA"])]
print(late[["ticker", "entity_name", "added_date"]].to_string(index=False))
```

```
500 503 312
ticker                  entity_name removed_date
   CPQ         COMPAQ COMPUTER CORP   2002-05-03
   ENE                   ENRON CORP   2001-11-29
   LEH LEHMAN BROTHERS HOLDINGS INC   2008-09-16
  WCOM          WORLDCOM INC GA NEW   2002-05-14
  YHOO                    YAHOO INC   2017-06-16
ticker        entity_name added_date
 GOOGL       ALPHABET INC 2006-04-03
  AMZN     AMAZON COM INC 2005-11-21
  META META PLATFORMS INC 2013-12-23
  NVDA        NVIDIA CORP 2001-11-30
  TSLA          TESLA INC 2020-12-21
```

The comparison runs on `entity_id`, which stays with a company through ticker changes, so a rename is not an exit. Of the 500 members on the first trading day of 2000, 312 (62%) are not in today's roster of 503 rows. The departures came early: 54 left during 2000 and 100 by the end of 2002. Seven of them left, later returned and are members today, AMD and PG&E among them, so 319 of the 500 left at least once while 312 are absent now.

Leaving is not the same as failing, and the five named exits show the range. Enron's last day in the index was 29 November 2001; it filed for Chapter 11 on 2 December 2001. WorldCom's last day was 14 May 2002, and it filed on 21 July 2002, according to its board's investigative report. Lehman Brothers filed on 15 September 2008 (its 8-K) and was out of the index after the following day. Compaq was bought: HP completed the merger on 3 May 2002, Compaq's last day in the index. Yahoo sold its operating business to Verizon on 13 June 2017 and renamed itself Altaba on 16 June (SEC order of 24 April 2018), its last day as a member. A survivor-only history loses bankruptcies and takeovers alike.

## 5. Backtests Built on Today's Constituents

A backtest that runs today's index members over the past twenty years has chosen its stocks with knowledge of the future. Started in January 2000, it would be missing 312 of its 500 proper starting positions, Enron and WorldCom among them. Price files that carry only live tickers do the same damage later, since a delisted symbol returns nothing and the loser leaves without a trace. How the bias enters through the universe and the price source is set out in [the explainer on survivorship bias in backtesting](/blog/what-is-survivorship-bias-in-backtesting).

## 6. "Best Stocks" Lists and Screens

A list of the best-performing stocks of the last twenty years usually ranks the stocks that trade today. It cannot include a company that went bankrupt in 2002, so it shows what winners look like and nothing about how often a similar stock became one. A screener whose history covers only today's tickers has the same weakness.

Of today's 503 S&P 500 roster rows, 188 were already members on 3 January 2000. A list of "S&P 500 stocks that compounded since 2000" drawn from today's index starts from those 188 and has discarded the 312 that left. Rank the starting population instead, each stock with its actual outcome.

## 7. Famous Winners and the Dot-Com Companies Beside Them

The usual story about technology investing is told through Amazon, Nvidia and Alphabet. None was in the S&P 500 at the start of 2000, when Apple and Microsoft already were: according to today's roster, Nvidia joined on 30 November 2001, Amazon on 21 November 2005 and Alphabet on 3 April 2006, with Meta Platforms following in December 2013 and Tesla in December 2020. The technology names the index did hold in January 2000 tell a different story: 3Com left in July 2000, Global Crossing in October 2001, Lucent Technologies in November 2006 and Sun Microsystems in January 2010, and Yahoo, a member only since 8 December 1999, left in 2017.

Hindsight picks the winners out of a crowd and forgets the crowd; the fair question needs the list as it stood before anyone knew the outcome.

## How to Avoid Survivorship Bias in Your Data

The seven cases share one correction: define the population on the start date and follow every member, including those that disappear.

- Start from a list dated to the beginning of the period: every fund open then, every index member then.
- Keep the exits and record how each ended (takeover, bankruptcy, demotion, fund merger), because that last return belongs in the result.
- Key the data on a permanent identifier, not a ticker; a recycled ticker can attach another company's history to a past member.
- For funds, use a database that keeps dead funds, and distrust history added when a fund joined it.

For stock indexes, xfinlink returns the S&P 500, Nasdaq-100 and Dow Jones Industrial Average membership on any past date the record covers through `index(as_of=...)`, members later removed, acquired or delisted included, on every plan including Free. The [index membership reference](/docs/index-membership) documents the call, and [the guide to S&P 500 additions and deletions history](/blog/sp500-additions-and-deletions-history) covers the dated log of every change behind the rosters.

## FAQ

**Does a company leaving the S&P 500 mean it failed?** No. Many are acquired, as Compaq was by HP in 2002; others have shrunk. A survivor-only dataset loses all of them.

**Is survivorship bias the same as look-ahead bias?** No. Survivorship bias is about who is in the sample; look-ahead bias is about using information before it was public. A universe chosen from today's index commits both.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
