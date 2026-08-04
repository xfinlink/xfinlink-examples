# What Is a 13F Filing? Institutional Holdings Explained

A Form 13F is a quarterly report of long positions in US-listed securities, filed with the SEC by any institutional investment manager that exercises investment discretion over $100 million or more in Section 13(f) securities. The report is due "within 45 days after the end of the calendar quarter". It states what the manager held on the final day of that quarter, and short positions are excluded by rule. Every headline about a fund building or abandoning a stake traces to this one form, and most of what those headlines imply is not in it.

## What is in a 13F filing?

One row per security. Each row carries the issuer name, a security identifier, the share count, the market value, and how voting authority splits between sole, shared and none. There is no cost basis, no purchase date, no cash balance, and no ranking.

The universe is narrower than "the portfolio". The SEC's official list of Section 13(f) securities "primarily includes U.S. exchange-traded stocks (e.g., NYSE, AMEX, NASDAQ), shares of closed-end investment companies, and shares of exchange-traded funds (ETFs)" (sec.gov Form 13F FAQ, read 4 August 2026). Mutual funds and securities traded on foreign exchanges are outside it.

Options make the arithmetic delicate. Certain equity options are reported as their own lines, flagged put or call. Of the 803 Apple positions worth at least $100 million at the 31 March 2026 quarter end, 65 were option lines rather than share lines. A large put line is a position against the stock, so a reader who sums every row for a security counts bearish exposure as though it were bullish.

Shorts do not appear anywhere. The same FAQ is blunt: "You should not include short positions on Form 13F. You also should not subtract your short position(s) in a security from your long position(s)." A fund long $500 million of a stock in one book and short more than that in another files the long leg alone, and the filing reads as conviction.

## How stale is 13F data when it becomes public?

Very. The 45-day deadline is not a target that filers beat; it is a wall they arrive at.

The 31 March 2026 quarter was due on 15 May 2026, day 45. Taking every manager that reported an Apple position worth at least $100 million for that quarter gives 750 filers, and the distribution of their filing dates looks like this.

| Days after quarter end | Share of the 750 managers that had filed |
|---|---|
| 15 | 6.5% |
| 30 | 25.3% |
| 43 | 65.7% |
| 45 (deadline) | 96.4% |
| 60 | 98.5% |

The median manager filed on day 41. A quarter of them had filed by day 30, 156 of the 750 filed on the deadline itself, and 3.6% arrived after it. The pattern belongs to the filers rather than to Apple: repeating the same measurement on Microsoft, Johnson & Johnson, Exxon Mobil, Waste Management and Ulta Beauty for the same quarter returns median lags of 42, 42, 43, 42 and 44 days.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.holdings("AAPL", quarter="2026-03-31", min_value=100_000_000)
managers = df.drop_duplicates("manager_id")
lag = (managers["filing_date"] - managers["report_date"]).dt.days

print(f"managers: {len(managers)}   median days to file: {lag.median():.0f}")
for d in (15, 30, 43, 45, 60):
    print(f"filed within {d} days: {(lag <= d).mean() * 100:5.1f}%")
```

```
managers: 750   median days to file: 41
filed within 15 days:   6.5%
filed within 30 days:  25.3%
filed within 43 days:  65.7%
filed within 45 days:  96.4%
filed within 60 days:  98.5%
```

So a position you read about in mid-May describes a portfolio as it stood at the end of March. Six weeks of trading sit between the two, and nothing in the filing tells you which side of it the manager is on now.

## What a 13F does not show

The absences matter more than anything the form contains.

Intra-quarter activity is invisible. A manager who bought a stock in January and sold it in March reports nothing, and a manager whose position is identical at both quarter ends may have traded around it the whole time. Differencing two consecutive filings gives net change, never turnover.

Shorts, cash and debt are absent, so the filing is not a portfolio and the values on it do not sum to assets under management. Confidential treatment can delay a holding's disclosure further: the SEC's FAQ states that the Commission "may prevent or delay public disclosure of Form 13F information for public interest reasons or the protection of investors".

Amendments arrive after the fact and revise what was already published. In the Apple sample above, 3.2% of the lines came from an amended filing rather than the original.

None of this makes the data useless. It makes it evidence about a date in the past, which is exactly how ownership research, crowding measures and long-horizon studies of institutional behaviour use it. Copying a filed portfolio is the one job it is poorly suited to, and any backtest of "guru" holdings inherits both the 45-day lag and the [survivorship problem](/blog/what-is-survivorship-bias-in-backtesting) of studying managers who are still filing today.

## Why the ticker is the wrong key to join on

A Form 13F never names a ticker. It names an issuer and a security identifier, and whoever hands you the data attaches the ticker afterwards from a mapping of their own. That mapping is a moving target: tickers change when companies rename or merge, and old tickers are reassigned to unrelated companies later. A holdings series joined on the ticker column silently splits one company into two, or fuses two companies into one.

The fix is to join on a permanent issuer identifier that survives ticker changes. xfinlink returns `entity_id` on every holdings row for that reason, and `xfl.resolve("TICKER")` returns every entity that has held a given ticker with the dates it held it, so a historical study can key on the company rather than on its current symbol. The same identifier links a 13F position to that issuer's prices and its [SEC filing fundamentals](/blog/sec-edgar-api-vs-fundamentals-api).

## Where can you get 13F data?

Everything below was read from each provider's own pages on 4 August 2026.

| Source | Coverage | Format | Price |
|---|---|---|---|
| SEC EDGAR filings | Every filing as submitted | One document per filer per quarter | Free |
| SEC Form 13F data sets | "July 2013 - May 2026", updated quarterly | Quarterly ZIP files of flattened XML | Free |
| WhaleWisdom Free | "Past 2 years of 13F data access" | Website tools, backtester | Free |
| WhaleWisdom Standard | "historical 13F data going back to 2001 for up to 50 funds and 50 stocks every 90 days" | Website, API, Excel add-in | $90 per quarter |
| WhaleWisdom Pro | Same, raised to "200 funds and 200 stocks every 90 days" | Website, API, Excel add-in | $150 per quarter |
| xfinlink Pro | Quarter ends 1978-12-31 to present, broad manager coverage from 1980 | Python DataFrames, REST, MCP | $29 per month |

Sources read on 4 August 2026: the SEC's [Form 13F data sets](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets) page, [whalewisdom.com/pricing](https://whalewisdom.com/pricing), and the xfinlink [pricing page](/pricing).

The SEC's own two products are the primary source and cost nothing, which is the right answer for anyone who wants a handful of filings and is willing to parse XML. The data sets are structured but arrive as quarterly archives, so assembling a decade of one manager's positions means downloading and stitching forty of them.

WhaleWisdom earns its place for a different reader: the fund-watching workflow, with backtesting and scoring already built, sits on its site rather than in your code.

Work that lives in Python is a different shape of problem. `xfl.holdings("AAPL", quarter="2026-03-31")` returns who held Apple that quarter as a DataFrame, `xfl.managers("berkshire")` finds a filer by name, and `xfl.manager_holdings(manager_id, quarter=...)` returns that firm's reported portfolio, each row already carrying `entity_id`, `filing_date`, `put_call` and `is_amendment`. Institutional holdings sit on paid plans from $29 a month; the [docs](/docs) list every field and the [pricing page](/pricing) lists the tiers.

## FAQ

**How often is 13F data updated?**
Four times a year. Each calendar quarter is reported within 45 days of its end, so a quarter is largely complete about six weeks after it closes and keeps filling in from late filers afterwards.

**Can you see a hedge fund's short positions in a 13F?**
No. The SEC instructs filers not to include short positions and not to net them against long positions, so a 13F shows the long side of US-listed equity exposure only.

**How far back does 13F data go?**
The SEC's structured Form 13F data sets start with filings from July 2013 as of August 2026. xfinlink's holdings series runs from the quarter ending 31 December 1978 to the present, with broad manager coverage from 1980 onward.

**Does a 13F tell you what a manager bought or sold?**
Only by comparing consecutive quarters, and only as a net change. Positions opened and closed inside a single quarter never appear.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
