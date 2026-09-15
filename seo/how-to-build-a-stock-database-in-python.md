# How to Build a Stock Database in Python

A local stock database is two tables and a refresh policy. Daily bars keyed on a permanent company id and a date, financial statements keyed on that same id and a fiscal period, both loaded through batched API calls and both updated by re-fetching an overlapping window instead of appending to the end. The initial load is the easy half: twenty companies and five years of daily bars came to 25,059 rows in a 1.77 MB SQLite file, written by a loader short enough to read in one screen. What goes wrong later is the assumption that a row, once stored, stays correct.

## Do You Need a Local Copy at All?

Storing rows locally pays off in two situations. The first is repetition: a dashboard or a nightly model that reads the same five years of bars on every run should read them from disk. The second is the join. Market data has to sit next to data nobody else holds, such as positions, signal values, or hand-labelled events, and doing that in SQL beats doing it in a notebook each morning.

A script that needs whatever is current, once, does not need a database. Neither does exploratory work, where a stale local copy costs more time than it saves. The separate question of whether to pull flat files or call an API is [answered in its own guide](/blog/bulk-stock-data-download-vs-api); what follows assumes the decision to keep a copy has already been made.

## What Tables Does a Stock Database Need?

Three, and the third is small.

A **prices** table at one row per company per session, holding the raw close, the adjusted close, and volume. A **fundamentals** table at one row per company per fiscal period, holding the statement fields the work actually reads, plus the period end and the filing date. A **securities** table with one row per company, carrying the current ticker, the name, and the sector, so the price table does not repeat a name string twenty-five thousand times.

Resist the fourth table. Computed ratios do not belong on disk. They are cheap to recalculate, and they go stale without saying so whenever an input is revised, which turns every stored ratio into something that has to be invalidated by hand. Store inputs, derive outputs.

## What Should the Primary Key Be?

Not the ticker. Symbols get reassigned to unrelated companies after a delisting, and they change hands quietly, without an error or a gap to notice. A table keyed on the ticker string will eventually hold two companies under one key and present the result as a single price history.

Key on a permanent company identifier and a date. Every xfinlink frame carries `entity_id` for exactly this, so the key survives a ticker change and the same id joins prices to statements. The choice between symbol, CIK, and other identifier schemes is worked through in [Ticker vs CIK vs FIGI](/blog/ticker-vs-cik-vs-figi). Keep the ticker as an ordinary column, never as an identity.

## How Do You Load It the First Time?

One call per table, batched by the client, then an upsert. The `INSERT OR REPLACE` matters as much as the fetch: it makes the loader safe to run twice.

```python
import sqlite3
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

COLS = ["entity_id", "date", "ticker", "close", "adj_close", "volume"]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG",
           "KO", "CVX", "WMT", "UNH", "HD", "CAT", "T", "DIS", "GS", "NKE"]

con = sqlite3.connect("stocks.db")
con.execute("""
CREATE TABLE IF NOT EXISTS prices (
    entity_id INTEGER, date TEXT, ticker TEXT,
    close REAL, adj_close REAL, volume INTEGER,
    PRIMARY KEY (entity_id, date))
""")

df = xfl.prices(TICKERS, period="5y", fields=["close", "adj_close", "volume"])
rows = df[COLS].copy()
rows["date"] = rows["date"].astype(str).str[:10]
con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?)",
                rows.itertuples(index=False, name=None))
con.commit()
```

Full script, including the incremental refresh: [how-to-build-a-stock-database-in-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/seo/how-to-build-a-stock-database-in-python.py)

```
stored 25,059 rows  2021-09-16 to 2026-09-14  1,773,568 bytes
refresh from 2026-09-04: 120 rows fetched, 0 new, 120 overwritten
```

The shape of the source API decides how much loader code exists. Where one request returns many companies, the backfill is a loop over batches and a budget question. Where one request returns one company, the backfill becomes a scheduler, and the free tiers get tight: Alpha Vantage's daily time series endpoint takes one `symbol` per call, its free keys are held to "25 API requests per day", and on that endpoint the full-length series is "available to premium keys", with the free `outputsize=compact` returning "only the latest 100 data points" (alphavantage.co/premium and alphavantage.co/documentation, checked 15 September 2026). At twenty-five calls a day and one symbol a call, backfilling five hundred symbols on a free key runs to twenty days, and the deep history that backfill needs sits behind a paid plan anyway.

For comparison, the xfinlink plan limits are per request rather than per symbol. The [pricing page](https://xfinlink.com/pricing) sets Pro at 100 tickers a request and 10,000 requests a day; the [docs page](https://xfinlink.com/docs) puts a free key at 1 ticker and the most recent twelve months. A twenty-name, five-year backfill is a single call and 25,059 rows, which sits inside one hour of the Pro row budget with room to spare.

Statement filings can also be picked up in bulk directly from the regulator, free of charge. The SEC states that "The most efficient means to fetch large amounts of API data is the bulk archive ZIP files, which are recompiled nightly", and that the ZIP behind the submissions API "is updated and republished nightly at approximately 3:00 a.m. ET" (sec.gov, EDGAR application programming interfaces, checked 15 September 2026). That is a genuine route into a local store, and it delivers filings rather than a joined panel: identity, share counts, and price alignment remain the reader's problem.

## Why Append-Only Breaks the Table

The adjusted price series is not a record of what happened. It is a calculation over everything that has happened since, and every split rewrites it backwards.

```
      date ticker      close  adj_close
2024-06-05   NVDA 1224.40002 122.440002
2024-06-06   NVDA 1209.97998 120.997998
2024-06-07   NVDA 1208.88000 120.888000
2024-06-10   NVDA  121.79000 121.790000
2024-06-11   NVDA  120.91000 120.910000
2024-06-12   NVDA  125.20000 125.200000
```

NVIDIA split ten for one on 10 June 2024. The raw close for 7 June stands at 1,208.88, as traded; the adjusted value for that same session now reads 120.888. A row written before the split would have carried 1,208.88 in both columns, and that stored adjusted value is now wrong by a factor of ten, in the column most models read. Nothing in the row says so. An append-only loader, which asks only for dates after the last one it stored, will never touch it again. The mechanics of the rewrite are covered in [Split Adjustment Explained](/blog/split-adjustment-explained).

Two habits fix it. Re-fetch an overlapping window on every run and replace what comes back, which also catches late corrections at the end of the series. Then, on a slower cycle, re-fetch full history for any company whose stored adjusted close disagrees with a freshly pulled value on an old date, and replace the whole series rather than patching rows.

The refresh in the run above pulled 120 rows from a ten-day overlap, added nothing, and overwrote 120. Doing no visible work is the point: the loader is idempotent, so a cron job that fires twice, or a backfill that gets rerun after a crash, leaves the table in the same state.

Financial statements move for a different reason. Figures get restated, and a quarter can arrive weeks after the period it describes closed. Refreshing only the newest quarter preserves whatever the previous vintage wrote; re-fetching a trailing year or two of periods and replacing them keeps the table matching the current filings.

## What to Refresh, and How Often

| Table | Grain | Key | Cadence | Mode |
|---|---|---|---|---|
| prices | One session | entity_id + date | Daily | Re-fetch a 5 to 10 session overlap, upsert |
| prices, adjusted column | One session | entity_id + date | Monthly, or after a corporate action | Full re-fetch for the affected company, replace the series |
| fundamentals | One fiscal period | entity_id + period_end | Weekly | Re-fetch the trailing 8 quarters, replace |
| securities | One company | entity_id | Monthly | Replace the table |
| Derived ratios | — | — | Never | Recompute on read |

## FAQ

**SQLite or Postgres?** SQLite handles a single-writer store of this size without complaint; the file above holds five years for twenty companies in 1.77 MB. Move to Postgres once several services write at once, or once other people need to query the store from their own machines.

**How much disk does a large universe need?** The measured store works out at roughly 71 bytes a row with six columns and a primary-key index. Five hundred companies across twenty years is about 2.5 million rows, so on the order of 180 MB. Storage is rarely the constraint; refresh discipline is.

**Should the raw close be stored as well as the adjusted one?** Yes. The raw close never changes, so it is the column that lets a stale adjusted series be detected and rebuilt. Keeping only the adjusted value leaves nothing to check it against.

**Can the stored data be redistributed?** Storing under a subscription and showing values to your own users are separate rights at most vendors. The xfinlink tiers are set out on the [pricing page](https://xfinlink.com/pricing), where internal use and end-user delivery are priced as different plans rather than negotiated.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
