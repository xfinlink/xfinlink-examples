# Full write-up: https://xfinlink.com/blog/how-to-build-a-stock-database-in-python
#
# Builds a local SQLite price store keyed on (entity_id, date), then refreshes it
# with an overlapping window so re-running the loader is idempotent.

import os
import sqlite3

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

DB = "stocks.db"
COLS = ["entity_id", "date", "ticker", "close", "adj_close", "volume"]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG",
           "KO", "CVX", "WMT", "UNH", "HD", "CAT", "T", "DIS", "GS", "NKE"]

con = sqlite3.connect(DB)
con.execute("""
CREATE TABLE IF NOT EXISTS prices (
    entity_id INTEGER,
    date      TEXT,
    ticker    TEXT,
    close     REAL,
    adj_close REAL,
    volume    INTEGER,
    PRIMARY KEY (entity_id, date))
""")


def upsert(df):
    rows = df[COLS].copy()
    rows["date"] = rows["date"].astype(str).str[:10]
    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?)",
                    rows.itertuples(index=False, name=None))
    con.commit()
    return len(rows)


# 1. Backfill: one call covers every ticker; the client batches by plan limit.
backfill = xfl.prices(TICKERS, period="5y", fields=["close", "adj_close", "volume"])
upsert(backfill)
count, first, last = con.execute(
    "SELECT COUNT(*), MIN(date), MAX(date) FROM prices").fetchone()
print(f"stored {count:,} rows  {first} to {last}  {os.path.getsize(DB):,} bytes")

# 2. Incremental refresh: re-fetch an overlapping window and replace, never append.
start = (pd.Timestamp(last) - pd.Timedelta(days=10)).date().isoformat()
before = count
fetched = upsert(xfl.prices(TICKERS, start=start,
                            fields=["close", "adj_close", "volume"]))
after = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
print(f"refresh from {start}: {fetched} rows fetched, "
      f"{after - before} new, {fetched - (after - before)} overwritten")

# 3. Why the overlap matters: the adjusted series is rewritten by every split.
nvda = xfl.prices("NVDA", start="2024-06-05", end="2024-06-12",
                  fields=["close", "adj_close"])
print(nvda[["date", "ticker", "close", "adj_close"]].to_string(index=False))

con.close()
