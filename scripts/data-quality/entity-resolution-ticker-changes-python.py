# Full write-up: https://xfinlink.com/blog/entity-resolution-ticker-changes-python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# ── Step 1: Resolve entity history for each ticker ──────────────────

tickers = ["META", "FB", "GM", "DELL"]
for t in tickers:
    info = xfl.resolve(t)
    entities = info["data"][t]["entities"]
    print(f"\n{t} — {len(entities)} entity(ies):")
    for e in entities:
        end = e["ticker_valid_to"] or "present"
        print(f"  {e['name']} (entity {e['entity_id']}): "
              f"{e['ticker_valid_from']} to {end}")

# ── Step 2: Show that META and FB share entity_id 2 ─────────────────

meta_entities = xfl.resolve("META")["data"]["META"]["entities"]
fb_entities = xfl.resolve("FB")["data"]["FB"]["entities"]

meta_id = meta_entities[0]["entity_id"]
fb_meta_id = [e["entity_id"] for e in fb_entities
              if e["name"] == "Meta Platforms Inc"][0]
print(f"\nMETA entity_id: {meta_id}")
print(f"FB->Meta entity_id: {fb_meta_id}")
print(f"Same entity: {meta_id == fb_meta_id}")

# ── Step 3: GM — two distinct legal entities ────────────────────────

gm_entities = xfl.resolve("GM")["data"]["GM"]["entities"]
for e in gm_entities:
    end = e["ticker_valid_to"] or "present"
    cik = e["cik"]
    print(f"\nGM entity {e['entity_id']}: {e['name']}")
    print(f"  CIK: {cik}  |  Ticker: {e['ticker_valid_from']} to {end}")

# ── Step 4: DELL — two eras ─────────────────────────────────────────

dell_entities = xfl.resolve("DELL")["data"]["DELL"]["entities"]
for e in dell_entities:
    end = e["ticker_valid_to"] or "present"
    print(f"\nDELL entity {e['entity_id']}: {e['name']}")
    print(f"  Ticker: {e['ticker_valid_from']} to {end}")

# ── Step 5: Pull price data to show why entity matters ──────────────

gm_prices = xfl.prices("GM", start="2010-11-01", end="2011-06-30",
                        fields=["close"])
print(f"\nGM post-IPO prices: {len(gm_prices)} rows, "
      f"{gm_prices['date'].min().date()} to {gm_prices['date'].max().date()}")
print(f"  First close: ${gm_prices['close'].iloc[0]:.2f}, "
      f"Last close: ${gm_prices['close'].iloc[-1]:.2f}")

meta_prices = xfl.prices("META", start="2022-01-01", end="2022-12-31",
                          fields=["close"])
print(f"\nMETA 2022 prices: {len(meta_prices)} rows")
print(f"  First close: ${meta_prices['close'].iloc[0]:.2f}, "
      f"Last close: ${meta_prices['close'].iloc[-1]:.2f}")

dell_prices = xfl.prices("DELL", start="2018-12-01", end="2019-06-30",
                          fields=["close"])
print(f"\nDELL Technologies post-re-IPO: {len(dell_prices)} rows")
print(f"  First close: ${dell_prices['close'].iloc[0]:.2f}, "
      f"Last close: ${dell_prices['close'].iloc[-1]:.2f}")
