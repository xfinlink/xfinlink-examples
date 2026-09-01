# Full write-up: https://xfinlink.com/blog/data-for-comparable-company-analysis
"""Build a comparable company analysis table: peer set, multiples, peer median."""
import pandas as pd
import xfinlink as xfl

pd.set_option("display.width", 200)
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TARGET = "DUK"
MULTIPLES = ["ev_ebitda", "ev_revenue", "pe_ratio", "pb_ratio"]

# 1. What is the target, in the classification the peer set will be drawn from?
info = xfl.resolve(TARGET)["data"][TARGET]["entities"][0]["classifications"]
print("target: %s | %s | %s | SIC %s"
      % (TARGET, info["gics_sector"], info["gics_sub_industry"], info["sic_code"]))

# 2. Every entity in the sector, paginated, then narrowed to the sub-industry
#    and to current index members.
pages, off = [], 0
while True:
    page = xfl.search(gics_sector=info["gics_sector"], limit=500, offset=off)
    if page.empty:
        break
    pages.append(page)
    off += len(page)
    if len(page) < 500:
        break
sector = pd.concat(pages, ignore_index=True)
live = xfl.index("sp500")
peers = sector[(sector["gics_sub_industry"] == info["gics_sub_industry"])
               & (sector["entity_id"].isin(live["entity_id"]))]
print("sector entities searched: %d   peers in the S&P 500: %d" % (len(sector), len(peers)))

# 3. The multiples, on the latest annual period for each peer.
m = xfl.metrics(entity_id=[int(i) for i in peers["entity_id"]],
                period_type="annual", period="2y",
                fields=["market_cap"] + MULTIPLES)
m = m.sort_values("period_end").groupby("entity_id").tail(1)

# Standard comps screen: a peer enters the median only if it reports every
# multiple in the table, so each column is computed on the same set of names.
comps = m.dropna(subset=MULTIPLES).sort_values("ev_ebitda")
print("peers reporting all %d multiples: %d" % (len(MULTIPLES), len(comps)))
print("fiscal period ends in the table: %s"
      % ", ".join(sorted(comps["period_end"].astype(str).unique())))

print()
print(comps[["ticker", "entity_name", "period_end", "market_cap"] + MULTIPLES]
      .to_string(index=False))

med = comps[MULTIPLES].median()
tgt = comps[comps["ticker"] == TARGET][MULTIPLES].iloc[0]
print("\npeer median vs %s" % TARGET)
for k in MULTIPLES:
    print("  %-11s median %7.2f   %s %7.2f   premium %+6.1f%%"
          % (k, med[k], TARGET, tgt[k], 100 * (tgt[k] / med[k] - 1)))
