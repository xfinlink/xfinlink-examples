# Full write-up: https://xfinlink.com/blog/ndx100-current-constituents
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Pull current Nasdaq 100 constituents
ndx100 = xfl.index("ndx100")

print(f"=== Nasdaq 100: {len(ndx100)} constituents ===")
print(ndx100[["ticker", "entity_name"]].head(15).to_string(index=False))
print()

# Historical snapshot — who was in the Nasdaq 100 five years ago?
ndx100_2021 = xfl.index("ndx100", as_of="2021-05-01")
departed = sorted(set(ndx100_2021["ticker"]) - set(ndx100["ticker"]))[:15]
joined = sorted(set(ndx100["ticker"]) - set(ndx100_2021["ticker"]))[:15]

print(f"Left the Nasdaq 100 since May 2021 (sample): {', '.join(departed)}")
print(f"Joined the Nasdaq 100 since May 2021 (sample): {', '.join(joined)}")
