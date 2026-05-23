# Full write-up: https://xfinlink.com/blog/djia-current-constituents
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# Pull current Dow Jones Industrial Average constituents (30 stocks)
djia = xfl.index("djia")

print(f"=== Dow Jones Industrial Average: {len(djia)} constituents ===")
print(djia[["ticker", "entity_name"]].to_string(index=False))
print()

# Historical snapshot — who was in the DJIA 10 years ago?
djia_2016 = xfl.index("djia", as_of="2016-05-01")
departed = sorted(set(djia_2016["ticker"]) - set(djia["ticker"]))
joined = sorted(set(djia["ticker"]) - set(djia_2016["ticker"]))

print(f"Left the DJIA since May 2016: {', '.join(departed) if departed else '(none)'}")
print(f"Joined the DJIA since May 2016: {', '.join(joined) if joined else '(none)'}")
