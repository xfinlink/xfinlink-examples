# Full write-up: https://xfinlink.com/blog/tesla-ceo-open-market-sells
# Pro-tier endpoint: requires a Pro API key. Free-tier keys receive 402.
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # Pro key required — see https://xfinlink.com/pricing

# Pull CEO open-market sells at Tesla over the last 2 years
df = xfl.insiders(
    "TSLA",
    transaction_type="open_market_sell",
    insider_role="CEO",
    start="2024-01-01",
)

print(f"=== TSLA — CEO open-market sells since 2024-01-01: {len(df)} transactions ===")
cols = ["transaction_date", "insider_name", "shares", "transaction_price", "transaction_value"]
print(df[cols].sort_values("transaction_date", ascending=False).head(20).to_string(index=False))
print()

total_value = df["transaction_value"].sum()
total_shares = df["shares"].sum()
print(f"Total shares sold: {total_shares:,.0f}")
print(f"Total dollar value:  ${total_value:,.0f}")
