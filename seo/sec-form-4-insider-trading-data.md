# How to Get SEC Form 4 Insider Trading Data in Python

Insider trading data comes from SEC Forms 3, 4 and 5, the ownership reports that officers, directors and large shareholders file under Section 16 of the Securities Exchange Act of 1934. Form 4 is the one that tracks trades, and it must be filed "before the end of the second business day following the day on which a transaction resulting in a change in beneficial ownership has been executed" ([sec.gov Form 4](https://www.sec.gov/about/forms/form4.pdf), read 10 August 2026). Every filing is public on EDGAR as an XML document. In Python, `xfl.insiders("NVDA", period="1y")` returns one row per transaction with the SEC transaction code already decoded, and that decoding is what decides whether the analysis holds up.

## What are Forms 3, 4 and 5?

Section 16 obliges a director, an officer, or a beneficial owner of more than 10% of a registered class of equity securities to report both holdings and trades. Three forms carry the obligation, and each has its own clock.

Form 3 is the initial statement of what an insider already owns, filed "within 10 days after the event by which the person becomes a reporting person" ([sec.gov Form 3](https://www.sec.gov/about/forms/form3.pdf), read 10 August 2026). Form 4 reports each subsequent change in beneficial ownership on the two-business-day clock quoted above. Form 5 sweeps up transactions that were exempt from Form 4 or were simply missed, and is filed "on or before the 45th day after the end of the issuer's fiscal year" ([sec.gov Form 5](https://www.sec.gov/about/forms/form5.pdf), read 10 August 2026).

That two-day deadline is the reason Form 4 is worth reading at all. Compare it with a Form 13F, the quarterly institutional holdings report explained in the [13F guide](https://xfinlink.com/blog/what-is-a-13f-filing): a 13F arrives up to 45 days after the quarter it describes and names funds rather than people. A Form 4 names the person, the trade date, the share count and the price, usually within the week.

## Why is the raw EDGAR feed hard to use for analysis?

Access is not the obstacle. Shape is.

Each Form 4 is a separate XML document, and one document can carry several transaction lines split across two tables, one for ordinary shares and one for derivatives. Building a panel means walking the daily filing index, fetching every document, and flattening those tables into rows. The SEC caps automated access at "10 requests/second" and asks callers to declare a User-Agent header carrying a company name and contact address ([sec.gov, Accessing EDGAR Data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), read 10 August 2026), so the exercise is a rate-limited crawl rather than a download.

A bulk route exists. The SEC publishes Insider Transactions Data Sets extracted from Forms 3, 4 and 5, covering January 2006 to June 2026, and states that "the data sets will be updated quarterly" ([sec.gov](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets), read 10 August 2026). That removes the crawl and adds two constraints in its place: history begins in 2006, and the current quarter is absent until the next posting. The same tradeoff appears in fundamentals, and the [EDGAR API comparison](https://xfinlink.com/blog/sec-edgar-api-vs-fundamentals-api) works through it in more detail.

Either route leaves the harder problem untouched. The filing carries a raw transaction code, a single letter, and nothing that tells a program what the letter means.

## How do you pull Form 4 data in Python?

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

df = xfl.insiders("NVDA", period="1y")
df["date"] = df["transaction_date"].dt.date

print(df[["date", "insider_name", "insider_role", "transaction_code",
          "transaction_type", "shares", "transaction_price"]].head(5).to_string(index=False))
print(df["transaction_type"].value_counts().to_string())
```

```
      date   insider_name insider_role transaction_code transaction_type  shares  transaction_price
2026-08-05     COXE TENCH     Director                G             gift  500000                0.0
2026-06-25  HUDSON DAWN E     Director                A   grant_or_award    1211                0.0
2026-06-25    Dabiri John     Director                A   grant_or_award    1211                0.0
2026-06-25 Neal Stephen C     Director                A   grant_or_award    1211                0.0
2026-06-25   LORA MELISSA     Director                A   grant_or_award    1211                0.0

transaction_type
open_market_sell    361
grant_or_award       24
tax_withholding      23
gift                 15
other                 5
```

Both the raw `transaction_code` and a decoded `transaction_type` sit on every row, alongside the insider name, the role, the acquired-or-disposed flag, the share count, the price, and the holdings figure after the transaction. History runs back to 1996, and `form_type` separates Forms 3, 4 and 5 when a study needs only one of them. The full field list is in the [docs](https://xfinlink.com/docs).

## Which Form 4 transaction codes matter?

Eight codes account for almost everything a large company files. The SEC descriptions below are quoted from the Form 4 general instructions ([sec.gov](https://www.sec.gov/about/forms/form4.pdf), read 10 August 2026).

| Code | SEC description | What it means for analysis | Decoded value |
|---|---|---|---|
| P | "Open market or private purchase of non-derivative or derivative security" | The insider paid cash. The signal everyone is looking for. | `open_market_buy` |
| S | "Open market or private sale of non-derivative or derivative security" | A genuine sale, though often a scheduled one. | `open_market_sell` |
| A | "Grant, award or other acquisition pursuant to Rule 16b-3(d)" | Compensation. No decision to buy is expressed. | `grant_or_award` |
| F | "Payment of exercise price or tax liability by delivering or withholding securities incident to the receipt, exercise or vesting of a security issued in accordance with Rule 16b-3" | Shares withheld to settle a tax bill. Mechanical. | `tax_withholding` |
| M | "Exercise or conversion of derivative security exempted pursuant to Rule 16b-3" | An option turning into stock, not a purchase. | `option_exercise` |
| G | "Bona fide gift" | A transfer at no price. | `gift` |
| D | "Disposition to the issuer of issuer equity securities pursuant to Rule 16b-3(e)" | Sold back to the company, not into the market. | `sale_to_issuer` |
| C | "Conversion of derivative security" | A change of instrument. | `conversion_of_derivative` |

Codes A, F, M and G describe things that happen to an insider rather than things an insider chooses. They dominate the row count.

## What is the most common mistake in insider analysis?

Counting every disposition as a sale. The `acquisition_or_disposition` flag reads "D" for an open-market sale, for shares withheld to pay tax, for a gift, and for the disposal leg of an option exercise, so a filter on that flag alone sweeps all four into one number.

Across eight large caps over three years, the difference is not marginal.

```python
tickers = ["AAPL", "MSFT", "JPM", "XOM", "KO", "WFC", "GM", "PFE"]
d = xfl.insiders(tickers, period="3y")

sold = d[d["acquisition_or_disposition"] == "D"]
print(sold.groupby("transaction_type")["transaction_value"].agg(["size", "sum"])
          .sort_values("sum", ascending=False).to_string())

open_market = sold[sold["transaction_type"] == "open_market_sell"]["transaction_value"].sum()
print(f"every disposition counted as selling: ${sold['transaction_value'].sum()/1e9:,.2f}bn")
print(f"open-market sales only:               ${open_market/1e9:,.2f}bn")
```

```
                  size           sum
transaction_type
open_market_sell   388  2.006649e+09
tax_withholding    508  1.419691e+09
other                2  3.120996e+05
gift                72  0.000000e+00
option_exercise     30  0.000000e+00

every disposition counted as selling: $3.43bn
open-market sales only:               $2.01bn
```

The naive figure is 1.71 times the real one. Tax withholding alone supplies $1.42 billion of the gap, spread over 508 rows, and those shares never reached the market: the issuer withheld them at vest to settle the recipient's tax liability. A dashboard built on the disposition flag reports a wave of executive selling every time restricted stock vests on schedule.

## How do you filter to the transactions that carry signal?

Filter on the decoded type rather than the direction, and add a size floor so that token purchases do not crowd the result.

```python
buys = xfl.insiders(tickers, period="3y",
                    transaction_type="open_market_buy",
                    min_value=250_000)
buys["date"] = buys["transaction_date"].dt.date

print(buys[["ticker", "date", "insider_name", "insider_role",
            "shares", "transaction_price", "transaction_value"]]
      .sort_values("date", ascending=False).to_string(index=False))
```

```
ticker       date       insider_name insider_role  shares  transaction_price  transaction_value
   PFE 2026-08-05 Buckley Mortimer J     Director   37632            25.5200       9.603686e+05
   PFE 2026-08-05  BLAYLOCK RONALD E     Director   39231            25.4600       9.988213e+05
    KO 2025-10-24      LEVCHIN MAX R     Director    7206            69.8706       5.034875e+05
    KO 2025-10-23      LEVCHIN MAX R     Director    4197            70.3062       2.950751e+05
  MSFT 2025-04-23 SMITH BRADFORD LEE    President    3842           377.4650       1.450221e+06
   PFE 2025-02-13  BLAYLOCK RONALD E     Director   19457            25.6500       4.990720e+05
    GM 2024-07-26    JACOBSON PAUL A          CFO   25000            44.1100       1.102750e+06
   XOM 2024-06-17    DREYFUS MARIA S     Director   18310           109.2510       2.000386e+06
   XOM 2023-11-06    UBBEN JEFFREY W     Director   50000           105.9882       5.299410e+06
   XOM 2023-11-06    UBBEN JEFFREY W     Director   50000           105.9872       5.299360e+06
   XOM 2023-11-06    UBBEN JEFFREY W     Director  150000           105.9510       1.589265e+07
```

Eleven rows out of 1,916. That ratio is the point of the whole exercise. Insiders at large companies are paid in stock, so acquisitions arrive on a vesting calendar and disposals arrive on a tax calendar, while a purchase requires the person to write a cheque against a position they already hold. Scarcity is what gives code P its information content, and averaging it together with the other codes destroys exactly that.

The remaining filters narrow it further. `insider_role` matches a substring, so `insider_role="CEO"` or `insider_role="Director"` separates the officers who see the operating numbers daily from board members who see them quarterly. `min_value` screens on dollars rather than shares, which keeps the comparison honest across a $25 stock and a $380 one. Setting `include_amendments=True` brings in corrected filings when the audit trail matters.

## FAQ

**Is Form 4 data free?**
The filings are free on EDGAR, and the SEC's quarterly data sets are free to download. The xfinlink `insiders` endpoint is on the paid plans; see [pricing](https://xfinlink.com/pricing) for the tiers and their limits.

**How quickly does a trade appear after it happens?**
Form 4 is due before the end of the second business day after the trade, so most transactions become public within a week of execution. Each row carries both `transaction_date` and `filing_date`, which is how the reporting lag can be measured directly rather than assumed.

**Do open-market purchases predict returns?**
That is an empirical question and the answer varies by horizon, company size and role. The requirement is a clean sample first: a study that treats grants and tax withholding as trades is measuring the compensation calendar, not insider conviction.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
