# Financial Data API Licensing: What You Can Redistribute

Most financial data plans priced under $100 a month license the data for one pair of eyes: yours. The moment a value reaches somebody who is not you or your employer, whether that is a customer opening your dashboard, a reader of a public chart, or a subscriber to your newsletter, a different license applies. On most vendors that license is either a separate contract or a tier several times the price. Read the license before the feature list, because coverage gaps and rate limits can be worked around, while a license that forbids the thing your product does cannot.

## What counts as redistribution?

The vocabulary is more settled than it looks. Massive (formerly Polygon.io) draws the clearest line in its business terms: Authorized Users are the customer's own employees, contractors and systems, Edge Users are "individuals or entities that are users of Customer's products and services", and section 6.1(e) forbids making the Information available "to anyone other than Customer, its Authorized Users, or its Edge Users" (as of September 2026). Tiingo puts the same boundary in one sentence, stating that "All data via the API is for internal consumption only" and that "Redistribution is only available upon special request and permission, and comes with additional fees" (as of September 2026).

So the practical test is a headcount. A Slack bot posting closing prices to your own team is internal use. A price rendered in your product's interface for a paying customer is end-user display. A nightly file of the full field set handed to another company is resale, and standard plans almost never permit that at any price.

Derived values sit in the awkward middle. A ratio you computed from licensed inputs generally travels with the license on those inputs, so treat a screen output the same way you treat the raw field it came from. Vendors word this differently, which makes it the one question worth asking in writing before launch rather than after.

## What do the cheap plans actually permit?

Alpha Vantage grants the right to install, use, access, display and run the software "for personal, non-commercial use, unless you and Alpha Vantage have agreed otherwise in writing". Use counts as commercial if you are using the platform "as or on behalf of a corporation, firm, partnership, trust or any other association and not as an individual", or if you provide information from it "as part of any type of commercial activity that allows individuals or entities other than User to access information directly or indirectly". Commercial interest is directed to an email address, premium@alphavantage.co. The premium page prices throughput rather than rights, running from $49.99 a month for 75 requests per minute to $249.99 for 1,200 (all as of September 2026).

Twelve Data separates the two questions explicitly. Individual plans run Basic free, Grow at $29, Pro at $99 and Ultra at $329 a month, and its support documentation states those plans are strictly for personal or internal use and do not permit redistribution or commercial display of data to third parties. The business tiers, named Venture, Enterprise and Enterprise+, permit commercial display and internal usage subject to exchange licensing, while "any redistribution of data requires a separate agreement with Twelve Data" (as of September 2026). Knowing which side of the line you are on before you pay is worth real money, and Twelve Data tells you.

Massive publishes both halves. Individual stock plans are listed at $0, $29, $79 and $199 a month and marked "Non-pros only"; Stocks Business carries a published price of $2,499 a month, with Enterprise quoted case by case (as of September 2026). Combined with the Edge Users clause above, that is a rare thing: a vendor that answers the end-user display question in public.

Tiingo prices internal commercial use at $50 a month, against $30 a month for individuals, and defines the limit plainly: "Internal use means you may only use the data for your own personal use and you may not display or share the data with another person or organization" (as of September 2026). Where redistribution is granted, the terms require the phrase "Data sourced by Tiingo" alongside the data.

xfinlink splits the same ladder into four published tiers. Free and Pro are an individual license covering your own analysis, research and internal applications. Max, at $79 a month, is a company-wide internal license held by one legal entity and used by its employees and contractors. Redistribution, at $249 a month, adds the right to display and deliver values to end users of your own products and services, and it excludes wholesale resale of the data as a dataset, operating a competing data API, and sublicensing redistribution rights onward. Attribution is not required for end-user display at that tier. Output from the entity resolution, search and index endpoints stays internal-use on every tier, which is stated in the terms rather than discovered later.

| Source | Cheapest paid plan | Shows data to your own users | License stated publicly |
| --- | --- | --- | --- |
| Alpha Vantage | $49.99/mo | Written agreement required | Terms of service |
| Twelve Data | $29/mo (Grow) | Business tiers; redistribution by separate agreement | Support documentation |
| Massive | $29/mo (Stocks Starter) | Stocks Business, $2,499/mo | Business terms and pricing page |
| Tiingo | $30/mo (Power) | By request, with fees and attribution | Terms of use |
| SEC EDGAR | Free | Yes, public record | Access guidance |
| xfinlink | $29/mo (Pro) | Redistribution tier, $249/mo | Terms and pricing page |

All figures verified against each vendor's own site in September 2026.

## What about the free sources?

Yahoo Finance is the default answer for hobby projects, and the [yfinance question has its own article](https://xfinlink.com/blog/can-you-use-yahoo-finance-data-commercially). The short version from Yahoo's own developer terms: you may not "sell, lease, share, transfer, or sublicense the Yahoo APIs or access or access codes thereto or derive income from the use or provision of the Yahoo APIs" without prior written permission (as of September 2026).

SEC EDGAR is the genuine exception, and it deserves credit for it. The SEC states that "Anyone can access and download this information for free", asks that automated callers declare a user agent, and caps request rates at 10 per second (as of September 2026). Filings are the public record, so republishing what a company reported carries no license question whatsoever. The cost moves somewhere else: parsing XBRL, reconciling restated periods against originals, tracking companies across name and ticker changes, and keeping all of it running on a schedule. That pipeline is the project, not a step inside it.

## Five questions to ask before paying

Which tier permits display to end users, and what does that tier cost? Are any endpoints carved out of the tier you buy, as reference and universe data commonly are? What happens to data you cached while subscribed once the subscription ends, a point xfinlink's terms settle by forbidding redistribution of cached data afterwards? Does attribution come attached to rights or instead of them? And last, is the price published at all, because a published number is a purchase while a quote is a negotiation, and negotiations have a way of arriving at whatever your funding round was.

Pick the license first and the features second. If the data only ever reaches you, nearly every option here works and the cheapest plan wins on price; our notes on [what a financial data API costs](https://xfinlink.com/blog/what-does-a-financial-data-api-cost) compare those tiers directly. If a value will appear in front of a customer, the shortlist narrows to vendors willing to state the price of that right in public, which among the sources checked here means Massive at $2,499 a month and xfinlink at $249, plus EDGAR for teams prepared to build the pipeline. Tier limits sit on the [pricing page](https://xfinlink.com/pricing) and the field set is in the [docs](https://xfinlink.com/docs).

## FAQ

**Is a chart on my public website redistribution?**
It is end-user display, which internal-use plans exclude. Whether a separate agreement or an upgraded tier covers it depends on the vendor, so check that specific tier before building the page.

**Does crediting the source make redistribution allowed?**
No. Attribution is a condition attached to rights already granted, not a substitute for them. Tiingo requires "Data sourced by Tiingo" where redistribution is permitted, and the permission comes first.

**Can I keep using data I downloaded after cancelling?**
Read the clause on cached data. xfinlink's terms allow the license to run as long as the subscription is active and prohibit redistributing cached or stored data after it ends, and most vendors address this somewhere in their terms.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
