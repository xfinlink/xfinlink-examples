# Can You Use Yahoo Finance Data Commercially?

No. Yahoo's Terms of Service, read on 29 July 2026, state: "Unless otherwise expressly stated, you may not access or reuse the Services, or any portion thereof, for any commercial purpose." They separately prohibit collecting data "using any automated means, devices, programs, algorithms or methodologies, including but not limited to robots, spiders, scrapers, data mining tools, or data gathering or extraction tools, for any purpose without our express, prior permission". A script that pulls prices from Yahoo and a report that bills a client for the resulting chart both sit inside those sentences. The library sitting in the middle does not change them.

Every source draws this line somewhere, and the position of the line rarely matches what people assume from the price. What follows is a reading of published terms as of 29 July 2026, not legal advice; the pages move, and only counsel can tell you where your particular use falls.

## What do Yahoo's terms actually say?

Three clauses matter. The commercial-purpose bar quoted above is the broad one. Below it sits a distribution clause: "Unless you have explicit written permission, you must not reproduce, modify, rent, lease, sell, trade, distribute, transmit, broadcast, publicly perform, create derivative works based on, or exploit for any commercial purposes, any portion or use of, or access to, the Services (including content, advertisements, APIs, and software)". The third one is aimed squarely at data products, prohibiting use of any material or content to "create any database, archive, mobile application, data feed, widget or any other aggregated data source that competes with or constitutes a material substitute for the Services ...".

yfinance, the Python library most people actually mean when they say "the Yahoo Finance API", is direct about its own position. Its documentation carries a legal disclaimer stating that "yfinance is not affiliated, endorsed, or vetted by Yahoo, Inc. It's an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes." Readers are told they "should refer to Yahoo!'s terms of use ... for details on your rights to use the actual data downloaded", and the disclaimer closes with a plain instruction: "Remember - the Yahoo! finance API is intended for personal use only."

That disclaimer is worth taking at face value. The library is licensed open source; the data flowing through it is not the library's to license. Installing something with pip grants rights over the code and none at all over what arrives in the DataFrame.

## What counts as commercial use?

Most people underestimate the reach of the term, because the intuitive test is whether money changed hands for the data. Vendors use a wider test. Alpha Vantage publishes the clearest version: its terms grant the right to use the platform "for personal, non-commercial use, unless you and Alpha Vantage have agreed otherwise in writing", then list the conditions under which usage is commercial. Two of them catch people who never sold anything. Usage is commercial if "you are using the Alpha Vantage Platform as or on behalf of a corporation, firm, partnership, trust or any other association and not as an individual", and also if "you are currently employed or have an active affiliation with a financial planning advisor, insurance company, investment advisor, investment bank, money manager, registered representative, securities broker-dealer, or any owner, partner, affiliate or associated person of the preceding".

Read the second one twice. An analyst at a registered advisor, running a private research script at home on a personal laptop, is a commercial user under that definition. Employment is the trigger, not revenue.

Four situations, roughly in order of how much trouble they cause:

A personal script on a personal machine, output seen by nobody else, is the case every free tier is written for. A tool used inside an employer already fails the individual-licence test at most vendors, even when the output never leaves the building. A number pasted into a deliverable a client pays for has left the building. A product that displays values to its own users is redistribution, which nearly every agreement treats as a separate grant rather than a bigger version of the same one.

## Where does each source draw the line?

Every figure and quotation below was read off the source's own pages on 29 July 2026.

| Source | What the published terms say | Route to commercial use |
|---|---|---|
| Yahoo Finance (including via yfinance) | "Unless otherwise expressly stated, you may not access or reuse the Services, or any portion thereof, for any commercial purpose"; automated collection requires "express, prior permission" | None published |
| SEC EDGAR, data.sec.gov REST APIs | "These APIs do not require any authentication or API keys to access"; the same page separately documents filer and user API tokens. Access still needs a User-Agent declaring contact details, and a generic one is answered with HTTP 403 | No licence fee, no key; filings are the public record |
| Alpha Vantage | Licence granted "for personal, non-commercial use, unless you and Alpha Vantage have agreed otherwise in writing", with employment at a financial firm listed as a commercial trigger | Written agreement, by email to premium@alphavantage.co |
| Twelve Data | Licence "solely for Internal Use", defined as "use solely for Customer's internal business purposes and not for redistribution or external commercial purposes"; free-tier users may not "use Free Tier data for commercial purposes" | Subscription tier, Redistribution Rights Add-On, or separate agreement via sales@twelvedata.com |
| Massive (polygon.io redirects here) | Self-serve stocks plans from $0 to $199 per month are each marked "Individual use", and the $199 Stocks Advanced plan carries "Non-pros only" as well; the $1,999 per month Stocks Business plan is marked "Business use" | Published and self-serve at the Business tier |
| xfinlink | Free and Pro are an individual licence; Business is a company-wide internal licence; Redistribution permits display of values to end users of your own products | Published and self-serve: $149 per month internal, $399 per month redistribution |

Sources in row order: the [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html); the [SEC's EDGAR API page](https://www.sec.gov/search-filings/edgar-application-programming-interfaces); the [Alpha Vantage terms of service](https://www.alphavantage.co/terms_of_service/); the [Twelve Data terms](https://twelvedata.com/terms); [massive.com/pricing](https://massive.com/pricing) and [massive.com/business](https://massive.com/business), reached because polygon.io returns a redirect to massive.com; the xfinlink [terms](https://xfinlink.com/terms) and [pricing page](https://xfinlink.com/pricing).

Note what the free rows have in common. Every free tier in that table, including xfinlink's, is an individual licence. Free access and commercial rights are separate products everywhere, and a generous request budget says nothing about the second one. The [guide to free stock market data APIs](https://xfinlink.com/blog/free-stock-market-data-apis) covers what the request limits themselves are worth.

## What does a commercial licence cost?

Two of these vendors make you ask. Alpha Vantage routes commercial use to an email address, and Twelve Data routes redistribution to a sales conversation and a separate agreement. Neither approach is unreasonable, and both mean the answer to "what will this cost" arrives days after the question, which is awkward when the decision is whether to start building at all.

Massive publishes its number, and the number is self-serve: $1,999 per month for Stocks Business, checkout on the page, no call required. For a firm that needs real-time trades and quotes across US exchanges, that is a fair price for a tier that includes exchange coverage a cheaper plan cannot carry.

The distinction that saves money is internal versus external. A quant team using data to make decisions is not doing the same thing as a dashboard showing the same numbers to paying users, and paying redistribution rates for internal analysis is a common and expensive mistake. xfinlink prices those as two steps: $149 per month for a company-wide internal licence, covering employees and contractors of the subscribing entity, and $399 per month to display values to end users of your own products. Both sit on the [pricing page](https://xfinlink.com/pricing) with the per-endpoint limits in the [docs](https://xfinlink.com/docs).

## Which licence do you actually need?

Work backwards from how far the number travels.

If it dies on your laptop, a free individual tier covers you, and yfinance is genuinely fine for a weekend script or a class exercise. If it reaches a colleague, you need a company licence, and your employer's industry may have already made you a commercial user before you wrote a line of code. If it reaches someone who pays you, or appears on a screen you do not control, you need redistribution rights in writing.

Then pick the source whose published terms match that answer without a negotiation. A licence you can read and buy on the same afternoon is worth more than a cheaper one that needs a salesperson to quote it.

## FAQ

**Is yfinance illegal to use?**
The library is open source and freely installable. The constraint sits on the data it retrieves, which is governed by Yahoo's terms rather than the library's licence, and those terms restrict commercial reuse and automated collection.

**Does an internal company tool count as commercial use?**
At most vendors, yes. Twelve Data's standard licence is for "Internal Use" and still requires a paid subscription; Alpha Vantage treats use "as or on behalf of a corporation, firm, partnership, trust or any other association" as commercial. Internal use is a cheaper licence, not a free one.

**Are SEC filings free to use commercially?**
The SEC states that its data.sec.gov REST APIs "do not require any authentication or API keys to access", and the filings themselves are the public record. No authentication is not the same as no conditions: requests need a User-Agent declaring contact details, and a generic one is answered with HTTP 403. The work after that is parsing, since raw XBRL arrives as filer-chosen tags, and mapping them to consistent fields across thousands of companies is the part vendors charge for.

**What happens if a data source changes its terms?**
Nothing warns you. Terms pages are updated without notice to existing users, which is an argument for building against a vendor that has a contract with you and an account to notify, rather than an endpoint you found.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
