# Ticker vs CIK vs FIGI: Which Company ID to Use

Store the CIK when the unit of work is a filing, store the FIGI when the unit of work is a traded instrument, and treat the ticker as a label rather than a key. The three are not competing versions of one thing. A CIK names whoever files with the SEC, a FIGI names a security as it trades, and a ticker names whichever company happens to be listed under that string today. Anything that joins financial statements to prices across a decade needs at least two of them, plus a company-level key that neither system sets out to provide.

## What does each identifier actually identify?

The SEC describes the CIK narrowly, and the wording repays a literal reading. The Central Index Key "is used on the SEC's computer systems to identify corporations and individual people who have filed disclosure with the SEC", EDGAR "assigns to filers a unique numerical identifier, known as a Central Index Key (CIK), when they sign up to make filings to the SEC", and "CIK numbers remain unique to the filer; they are not recycled". Both pages were read on 18 August 2026. The object being identified is a filer. Executives who file Form 4 carry their own CIK for the same reason a corporation does.

OpenFIGI writes in a similar register about instruments. "FIGIs, however, never change, are never reused, and are permanent, allowing users to maintain data integrity over a time period of multiple corporate actions and changes", and in equities the identifier "is assigned at venue, country and global share class level". FIGI is a standard of the Object Management Group with Bloomberg as Registration Authority, and the site states that the web API "is free to use without daily, weekly or monthly limitations". Read 18 August 2026.

Neither definition mentions a company. That omission is the whole subject of this guide.

| Identifier | Assigned by | Names | Permanent | Reaches |
|---|---|---|---|---|
| Ticker | The listing venue | Whatever is listed under the string now | No | Quotes and prices, for as long as the listing lasts |
| CIK | EDGAR, at filer registration | The filer, corporate or personal | Yes, and not recycled | Filings, XBRL statements, insider forms |
| FIGI | OMG standard, Bloomberg as Registration Authority | An instrument, at venue, country and share class level | Yes, never reused | Instrument-level market and reference data |

Sources, read 18 August 2026: the [SEC CIK lookup page](https://www.sec.gov/search-filings/cik-lookup), the [SEC Accessing EDGAR Data page](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), and the [OpenFIGI features page](https://www.openfigi.com/about/features) plus the [OpenFIGI about page](https://www.openfigi.com/about).

## Why does a ticker break a database?

Because the string outlives its owner, and nothing in the data announces the change of hands.

```python
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

info = xfl.resolve("GM")
for e in info["data"]["GM"]["entities"]:
    print(e["entity_id"], e["name"], e["cik"],
          e["ticker_valid_from"], e["ticker_valid_to"] or "current")
```

```
4 General Motors Corporation (pre-2009 bankruptcy) 0000040730 1962-07-02 2009-06-01
5 General Motors Company 0001467858 2010-11-18 current
```

Two companies, two CIKs, one set of letters. The SEC issued 0001467858 because a new legal entity began filing, which is the documented behaviour: a CIK is unique to the filer, so a new filer means a new number. The identifier is doing its job. It simply does not carry the information that the letters GM meant a different company until June 2009, nor that no company held those letters between June 2009 and November 2010.

The consequence shows up the moment prices are requested.

```python
live = xfl.prices("GM", period="max", fields=["close"])
old = xfl.prices(entity_id=4, start="2008-01-02", end="2008-12-31", fields=["close"])

print(f"ticker GM:   {len(live)} sessions from {live['date'].min().date()}")
print(f"entity_id 4: {len(old)} sessions, close {old['close'].iloc[0]} -> {old['close'].iloc[-1]}")
```

```
ticker GM:   3956 sessions from 2010-11-18
entity_id 4: 253 sessions, close 24.41 -> 3.2
```

The ticker's full history starts in November 2010, because that is when the current company listed. The 2008 collapse from $24.41 to $3.20 belongs to a different legal person, reachable by its entity identifier. A vendor that keys on the string alone has two ways to be wrong here: drop the earlier company entirely, or paste the two series together into a chart of a 2008 crash followed by a recovery that no shareholder ever received. We have worked through [the GM case](/blog/gm-bankruptcy-entity-resolution-python) and [recycled tickers more generally](/blog/ticker-recycling-dangers-python) with the numbers attached.

## When is the CIK the right key?

Whenever the row being stored is a filing. Reconciling to a 10-K, pulling XBRL facts, tracking Form 4 activity, checking who signed a registration statement: all of these are filer-level questions, and the filer-level identifier is the one EDGAR itself uses. It joins cleanly to every SEC endpoint, it survives a name change, and it costs nothing.

Two limits are worth knowing before treating it as a company key. A CIK belongs to whoever registered, so a restructured business appears under a new number while the old one stays valid and frozen, exactly as the GM output shows. And EDGAR carries no prices, so a CIK reaches statements and stops there.

Market-data vendors vary in whether they hand you the join. Alpha Vantage's Company Overview endpoint, called with its documented demo key on 18 August 2026, returns a `CIK` field alongside `Symbol` and `Name`, which makes the hop from a price row to a filing straightforward for anyone already using that endpoint.

## When is the FIGI the right key?

Whenever the row being stored is an instrument. Order routing, execution records, position keeping, anything where a share class or a listing venue is the real subject: the FIGI was built for that, and its permanence guarantee is exactly what a trade blotter needs.

That same precision rules it out as a company key. Assignment happens at venue, country and share class level, by the standard's own description, so a dual-class issuer holds more than one identifier and a company-level total built by summing FIGIs double-counts or splits, depending on which way the error falls. Instrument identity and issuer identity are different questions, and the FIGI answers the first one deliberately.

## How do you carry all three in Python?

Keep a company-level key of your own, and hang the external identifiers off it. That is the shape xfinlink stores: a permanent `entity_id` per company, built from SEC EDGAR public filings and market data, with the public identifiers attached to the same record.

```python
res = xfl.resolve(["AAPL", "MSFT", "META", "JPM", "XOM"])
for t, payload in res["data"].items():
    e = payload["entities"][0]
    print(f"{t:<6}{e['entity_id']:>6}  {e['cik']}  {e['figi']}  {e['name']}")
```

```
AAPL       1  0000320193  BBG000B9XRY4  Apple Inc
JPM     1537  0000019617  BBG000DMBXR2  JPMORGAN CHASE & CO
META       2  0001326801  BBG000MM2P62  Meta Platforms Inc
MSFT    8611  0000789019  BBG000BPH459  MICROSOFT CORP
XOM     2735  0000034088  BBG000GZQ728  EXXON MOBIL CORP
```

One call returns the internal key for joining prices, fundamentals, metrics, insider transactions and institutional holdings, the CIK for reaching EDGAR directly, and the composite FIGI for reconciling against an instrument-keyed system. `entity_id` is accepted as an argument everywhere a ticker is, which is how the pre-2009 General Motors was reached above. The [docs](/docs) list what each endpoint returns, and the [pricing page](/pricing) covers history depth by plan. Classification codes work the same way and follow the same trap of looking interchangeable when they are not, which the [GICS, SIC and NAICS comparison](/blog/gics-vs-sic-vs-naics) sets out.

## FAQ

**Can two different companies share one ticker?**
Yes, and it is common enough to matter. The letters GM covered one company from 1962 to 2009 and a different one from 2010 onward, each with its own CIK and its own price history.

**Is a CIK enough to identify a company over time?**
For filings, yes, and the SEC states that CIK numbers are not recycled. For a business that has passed through a restructuring, no: a new filer receives a new CIK, so two numbers describe what most people would call one company.

**Does a FIGI cost anything?**
No. FIGI is a standard of the Object Management Group, and OpenFIGI states that its web API is free to use without daily, weekly or monthly limitations.

**Which key should a research database use?**
A company-level key you control, with the CIK and the FIGI stored as attributes rather than as the primary key. Filings and instruments are both downstream of the company, and only the company key survives a ticker change, a share-class split and a restructuring at once.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
