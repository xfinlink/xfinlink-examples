# How to Give an LLM Financial Data With an MCP Server

An MCP server is the piece that lets a chat model read real market data instead of recalling something from its training run: paste a server URL into the client's connector settings, and the model starts answering price and financial-statement questions by calling a vendor API. Alpha Vantage, Massive, and xfinlink each publish one as of 26 July 2026. For the hosted versions there is nothing to install and no code to write.

## What does an MCP server actually do?

MCP, the Model Context Protocol, is an open standard for advertising tools to a language model. The server publishes a list of tools, each with a description and a JSON schema for its arguments; the model picks one, sends arguments, and receives structured JSON in return. Because the list is discoverable, the client learns what is available at connection time rather than from a system prompt somebody has to keep current.

Asking the xfinlink server what it offers returns eight tools:

```bash
# free at https://xfinlink.com/signup
curl -s -X POST "https://api.xfinlink.com/mcp?api_key=YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

```
['get_prices', 'get_fundamentals', 'get_metrics', 'resolve_ticker',
 'search', 'get_index', 'get_index_events', 'get_insiders']
```

A model with no data connection, asked for Apple's closing price last Tuesday, has two options: decline, or produce a number that looks right. Training cutoffs make the second outcome common, and the fabricated number arrives in the same confident prose as a real one. A tool call removes the guess for anything the tools cover, and it leaves an inspectable record of what was requested.

## Which vendors publish a financial data MCP server?

Every figure below was checked against the vendor's own pages on 26 July 2026. Terms move, so confirm before committing.

| Vendor | Hosted server | Auth | Tool design | Free tier |
|---|---|---|---|---|
| Alpha Vantage | `https://mcp.alphavantage.co/mcp`, plus a local stdio option via `uvx` | OAuth on the remote server | 100+ tools across nine categories, including options, technical indicators, forex and economic series | 25 API requests per day; premium plans start at $49.99/month |
| Massive | `https://mcp.massive.com/` | OAuth | A small composable set: search the endpoint catalogue, read parameter documentation, then execute a REST call | Not stated on the AI-tools quickstart page |
| xfinlink | `https://api.xfinlink.com/mcp?api_key=YOUR_API_KEY` | API key in the URL | Eight tools, one per endpoint: prices, fundamentals, metrics, entity resolution, search, index constituents, index events, insider filings | 100 requests per day on a rolling one-year history window; Pro $29/month unlocks full history |
| yfinance | None documented | n/a | Python library only: `Ticker`, `Tickers`, `download()` | Free library, no key |

Sources for each row: the [Alpha Vantage MCP repository](https://github.com/alphavantage/alpha_vantage_mcp) and its [premium pricing page](https://www.alphavantage.co/premium/); the [Massive AI tools quickstart](https://massive.com/docs/ai-tools/quickstart), noting that `polygon.io` returns a 301 redirect to `massive.com`; the xfinlink [docs](https://xfinlink.com/docs) and [pricing page](https://xfinlink.com/pricing); the [yfinance documentation](https://ranaroussi.github.io/yfinance/), which states that the project is not affiliated with or endorsed by Yahoo and that the underlying API is intended for personal use.

Pick on coverage first. Options chains, intraday bars and technical indicators point at Alpha Vantage or Massive. Daily US equity history, SEC-filing fundamentals and index membership are what the xfinlink tools return. yfinance remains a reasonable free downloader for a hobby project, but reaching it from a chat client means writing the wrapper yourself.

## How do you connect one?

Two shapes cover almost every client. Web products (claude.ai, ChatGPT with developer mode enabled, Perplexity, Grok) take a name and a server URL in a connectors settings page. Desktop and IDE products take a JSON block. Cursor reads `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "xfinlink": {
      "url": "https://api.xfinlink.com/mcp?api_key=YOUR_API_KEY"
    }
  }
}
```

Claude Desktop expects the same server behind `mcp-remote` in `claude_desktop_config.json`, and needs a restart afterwards. Step-by-step instructions for each client are in the [xfinlink docs](https://xfinlink.com/docs#docs-vibe).

## What comes back from a tool call?

Here is the raw result of `get_prices` for Apple over five sessions, truncated to two rows and reformatted for width. This is what the model reads before it writes a sentence:

```json
{"data":[
 {"entity_id":1,"ticker":"AAPL","entity_name":"Apple Inc","gics_sector":"Information Technology",
  "date":"2026-07-23","open":321.73,"high":323.3,"low":319.35,"close":321.66,"adj_close":321.66,
  "volume":40795222,"return_daily":-0.0129798398232531,"shares_outstanding":15115823000,
  "split_ratio":null,"dividend":null},
 {"entity_id":1,"ticker":"AAPL","entity_name":"Apple Inc","gics_sector":"Information Technology",
  "date":"2026-07-24","open":322.04,"high":334.37,"low":321.62,"close":333.02,"adj_close":333.02,
  "volume":47402209,"return_daily":0.0353167941304482,"shares_outstanding":15115823000,
  "split_ratio":null,"dividend":null}],
 "meta":{"tickers_resolved":{"AAPL":1},"interval":"1d","data_through":"2026-07-24",
  "count":5,"has_more":false}}
```

Three details in that payload change how an answer should be read. `close` is the raw traded price and steps across split dates; `adj_close` carries the split adjustment; `return_daily` is the total return including dividends. The tool descriptions state those conventions, so a model that reads them will usually pick the right column, and `meta.data_through` tells it how fresh the series is instead of leaving it to assume today.

## Why does the model need to know which company held a ticker?

Ask about GM in 2008 and the honest answer is a question back: which General Motors? The pre-bankruptcy corporation and the company that listed in 2010 are separate legal entities that share three letters. `resolve_ticker` returns both:

```json
{"data":{"GM":{"entities":[
  {"entity_id":4,"name":"General Motors Corporation (pre-2009 bankruptcy)",
   "entity_type":"corporation","country":"US","cik":"0000040730"},
  {"entity_id":5,"name":"General Motors Company",
   "entity_type":"corporation","country":"US","cik":"0001467858"}]}}}
```

Each carries its own SEC Central Index Key and its own permanent entity ID, and the price and fundamentals tools key on that ID rather than on the ticker string. The same mechanism keeps a series intact through a rename: Facebook to Meta, or the several symbols Travelers has traded under. An agent that stitches history by ticker alone will silently glue two companies together, which is [what ticker recycling does to a backtest](https://xfinlink.com/blog/ticker-recycling-dangers-python), and the [GM case](https://xfinlink.com/blog/gm-bankruptcy-entity-resolution-python) shows the shape of the damage.

## What to check before wiring a server into an agent

Read the tool descriptions first. They are the only documentation the model gets, and a server whose descriptions omit adjustment conventions will produce answers that are wrong in ways the transcript does not reveal.

Then the boring operational things. Check where the key lives: a key in a URL is convenient and ends up in client config files, whereas OAuth keeps it out of them. Check the row cap per call, because a tool that silently returns the first hundred rows of a five-year request will make a model narrate a truncated series as if it were complete. Check the daily request budget against how chatty your agent is; a single research question can fire a dozen tool calls.

## FAQ

**Does an MCP server stop a model inventing numbers?**
It removes the reason to invent them for anything the tools cover. The model can still misread a column or aggregate badly, so for anything that matters, ask it to show the tool output alongside its answer.

**Do I need to be able to code to use one?**
No. The hosted servers need a URL pasted into a settings page, and the model writes the queries. Code becomes worthwhile when the same analysis runs on a schedule, at which point the underlying REST API or Python client is the better target.

**Can an MCP server run a screen across hundreds of tickers?**
Within limits. Per-call ticker caps and row caps apply the same way they do over REST, so wide screens work better as a batched script than as a chat request.

**Is there an official yfinance MCP server?**
Not in its documentation, which covers `Ticker`, `Tickers` and `download()` and makes no mention of MCP as of 26 July 2026. Community wrappers exist, and they inherit yfinance's own terms.

---

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install xfinlink`*
