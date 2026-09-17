# Full write-up: https://xfinlink.com/blog/recycled-ticker-price-history-risk-python
#
# Current S&P 500 members whose ticker symbol had an earlier holder. Each symbol's
# price rows are attributed to the company that held the symbol on the date, then the
# symbol-keyed splice is compared against the entity-resolved series on annualised
# volatility, maximum drawdown, and the return the join fabricates.

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

MIN_SESSIONS = 250      # one year under the symbol, per holder
MIN_COVERAGE = 0.90     # share of trading sessions a holder's slice must carry
ARTEFACT = 1.00         # single-session move inside one tenure

roster = xfl.index("sp500")
holder = {t: int(e) for t, e in zip(roster["ticker"], roster["entity_id"])}
symbols = sorted(holder)

# resolve() takes ten symbols per call and returns every company that has held each one.
def resolve_batch(batch):
    return xfl.resolve(batch)["data"]

history = {}
with ThreadPoolExecutor(8) as pool:
    for part in pool.map(resolve_batch, [symbols[i:i + 10] for i in range(0, len(symbols), 10)]):
        history.update(part)

cases = []
for sym, block in history.items():
    ents = block.get("entities", [])
    now = [e for e in ents if e["ticker_valid_to"] is None and e["entity_id"] == holder.get(sym)]
    if len(now) != 1:
        continue
    prior = [e for e in ents
             if e["entity_id"] != now[0]["entity_id"]
             and e["ticker_valid_to"]
             and e["ticker_valid_to"] < now[0]["ticker_valid_from"]
             and e["ticker_valid_to"] >= "1996-01-01"]
    if prior:
        cases.append({"sym": sym, "now": now[0], "prior": prior})

ids = sorted({c["now"]["entity_id"] for c in cases} | {p["entity_id"] for c in cases for p in c["prior"]})

def series(entity_id):
    return xfl.prices(entity_id=entity_id, start="1990-01-01",
                      fields=["close", "adj_close"], max_rows=40000)

with ThreadPoolExecutor(8) as pool:
    px = pd.concat([f for f in pool.map(series, ids) if not f.empty], ignore_index=True)
px["date"] = pd.to_datetime(px["date"])

# A symbol-keyed archive files a row under the symbol it traded as. Attribute each row
# to the company that held the symbol on that date.
def tenure(sym, ent):
    end = pd.Timestamp(ent["ticker_valid_to"]) if ent["ticker_valid_to"] else pd.Timestamp("2100-01-01")
    rows = px[(px["entity_id"] == ent["entity_id"]) & (px["ticker"] == sym)
              & (px["date"] >= pd.Timestamp(ent["ticker_valid_from"])) & (px["date"] <= end)]
    return rows.sort_values("date")

def risk(level):
    r = level.pct_change().dropna()
    growth = (1 + r).cumprod()
    return r.std() * np.sqrt(252), (growth / growth.cummax() - 1).min(), r.abs().max()

def covered(rows):
    """Share of the trading sessions between the first and last row that are present."""
    span = np.busday_count(rows["date"].min().date(), rows["date"].max().date()) + 1
    return len(rows) / span


table, thin, flagged, patchy = [], [], [], []
for case in cases:
    sym = case["sym"]
    old = [tenure(sym, e) for e in case["prior"]]
    old = [o for o in old if len(o)]
    new = tenure(sym, case["now"])
    if len(new) < MIN_SESSIONS or sum(len(o) for o in old) < MIN_SESSIONS:
        thin.append(sym)
        continue
    if min(covered(o) for o in old) < MIN_COVERAGE or covered(new) < MIN_COVERAGE:
        patchy.append(sym)
        continue
    worst = max([o.set_index("date")["adj_close"].pct_change().abs().max() for o in old]
                + [new.set_index("date")["adj_close"].pct_change().abs().max()])
    if worst > ARTEFACT:
        flagged.append(sym)
        continue
    spliced = pd.concat(old + [new]).sort_values("date")
    last_old = spliced.iloc[len(spliced) - len(new) - 1]
    first_new = new.iloc[0]
    vol_s, dd_s, big_s = risk(spliced.set_index("date")["adj_close"])
    vol_r, dd_r, _ = risk(new.set_index("date")["adj_close"])
    join = first_new["adj_close"] / last_old["adj_close"] - 1
    earlier = case["prior"][-1]
    table.append({
        "join_is_largest": bool(np.isclose(big_s, abs(join))),
        "sym": sym,
        # Identify the earlier holder by its symbol tenure, not by a name.
        "earlier_tenure": f"{earlier['ticker_valid_from']} to {earlier['ticker_valid_to']}",
        "spliced_from": spliced["date"].min().date(),
        "resolved_from": first_new["date"].date(),
        "foreign": (len(spliced) - len(new)) / len(spliced),
        "gap_yrs": (first_new["date"] - last_old["date"]).days / 365.25,
        "join": join,
        "vol_spliced": vol_s, "vol_resolved": vol_r,
        "dd_spliced": dd_s, "dd_resolved": dd_r,
    })

d = pd.DataFrame(table).sort_values("join", key=abs, ascending=False).reset_index(drop=True)

print("Current S&P 500 members whose symbol had an earlier holder quoted in the modern record")
print(f"symbols examined: {len(symbols)}   with an earlier holder: {len(cases)}   "
      f"retained: {len(d)}")
print(f"dropped: under {MIN_SESSIONS} sessions on one side {len(thin)}   "
      f"below {MIN_COVERAGE:.0%} session coverage {len(patchy)} {patchy}   "
      f"single-session artefact {len(flagged)} {flagged}")
print(f"returns from split-adjusted closes   price record opens 1996-01-02   "
      f"latest session {px['date'].max().date()}")
print()
head = (f"{'sym':<6}{'earlier holder held the symbol':<32}{'series from':>13}{'foreign':>9}"
        f"{'gap':>7}{'join':>10}{'vol':>16}{'max drawdown':>18}")
print(head)
print("-" * len(head))
for _, r in d.iterrows():
    print(f"{r['sym']:<6}{r['earlier_tenure']:<32}"
          f"{str(r['spliced_from']):>13}{r['foreign'] * 100:>8.1f}%{r['gap_yrs']:>6.0f}y"
          f"{r['join'] * 100:>9.0f}%"
          f"{r['vol_spliced'] * 100:>8.1f}%{r['vol_resolved'] * 100:>8.1f}%"
          f"{r['dd_spliced'] * 100:>9.1f}%{r['dd_resolved'] * 100:>9.1f}%")
print("-" * len(head))
print(f"{'':<38}{'':>13}{d['foreign'].median() * 100:>8.1f}%{'':>7}"
      f"{d['join'].abs().median() * 100:>9.0f}%"
      f"{d['vol_spliced'].median() * 100:>8.1f}%{d['vol_resolved'].median() * 100:>8.1f}%"
      f"{d['dd_spliced'].median() * 100:>9.1f}%{d['dd_resolved'].median() * 100:>9.1f}%   median")
print()
print(f"join exceeds a 20% single-session move: {int((d['join'].abs() > 0.20).sum())} of {len(d)}")
print(f"join is the largest single session in the spliced series: "
      f"{int(d['join_is_largest'].sum())} of {len(d)}")
print(f"volatility overstated by the splice: {int((d['vol_spliced'] > d['vol_resolved']).sum())} of {len(d)}   "
      f"median ratio {(d['vol_spliced'] / d['vol_resolved']).median():.2f}x")
print(f"maximum drawdown deepened by the splice: {int((d['dd_spliced'] < d['dd_resolved'] - 1e-9).sum())} of {len(d)}")

# ---- chart -------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1, 1.5]})

gm = [c for c in cases if c["sym"] == "GM"][0]
old_gm = pd.concat([tenure("GM", e) for e in gm["prior"]]).sort_values("date")
new_gm = tenure("GM", gm["now"])
ax1.semilogy(old_gm["date"], old_gm["adj_close"], color="#6b7280", lw=1.1,
             label="Earlier holder of the symbol")
ax1.semilogy(new_gm["date"], new_gm["adj_close"], color="#3b82f6", lw=1.1,
             label="Company holding the symbol today")
ax1.semilogy([old_gm["date"].iloc[-1], new_gm["date"].iloc[0]],
             [old_gm["adj_close"].iloc[-1], new_gm["adj_close"].iloc[0]],
             color="#f59e0b", lw=1.4, ls="--", label="Session created by the join")
ax1.set_title("One symbol, two companies: the price file for GM", loc="left", fontsize=11)
ax1.set_ylabel("Closing price (log scale)")
ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
ax1.tick_params(length=3)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

order = d.sort_values("join")
colors = ["#f59e0b" if v < 0 else "#3b82f6" for v in order["join"]]
ax2.barh(order["sym"], order["join"] * 100, color=colors, height=0.68)
ax2.set_xscale("symlog", linthresh=100)
ax2.axvline(0, color="#e0e0e0", lw=0.8)
ax2.set_xlabel("Return of the single session the join creates (%, symmetric log scale)")
ax2.set_title("Every retained symbol: the move that is not a market move",
              loc="left", fontsize=11)
ax2.tick_params(length=3, labelsize=8.5)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("recycled-ticker-price-history-risk-python.png", dpi=150,
            facecolor="#0a0a0a")
