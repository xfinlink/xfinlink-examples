# Full write-up: https://xfinlink.com/blog/ticker-recycling-sp500-symbol-decay-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

VINTAGES = ["1996-12-31", "2000-12-31", "2005-12-31", "2010-12-31",
            "2015-12-31", "2020-12-31"]
TODAY_START, TODAY_END = "2026-06-01", "2026-08-14"


def price_batch(start, end, ids=None, syms=None):
    """Fetch closes in batches of 50 and stack the results."""
    keys = ids if ids is not None else syms
    parts = []
    for i in range(0, len(keys), 50):
        chunk = keys[i:i + 50]
        if ids is not None:
            parts.append(xfl.prices(entity_id=chunk, start=start, end=end,
                                    fields=["close"], max_rows=50000))
        else:
            parts.append(xfl.prices(chunk, start=start, end=end,
                                    fields=["close"], max_rows=50000))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --- 1. The symbol each member actually traded under, per vintage ----------
vintage_frames = {}
for as_of in VINTAGES:
    roster = xfl.index("sp500", as_of=as_of).dropna(subset=["entity_id"])
    roster["entity_id"] = roster["entity_id"].astype(int)
    ids = roster["entity_id"].tolist()
    window_start = (pd.Timestamp(as_of) - pd.Timedelta(days=20)).date().isoformat()
    px = price_batch(window_start, as_of, ids=ids)
    then = (px.groupby("entity_id").agg(symbol=("ticker", "last")).reset_index()
              .merge(roster[["entity_id", "entity_name"]], on="entity_id",
                     how="left")
              .drop_duplicates("symbol"))
    vintage_frames[as_of] = then
    print(f"{as_of[:4]} roster: {len(roster):3d} members, "
          f"{len(then):3d} with a traded symbol on the tape")

# --- 2. Which company answers to that symbol today -------------------------
all_syms = sorted({s for f in vintage_frames.values() for s in f["symbol"]})
now = price_batch(TODAY_START, TODAY_END, syms=all_syms)
cur = (now.groupby("ticker")
          .agg(now_id=("entity_id", "last"), now_name=("entity_name", "last"),
               now_sector=("gics_sector", "last"))
          .reset_index())
print(f"\nDistinct symbols tested: {len(all_syms)}   "
      f"answering in {TODAY_START[:7]}-{TODAY_END[:7]}: {len(cur)}")

# --- 3. Classify every symbol ---------------------------------------------
rows = []
for as_of, then in vintage_frames.items():
    j = then.merge(cur, left_on="symbol", right_on="ticker", how="left")
    j["outcome"] = np.where(j["now_id"].isna(), "no data",
                            np.where(j["now_id"] == j["entity_id"],
                                     "same company", "different company"))
    j["vintage"] = as_of[:4]
    rows.append(j)
res = pd.concat(rows, ignore_index=True)

share = (res.pivot_table(index="vintage", columns="outcome", values="symbol",
                         aggfunc="count").fillna(0)
         .reindex(columns=["same company", "different company", "no data"],
                  fill_value=0))
pct = share.div(share.sum(axis=1), axis=0) * 100

print("\nWhat a saved ticker list returns when replayed in August 2026")
print(f"{'list from':>10}  {'symbols':>7}  {'same co':>16}  "
      f"{'different co':>16}  {'no data':>16}")
for v in share.index:
    tot = int(share.loc[v].sum())
    print(f"{v:>10}  {tot:>7}  "
          + "  ".join(f"{int(share.loc[v, c]):5d} ({pct.loc[v, c]:4.1f}%)"
                      for c in share.columns))

# --- 4. The 2005 list in detail: who took the symbol over ------------------
sw = res[(res["vintage"] == "2005") & (res["outcome"] == "different company")]
sw = sw.sort_values("symbol")
print(f"\n2005 symbols now answering for a different company: {len(sw)}")
print(f"  {'symbol':<7}{'held in 2005 by (entity, name today)':<38}"
      f"answering today")
for _, r in sw.iterrows():
    print(f"  {r['symbol']:<7}{r['entity_name'][:36]:<38}{r['now_name'][:36]}")

# --- 5. Would a 2006-2016 pull have noticed? -------------------------------
mid = price_batch("2010-06-01", "2010-06-30", syms=sw["symbol"].tolist())
silent = mid.groupby("ticker")["entity_id"].last() if not mid.empty else pd.Series(dtype=int)
sw = sw.assign(traded_2010=sw["symbol"].map(silent).notna())
n_silent = int(sw["traded_2010"].sum())
print(f"\nOf those {len(sw)} symbols, {n_silent} were already quoted in June 2010 "
      f"under the new owner.")
print(f"A 2006-2016 backtest keyed on the 2005 symbol list would have returned "
      f"prices for {n_silent} companies it never meant to hold, "
      f"and nothing at all for {int(share.loc['2005', 'no data'])} it did.")

# --- 6. Chart -------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#2e2e2e", "font.size": 11,
})
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"same company": "#3b82f6", "different company": "#f59e0b",
          "no data": "#3f3f46"}
left = np.zeros(len(pct))
ypos = np.arange(len(pct))
for col in pct.columns:
    ax.barh(ypos, pct[col].values, left=left, color=colors[col],
            label=col, height=0.62)
    for y, (v, l) in enumerate(zip(pct[col].values, left)):
        if v >= 4:
            ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                    color="#0a0a0a" if col != "no data" else "#e0e0e0",
                    fontsize=10, fontweight="bold")
    left += pct[col].values
ax.set_yticks(ypos)
ax.set_yticklabels([f"{v} list" for v in pct.index])
ax.set_xlim(0, 100)
ax.set_xlabel("Share of symbols, replayed in August 2026 (%)")
ax.set_title("What a saved S&P 500 ticker list returns years later")
ax.invert_yaxis()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
          frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("ticker-recycling-sp500-symbol-decay-python.png", dpi=150,
            facecolor="#0a0a0a")
