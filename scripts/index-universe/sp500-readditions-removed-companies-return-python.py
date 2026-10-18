# Full write-up: https://xfinlink.com/blog/sp500-readditions-removed-companies-return-python
#
# How many companies removed from the S&P 500 came back, how long were they out,
# and what does a one-span-per-company membership table get wrong as a result?

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose

pages = [xfl.index_events("sp500", start="1925-12-31")]
while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
    pages.append(xfl.index_events("sp500", start="1925-12-31", offset=1000 * len(pages)))
ev = pd.concat(pages, ignore_index=True).dropna(subset=["entity_id"]).astype({"entity_id": int})
ev["effective_date"] = pd.to_datetime(ev["effective_date"])
ev = ev.sort_values(["entity_id", "effective_date"], kind="stable")

# Rebuild membership spells per security; founding members open before their first event
spells = []
for eid, g in ev.groupby("entity_id"):
    start = None if g["event_type"].iloc[0] == "added" else pd.Timestamp("1925-12-31")
    tick_in = g["ticker"].iloc[0]
    for r in g.itertuples():
        if r.event_type == "added":
            start, tick_in = r.effective_date, r.ticker
        elif start is not None:
            spells.append((eid, r.entity_name, start, r.effective_date, tick_in, r.ticker))
            start = None
    if start is not None:
        spells.append((eid, g["entity_name"].iloc[-1], start, pd.NaT, tick_in, None))
spells = pd.DataFrame(spells, columns=["entity_id", "name", "added", "removed", "ticker_in", "ticker_out"])
# a stay under 30 days is a placement (a spin-off or merger vehicle), not a membership
spells = spells[~((spells["removed"] - spells["added"]).dt.days < 30)]

# Each gap between two memberships of the same security is a removal followed by a return
spells["next_added"] = spells.groupby("entity_id")["added"].shift(-1)
spells["ticker_back"] = spells.groupby("entity_id")["ticker_in"].shift(-1)
gaps = spells.dropna(subset=["next_added"]).copy()
gaps["years_out"] = (gaps["next_added"] - gaps["removed"]).dt.days / 365.25

exits = spells[spells["removed"] >= "1990-01-01"]
back = gaps[gaps["removed"] >= "1990-01-01"].sort_values("removed")
renamed = back[back["ticker_out"] != back["ticker_back"]]
print(f"Removals since 1990: {len(exits)}; later re-added as the same security: {len(back)} ({len(back) / len(exits):.1%})")
print(f"Years outside: median {back['years_out'].median():.1f}, shortest {back['years_out'].min():.1f}, "
      f"longest {back['years_out'].max():.1f}; over 5 years: {(back['years_out'] > 5).sum()}")
print(f"Returned under a different ticker: {len(renamed)}  "
      + ", ".join(f"{a}->{b}" for a, b in zip(renamed["ticker_out"], renamed["ticker_back"])))
print("Returns by decade:", back.groupby(back["next_added"].dt.year // 10 * 10).size().to_dict())

# A table with one first-added / last-removed span per security counts each of these gaps as membership
days = pd.bdate_range("1990-01-01", "2025-12-31")
wrong = pd.Series(0, index=days)
for g in back.itertuples():
    wrong[(days > g.removed) & (days < g.next_added)] += 1
print(f"One-span table, 1990-2025: {wrong.mean():.1f} non-members counted per weekday, "
      f"{(wrong > 0).mean():.0%} of weekdays affected, peak {wrong.max()} on {wrong.idxmax():%Y-%m-%d}")

print("\nRemoved since 1990 and re-added:")
out = back.assign(left=back["removed"].dt.strftime("%Y-%m-%d"), returned=back["next_added"].dt.strftime("%Y-%m-%d"))
print(out[["name", "ticker_out", "ticker_back", "left", "returned", "years_out"]].round(1).to_string(index=False))

# Chart: each line is one security's time outside the index
fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
for i, g in enumerate(back.itertuples()):
    ax.plot([g.removed, g.next_added], [i, i], color="#3b82f6", lw=2.2, solid_capstyle="round")
    ax.text(g.next_added + pd.Timedelta(days=90), i, g.ticker_back, color="#e0e0e0", va="center", fontsize=7.5)
ax.set_yticks([])
ax.invert_yaxis()
ax.set_xlabel("Removed (left end) to re-added (right end)", color="#e0e0e0")
ax.set_title("S&P 500 Members Removed Since 1990 That Came Back", color="#e0e0e0", fontsize=13)
ax.tick_params(colors="#e0e0e0")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#555555")
plt.tight_layout()
plt.savefig("sp500-readditions-removed-companies-return-python.png", dpi=150, facecolor="#0a0a0a")
