# Full write-up: https://xfinlink.com/blog/dividend-yield-payout-price-decomposition-python
"""Split the five-year change in S&P 500 dividend yields into a payout leg and
a price leg, then rank the cross-section by 2025 yield.

Dividend yield is dividends per share divided by price, so any change in yield
is the dividend growth rate minus the price return:

    log(y_2025 / y_2020) = log(DPS_2025 / DPS_2020) - log(P_2025 / P_2020)

Both inputs are unit-free or split-adjusted, so stock splits cancel out.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2020-12-31", "2025-12-31"
YIELD_FLOOR = 0.01
RECON_TOL = 0.25
CHUNK = 50

# Entity 9424: the 2020 anchor quote could not be confirmed for this company, so
# the name drops rather than anchoring a five-year window on an unconfirmed price.
UNCONFIRMED_ANCHOR = {9424}


def chunked(fetch, tickers, size=CHUNK):
    """Run fetch() over ticker batches with a short backoff between attempts."""
    frames = []
    for i in range(0, len(tickers), size):
        for attempt in range(5):
            try:
                frames.append(fetch(tickers[i:i + size]))
                break
            except xfl.XfinlinkError:
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"batch starting at {i} did not return")
    return pd.concat(frames, ignore_index=True)


universe = sorted(xfl.index("sp500")["ticker"].dropna().unique().tolist())
print(f"S&P 500 constituents: {len(universe)}")


def annual_yield(date):
    year = date[:4]
    m = chunked(lambda b: xfl.metrics(b, period_type="annual", start=f"{year}-01-01",
                                      end=f"{year}-12-31",
                                      fields=["dividend_yield"]), universe)
    return m[m["period_end"] == date].drop_duplicates("entity_id")


def close_on(date):
    start = (pd.Timestamp(date) - pd.Timedelta(days=16)).date().isoformat()
    p = chunked(lambda b: xfl.prices(b, start=start, end=date,
                                     fields=["close", "adj_close"]), universe)
    return p.sort_values("date").drop_duplicates("entity_id", keep="last")


y_end, y_start = annual_yield(END), annual_yield(START)
p_end, p_start = close_on(END), close_on(START)

# entity_id, not ticker: a symbol can change hands between the two dates
d = (y_end[["entity_id", "entity_name", "dividend_yield"]].rename(columns={"dividend_yield": "y1"})
     .merge(y_start[["entity_id", "dividend_yield"]].rename(columns={"dividend_yield": "y0"}), on="entity_id")
     .merge(p_end[["entity_id", "ticker", "gics_sector", "close", "adj_close"]].rename(
         columns={"close": "c1", "adj_close": "a1"}), on="entity_id")
     .merge(p_start[["entity_id", "close", "adj_close"]].rename(
         columns={"close": "c0", "adj_close": "a0"}), on="entity_id"))
print(f"December fiscal year end with both snapshots: {len(d)}")

d = d[(d["y0"] >= YIELD_FLOOR) & (d["y1"] >= YIELD_FLOOR)].copy()
print(f"Yielding at least {YIELD_FLOOR:.0%} at both ends: {len(d)}")


# ── sample hygiene, applied identically at both anchor dates ─────────────────
def dividends_from_quarters(year, tickers):
    """Rebuild a fiscal year's dividend per share from the quarterly record."""
    q = chunked(lambda b: xfl.fundamentals(b, period_type="quarterly",
                                           start=f"{year}-01-01", end=f"{year}-12-31",
                                           fields=["dividends_per_share"]), tickers)
    q = q.dropna(subset=["dividends_per_share"]).drop_duplicates(
        ["entity_id", "fiscal_year", "fiscal_period"])
    w = (q.pivot_table(index="entity_id", columns="fiscal_period",
                       values="dividends_per_share")
          .reindex(columns=["Q1", "Q2", "Q3", "Q4"]).dropna())
    # some filers carry the quarterly column year to date rather than standalone;
    # differencing the running total recovers the same annual figure either way
    running = (w["Q2"] > w["Q1"] * 1.6) & (w["Q3"] > w["Q2"] * 1.4)
    return pd.Series(np.where(running, 2 * w["Q3"] - w["Q2"], w.sum(axis=1)),
                     index=w.index, name="rebuilt")


before, tested = len(d), pd.Series(False, index=d.index)
for year, y, c in [(int(START[:4]), "y0", "c0"), (int(END[:4]), "y1", "c1")]:
    rebuilt = dividends_from_quarters(year, sorted(d["ticker"].tolist()))
    d = d.join(rebuilt, on="entity_id")
    gap = (d[y] * d[c] / d["rebuilt"] - 1).abs()
    tested = tested.reindex(d.index, fill_value=False) | d["rebuilt"].notna()
    d = d[(gap <= RECON_TOL) | d["rebuilt"].isna()].drop(columns="rebuilt")
print(f"Dropped, annual and quarterly dividend records disagree: {before - len(d)}")
print(f"Dividend record reconciled for {int(tested.reindex(d.index, fill_value=False).sum())} "
      f"of the {len(d)} remaining; the rest carry no full quarterly record to test against")

# the symbol must already have belonged to this company at the 2020 anchor
held = chunked(lambda b: pd.DataFrame(
    [{"entity_id": e["entity_id"], "held_from": e.get("ticker_valid_from")}
     for v in xfl.resolve(b)["data"].values() for e in v.get("entities", [])]),
    sorted(d["ticker"].tolist()), size=10)
held = (held.assign(held_from=pd.to_datetime(held["held_from"]))
            .groupby("entity_id", as_index=False)["held_from"].min())
before = len(d)
d = d.merge(held, on="entity_id", how="left")
d = d[~(d["held_from"] > START) & ~d["entity_id"].isin(UNCONFIRMED_ANCHOR)].copy()
print(f"Dropped, anchor date not confirmable for the same company: {before - len(d)}")
print(f"Final sample: {len(d)}")

# adj_close is split-adjusted, so the price ratio is gap-free across splits;
# the yield ratio is unit-free, so the payout leg falls out of the identity.
d["price_leg"] = -np.log(d["a1"] / d["a0"])
d["yield_leg"] = np.log(d["y1"] / d["y0"])
d["payout_leg"] = d["yield_leg"] - d["price_leg"]

# cross-check: yield times price should reproduce the reported dividend per share
rep = chunked(lambda b: xfl.fundamentals(b, period_type="annual", start=END, end=END,
                                         fields=["dividends_per_share"]),
              sorted(d["ticker"].tolist()))
chk = d.merge(rep[["entity_id", "dividends_per_share"]], on="entity_id").dropna(
    subset=["dividends_per_share"])
chk = chk[chk["dividends_per_share"] > 0]
err = (chk["y1"] * chk["c1"] / chk["dividends_per_share"] - 1).abs()
print(f"Dividend-per-share cross-check: {len(chk)} names, "
      f"{(err < 0.02).mean():.1%} agree within 2%")

# share of the cross-sectional variance in yield changes carried by each leg
cov = np.cov(np.vstack([d["yield_leg"], d["payout_leg"], d["price_leg"]]))
payout_share, price_share = cov[0, 1] / cov[0, 0], cov[0, 2] / cov[0, 0]
rho_price = spearmanr(d["y1"], -d["price_leg"])[0]
rho_payout = spearmanr(d["y1"], d["payout_leg"])[0]

d["quintile"] = pd.qcut(d["y1"], 5, labels=[1, 2, 3, 4, 5])
pct = lambda s: np.expm1(s.median()) * 100
table = d.groupby("quintile", observed=True).agg(
    n=("entity_id", "size"),
    yield_2020=("y0", lambda s: s.median() * 100),
    yield_2025=("y1", lambda s: s.median() * 100),
    dps_growth=("payout_leg", pct),
    price_change=("price_leg", lambda s: np.expm1(-s.median()) * 100),
)

print("\nS&P 500 dividend payers, five-year yield decomposition (medians)")
print("quintile  n   yield 2020  yield 2025  DPS growth  price change")
for q, r in table.iterrows():
    print(f"    {q}    {int(r['n']):3d}   {r['yield_2020']:8.1f}%   {r['yield_2025']:8.1f}%"
          f"   {r['dps_growth']:8.1f}%   {r['price_change']:9.1f}%")

print(f"\nPrice leg carries {price_share:.1%} of the cross-sectional variance in yield changes")
print(f"Payout leg carries {payout_share:.1%}")
print(f"Spearman rank correlation, 2025 yield vs five-year price change:  {rho_price:+.3f}")
print(f"Spearman rank correlation, 2025 yield vs five-year DPS growth:    {rho_payout:+.3f}")

top = d[d["quintile"] == 5]
mix = pd.DataFrame({"top_quintile": top["gics_sector"].value_counts(),
                    "in_sample": d["gics_sector"].value_counts()}).fillna(0).astype(int)
mix["share"] = mix["top_quintile"] / mix["in_sample"] * 100
print("\nSector, share of its dividend payers landing in the top yield quintile")
for s, r in mix.sort_values("share", ascending=False).iterrows():
    print(f"  {s:<24} {int(r['top_quintile']):2d} of {int(r['in_sample']):2d}   {r['share']:5.0f}%")

# ── chart ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 11,
})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(table))
ax.bar(x - 0.2, table["dps_growth"], 0.4, color="#9ca3af", label="Dividend per share")
ax.bar(x + 0.2, table["price_change"], 0.4, color="#3b82f6", label="Share price")
ax.axhline(0, color="#3a3a3a", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f"Q{q}\n{r.yield_2025:.1f}% yield" for q, r in table.iterrows()])
ax.set_ylabel("Median five-year change (%)")
ax.set_xlabel("2025 dividend yield quintile, lowest to highest")
ax.set_title("Where high dividend yields come from, S&P 500 2020-2025")
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("dividend-yield-payout-price-decomposition-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart written")
