# Full write-up: https://xfinlink.com/blog/sp500-companies-nasdaq-vs-nyse-history-python
#
# How many S&P 500 companies are on Nasdaq, and how has that changed since 1975?
# Each year-end roster is rebuilt point in time, and each member gets the listing
# exchange it had on that date. The result is compared with the shortcut of taking
# the end-2024 list and its end-2024 exchanges and projecting them backwards.
#
# Counting unit: one index line (one share class). GOOG/GOOGL, FOX/FOXA and NWS/NWSA
# count twice; both classes always share a venue, and the script prints a
# company-level check for 1999 and 2024.

import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST, START, END = 1962, 1975, 2024   # coverage check from 1962; analysis 1975-2024
CHUNK = 150                            # security ids per prices call


def fetch():
    """Year-end rosters and each member's listing exchange at each year-end."""
    # Trading calendar: IBM (security id 10001962) traded on every session in the window
    cal = xfl.prices(entity_id=10001962, start=f"{FIRST}-01-01", end=f"{END}-12-31",
                     fields=["close"], max_rows=100_000)
    last_days = cal.groupby(cal["date"].dt.year)["date"].max()

    rosters = []
    for d in last_days:
        r = xfl.index("sp500", as_of=d.date().isoformat())
        r["as_of"] = d.date().isoformat()
        rosters.append(r)
    rosters = pd.concat(rosters, ignore_index=True)

    # Yearly bars: exchange_code is the listing exchange on the bucket's LAST trading
    # day, which for a year-end member is the roster date. `date` is the bucket's first day.
    ids = sorted(set(rosters["entity_id"].astype(int)))
    venue = pd.concat([xfl.prices(entity_id=ids[i:i + CHUNK], start=f"{FIRST}-01-01",
                                  end=f"{END}-12-31", interval="1y",
                                  fields=["exchange_code"], max_rows=500_000)
                       for i in range(0, len(ids), CHUNK)], ignore_index=True)
    return rosters, venue


def analyze(rosters, venue):
    venue = venue.assign(year=pd.to_datetime(venue["date"]).dt.year)
    ven = venue.set_index(["entity_id", "year"])["exchange_code"]
    m = rosters.assign(year=pd.to_datetime(rosters["as_of"]).dt.year)
    m["venue"] = ven.reindex(pd.MultiIndex.from_frame(m[["entity_id", "year"]])).values

    out = {"coverage": m.groupby("year")["venue"].agg(lines="size", no_venue=lambda s: s.isna().sum())}
    m = m[m["year"] >= START]
    known = m.dropna(subset=["venue"])

    # (a) point in time: the roster on each date, each member's exchange on that date
    pit = known.groupby("year")["venue"].agg(
        lines="size", nasdaq=lambda s: (s == "NASDAQ").sum(), nyse=lambda s: (s == "NYSE").sum())
    pit["pct"] = pit["nasdaq"] / pit["lines"] * 100

    # (b) the shortcut: the end-2024 list, every member that traded that year,
    #     labelled with its end-2024 exchange; (b') the same names with their exchange on the date
    last = m[m["year"] == END].set_index("entity_id")["venue"]
    traded = venue[venue["entity_id"].isin(last.index)]
    short = traded.groupby("year").apply(lambda g: pd.Series({
        "lines_list2024": len(g),
        "pct_today_label": (last.reindex(g["entity_id"]) == "NASDAQ").mean() * 100,
        "pct_label_on_date": (g["exchange_code"] == "NASDAQ").mean() * 100}))
    out["series"] = pit.join(short).loc[START:END]

    # Year-to-year accounting of the point-in-time Nasdaq count
    m = m.sort_values(["entity_id", "year"])
    rows = []
    for y in range(START + 1, END + 1):
        a, b = m[m["year"] == y], m[m["year"] == y - 1]
        both = a.merge(b, on="entity_id", suffixes=("", "_prev"))
        both = both.dropna(subset=["venue", "venue_prev"])
        moved = both[both["venue"] != both["venue_prev"]]
        rows.append({"year": y,
                     "added_nasdaq": (a.loc[~a["entity_id"].isin(b["entity_id"]), "venue"] == "NASDAQ").sum(),
                     "removed_nasdaq": (b.loc[~b["entity_id"].isin(a["entity_id"]), "venue"] == "NASDAQ").sum(),
                     "moved_to_nasdaq": (moved["venue"] == "NASDAQ").sum(),
                     "moved_from_nasdaq": (moved["venue_prev"] == "NASDAQ").sum()})
        for _, s in moved.iterrows():
            out.setdefault("moves", []).append((y, s["ticker"], s["entity_name"], s["venue_prev"], s["venue"]))
    out["flows"] = pd.DataFrame(rows).set_index("year")
    out["moves"] = pd.DataFrame(out["moves"], columns=["year", "ticker", "name", "from", "to"])
    out["m"] = m
    out["venue"] = venue
    return out


def report(out):
    s, f, mv, m = out["series"], out["flows"], out["moves"], out["m"]
    cov = out["coverage"]
    last24 = m[m["year"] == END].set_index("entity_id")["venue"]
    print(f"Member lines {FIRST}-{END}: {cov['lines'].sum():,}; with a listing exchange on the "
          f"roster date: {cov['lines'].sum() - cov['no_venue'].sum():,}")
    print("Lines without one, by year:", cov.loc[cov["no_venue"] > 0, "no_venue"].to_dict())
    print(f"Point-in-time Nasdaq lines before 1976: {int(s.loc[START, 'nasdaq'])} of {int(s.loc[START, 'lines'])} in {START}")

    print("\nNasdaq share of S&P 500 lines at year-end (%)")
    print(f"{'year':>4} {'nasdaq':>6} {'lines':>5} {'point-in-time':>13} | {'end-2024 list: names':>20} "
          f"{'exchange on date':>16} {'end-2024 exchange':>17}")
    for y in [1975, 1980, 1985, 1990, 1995, 1998, 1999, 2000, 2001, 2005, 2008, 2010, 2015, 2020, 2024]:
        r = s.loc[y]
        print(f"{y:>4} {int(r['nasdaq']):>6} {int(r['lines']):>5} {r['pct']:>13.1f} | {int(r['lines_list2024']):>20} "
              f"{r['pct_label_on_date']:>16.1f} {r['pct_today_label']:>17.1f}")

    print("\nGap between the shortcut and the point-in-time share (points)")
    for y in [1990, 1999, 2010]:
        r = s.loc[y]
        print(f"  {y}: total {r['pct_today_label'] - r['pct']:+.1f} = membership "
              f"{r['pct_label_on_date'] - r['pct']:+.1f} + exchange label {r['pct_today_label'] - r['pct_label_on_date']:+.1f}")
    gap = s["pct_today_label"] - s["pct"]
    for y in [1999, 2000]:
        v = out["venue"]
        names = v.loc[(v["year"] == y) & v["entity_id"].isin(last24.index), "entity_id"]
        shortcut = set(names[last24.reindex(names).values == "NASDAQ"])
        true = set(m.loc[(m["year"] == y) & (m["venue"] == "NASDAQ"), "entity_id"])
        print(f"  {y}: shortcut Nasdaq basket {len(shortcut)} names; Nasdaq-listed members that year "
              f"{len(true)}; in both {len(shortcut & true)}")
    gone = m[(m["year"] == 2000) & (m["venue"] == "NASDAQ") & ~m["entity_id"].isin(last24.index)]
    print(f"  Nasdaq lines at end-2000 not on the end-{END} list: {len(gone)} of "
          f"{int(s.loc[2000, 'nasdaq'])}, e.g. {', '.join(t for t in ['SUNW', 'WCOM', 'YHOO', 'PSFT', 'PALM'] if t in set(gone['ticker']))}")
    print(f"  largest gap: {gap.idxmax()} ({gap.max():.1f} points); mean 1975-2023: {gap.loc[:2023].mean():.1f}")

    print("\nWhere the point-in-time Nasdaq count came from")
    for a, b in [(1976, 1997), (1998, 2001), (2002, 2007), (2008, 2024)]:
        g = f.loc[a:b].sum()
        start, end = int(s.loc[a - 1, "nasdaq"]), int(s.loc[b, "nasdaq"])
        print(f"  {a}-{b}: {start:>3} -> {end:>3}  added +{g['added_nasdaq']:<3} removed -{g['removed_nasdaq']:<3} "
              f"moved in +{g['moved_to_nasdaq']:<3} moved out -{g['moved_from_nasdaq']}")

    print(f"\nExchange moves by members between consecutive year-ends {START}-{END}: {len(mv)}")
    print(mv.groupby(["from", "to"]).size().to_string())
    mv["period"] = pd.cut(mv["year"], [1975, 2004, 2024], labels=["1976-2004", "2005-2024"])
    print(mv[mv["to"].isin(["NASDAQ", "NYSE"])].groupby(["period", "from", "to"], observed=True).size().to_string())
    print(mv[mv["ticker"].isin(["ORCL", "PEP", "TXN"])].to_string(index=False))

    print("\nCompany-level check (dual-class lines merged by company name)")
    for y in [1999, END]:
        c = m[m["year"] == y].drop_duplicates("entity_name")
        print(f"  {y}: {(c['venue'] == 'NASDAQ').sum()} of {len(c)} companies on Nasdaq, "
              f"{(c['venue'] == 'NASDAQ').mean() * 100:.1f}%")
    last = m[m["year"] == END]["venue"].value_counts()
    print(f"  {END} lines by exchange: {last.to_dict()}")


def chart(out, path="sp500-companies-nasdaq-vs-nyse-history-python.png"):
    s = out["series"]
    plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                         "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                         "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0", "axes.edgecolor": "#555555"})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s.index, s["pct_today_label"], color="#f59e0b", lw=2, ls="--",
            label="End-2024 list, end-2024 exchange (shortcut)")
    ax.plot(s.index, s["pct_label_on_date"], color="#9ca3af", lw=1.8, ls=":",
            label="End-2024 list, exchange on the date")
    ax.plot(s.index, s["pct"], color="#3b82f6", lw=2.6, label="Point-in-time roster, exchange on the date")
    for y in [1999, 2024]:
        ax.annotate(f"{s.loc[y, 'pct']:.1f}%", (y, s.loc[y, "pct"]), textcoords="offset points",
                    xytext=(-34 if y == 2024 else 6, -14), color="#3b82f6", fontsize=10)
    ax.annotate(f"{s.loc[1999, 'pct_today_label']:.1f}%", (1999, s.loc[1999, "pct_today_label"]),
                textcoords="offset points", xytext=(-10, 8), color="#f59e0b", fontsize=10)
    ax.set_title("How Many S&P 500 Companies Are on Nasdaq? Year-End 1975-2024", color="#e0e0e0")
    ax.set_xlabel("Year-end")
    ax.set_ylabel("Nasdaq-listed share of index lines (%)")
    ax.set_ylim(0, 35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    rosters, venue = fetch()
    out = analyze(rosters, venue)
    report(out)
    chart(out)
