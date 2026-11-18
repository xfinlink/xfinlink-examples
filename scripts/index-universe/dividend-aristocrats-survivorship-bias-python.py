# Full write-up: https://xfinlink.com/blog/dividend-aristocrats-survivorship-bias-python
#
# Dividend Aristocrats backtest, 2000-2024: how much does survivorship bias add?
# A Dividend Aristocrat is an S&P 500 member that has raised its dividend for at
# least 25 consecutive years. Streaks are rebuilt from cash distributions since
# 1962, restated for splits and summed by calendar year of the ex-date. Special
# dividends do not count; a payment that adds a special to a regular dividend
# counts as the previous regular payment. Each December from 1999 to 2023 the
# 25-year rule is applied to the S&P 500 roster in force that day (point-in-time)
# and the passers are held for the next calendar year, equal weight, bought and
# held. The survivor version holds the end-2024 list for all 25 years. A holding
# that stops trading takes its recorded delisting return and sits in cash for the
# rest of the year. This is an aristocrat-style screen, not the official index.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SPECIAL = "special_dividends"
ymd = lambda d: d.strftime("%Y-%m-%d")

# S&P 500 rosters on the last business day of each December, 1999-2024
forms = pd.date_range("1999-12-01", "2024-12-31", freq="BYE")
rosters = {d.year: xfl.index("sp500", as_of=ymd(d)) for d in forms}
member = {y: set(r["entity_id"].dropna().astype(int)) for y, r in rosters.items()}
ids = sorted(set().union(*member.values()))

# Monthly bars since 1962 for every company on any of those rosters
px = pd.concat(xfl.prices(entity_id=ids[i:i + 50], start="1962-01-01", end="2024-12-31", interval="1mo",
                          fields=["close", "adj_close", "return_daily", "dividend", "split_ratio"],
                          max_rows=250_000) for i in range(0, len(ids), 50)).reset_index(drop=True)
px["m"] = px["date"].dt.to_period("M")
px["year"] = px["date"].dt.year
px = px.drop_duplicates(["entity_id", "m"]).sort_values(["entity_id", "m"]).reset_index(drop=True)
px["split"] = px["split_ratio"].fillna(1.0)
# Splits after each month, so every dividend is restated to today's share basis
px["later"] = px.groupby("entity_id")["split"].transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
kinds = px["dividend_type"].fillna("")

# A month with both a split and a dividend, or a special and another payment, is read day by day
amb = px[((px["split"] != 1) & px["dividend"].notna()) | (kinds.str.contains(SPECIAL) & kinds.str.contains(","))]
pay = [px.loc[px["dividend"].notna() & ~px.index.isin(amb.index)]
       .assign(amount=lambda x: x["dividend"] / x["later"])]
for y, g in amb.groupby("year"):
    d = xfl.prices(entity_id=sorted(g["entity_id"].unique()), start=f"{y}-01-01", end=f"{y}-12-31",
                   fields=["dividend", "split_ratio"], max_rows=100_000)
    d["m"] = d["date"].dt.to_period("M")
    d = d.merge(g[["entity_id", "m", "later"]], on=["entity_id", "m"]).sort_values(["entity_id", "date"])
    within = d.groupby(["entity_id", "m"])["split_ratio"].transform(          # splits later that month
        lambda s: s.fillna(1.0)[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
    pay.append(d.assign(amount=d["dividend"] / within / d["later"]).dropna(subset=["dividend"]))
pay = pd.concat(pay)[["entity_id", "date", "dividend", "dividend_type", "amount"]].sort_values(["entity_id", "date"])
# Four payments classed from the companies' own announcements: regular cash per share in that
# month's payments (the rest is special)
CLASSED = {
    (10013874, "2007-04"): 0.0,   # BF-B 1.6533 special distribution, 8-K/10-K FY2007:
    # https://www.sec.gov/Archives/edgar/data/0000014693/000001469307000066/form10k2007.txt
    (10010889, "2015-12"): 0.46,  # CINF 0.46 special beside the 0.46 regular, 8-K 2015-11-20:
    # https://www.sec.gov/Archives/edgar/data/0000020286/000114420415067177/v425270_ex99-1.htm
    (10011063, "2014-11"): 0.85,  # CTAS 1.70 = 0.85 annual + 0.85 special, 10-Q for 2014-11-30:
    # https://www.sec.gov/Archives/edgar/data/0000723254/000072325415000002/ctas-2014x1130x10xq.htm
    (10011063, "2020-11"): 3.51,  # CTAS 3.51 = 2.81 annual + 0.70 first quarterly, both regular:
    # https://www.businesswire.com/news/home/20201027005701/en/
}
key = list(zip(pay["entity_id"], pay["date"].dt.strftime("%Y-%m")))
regular = pd.Series([CLASSED.get(k) for k in key], index=pay.index, dtype=float)
pay["amount"] *= (regular / pay["dividend"]).fillna(1.0)
pay["dividend_type"] = pay["dividend_type"].where(regular.isna(), "cash_dividend")
pay = pay[(pay["dividend_type"] != SPECIAL) & (pay["amount"] > 0)]   # special dividends do not count
# A payment that adds a special dividend to a regular one counts as the previous regular payment
both = pay["dividend_type"].str.contains(SPECIAL)
last_regular = pay["amount"].mask(both).groupby(pay["entity_id"]).ffill()
pay["amount"] = np.where(both, np.fmin(pay["amount"], last_regular), pay["amount"])

# Annual regular dividend per share, and the streak of consecutive yearly increases
years = range(1962, 2025)
dec = px[px["date"].dt.month == 12].assign(on=True).pivot(index="entity_id", columns="year", values="on")
dec = dec.reindex(columns=years).notna()                       # traded through December
pay["year"] = pay["date"].dt.year
dps = pay.pivot_table(index="entity_id", columns="year", values="amount", aggfunc="sum")
n = pay.pivot_table(index="entity_id", columns="year", values="amount", aggfunc="count")
dps = dps.reindex(index=dec.index, columns=years).fillna(0).where(dec)
n = n.reindex(index=dec.index, columns=years).fillna(0).where(dec)
full = dec & dec.shift(axis=1, fill_value=False)               # on record the whole calendar year
prev, mean = dps.shift(axis=1), dps / n
# A raise: more paid than the year before. When the number of payments changed because an ex-date
# crossed the year end (one payment more or fewer, or one brought forward a year, as in December
# 2012), the average payment must rise instead.
n1, usual = n.shift(axis=1), n.T.rolling(5, min_periods=3).median().T.shift(axis=1)
moved = (n != n1) & (((n - n1).abs() == 1) | (n + n1 == 2 * usual))
up = (full & full.shift(axis=1, fill_value=False) & (prev > 0)
      & ((dps > prev * 1.001) | (moved & (mean > mean.shift(axis=1) * 1.001))))
streak = pd.DataFrame(0, index=up.index, columns=up.columns)
for y in years[1:]:
    streak[y] = (streak[y - 1] + 1) * up[y]

# Check the screen against S&P Dow Jones Indices' list after its January 2025 rebalance
OFFICIAL = set("""ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B BRO CAH CAT CB CHD CHRW CINF CL CLX
CTAS CVX DOV ECL ED EMR ERIE ES ESS EXPD FAST FDS FRT GD GPC GWW HRL IBM ITW JNJ KMB KO KVUE LIN LOW
MCD MDT MKC NDSN NEE NUE O PEP PG PNR PPG ROP SHW SJM SPGI SWK SYY TGT TROW WMT WST XOM""".split())
tick = {y: dict(zip(r["entity_id"].astype(int), r["ticker"])) for y, r in rosters.items()}
aristo = {y: {e for e in member[y] if e in streak.index and streak.at[e, y] >= 25} for y in member}
survivors = aristo[2024]                                                   # the end-2024 list

# Monthly total returns 2000-2024, with each exit's recorded delisting return
R = px.pivot(index="m", columns="entity_id", values="return_daily")
R = R.reindex(index=pd.period_range("2000-01", "2024-12", freq="M"))
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2024-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):   # one exit year per call, single pages
    g = g.index.tolist()
    for i in range(0, len(g), 300):
        d = xfl.prices(entity_id=g[i:i + 300], start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1
traded = {y: set(px.loc[px["m"] == pd.Period(f"{y}-12", "M"), "entity_id"]) for y in range(1999, 2024)}

def held(names_, y):
    """Equal weight at the start of year y+1, bought and held to December; cash after an exit."""
    r = R.loc[f"{y + 1}-01":f"{y + 1}-12", sorted(set(names_) & traded[y])]
    return ((1 + r.fillna(0)).prod() - 1).mean(), r.shape[1]

PIT, SURV, R24, EW = ("Point-in-time rosters", "End-2024 list, held from 2000",
                      "End-2024 roster, rule each December", "S&P 500 members, equal weight")
books = {PIT: lambda y: aristo[y], SURV: lambda y: survivors,
         R24: lambda y: {e for e in member[2024] if e in streak.index and streak.at[e, y] >= 25},
         EW: lambda y: member[y]}
out = {k: [held(f(y), y) for y in range(1999, 2024)] for k, f in books.items()}
res = pd.DataFrame({k: [r for r, _ in v] for k, v in out.items()}, index=range(2000, 2025))
cnt = pd.DataFrame({k: [c for _, c in v] for k, v in out.items()}, index=range(2000, 2025))
cagr = lambda r: (1 + r).prod() ** (1 / len(r)) - 1
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5

# ---------------- report ----------------
hit = {tick[2024][e] for e in survivors} & OFFICIAL
print(f"{len(rosters)} December rosters 1999-2024, {len(ids):,} securities, {len(amb):,} months read day by day, "
      f"{len(delist)} exit records")
print(f"\nEnd-2024 roster, 25-year rule: {len(survivors)} pass; official list: {len(OFFICIAL)}; on both: {len(hit)} "
      f"(recall {len(hit) / len(OFFICIAL):.1%}, precision {len(hit) / len(survivors):.1%})")
print("  official, not passing:", " ".join(sorted(OFFICIAL - hit)))
print("  passing, not official:", " ".join(sorted(tick[2024][e] for e in survivors if tick[2024][e] not in OFFICIAL)) or "none")

print("\nCalendar-year returns 2000-2024, equal weight, bought and held each year")
print(f"{'portfolio':38}{'names':>6}{'a year':>8}{'mean':>8}{'vs S&P EW':>10}{'t':>6}")
for k in books:
    x = res[k] - res[EW]
    print(f"{k:38}{cnt[k].mean():6.0f}{cagr(res[k]):8.2%}{res[k].mean():8.2%}"
          + (f"{x.mean() * 100:+10.2f}{tstat(x):6.2f}" if k != EW else ""))
for a, b, label in [(SURV, PIT, "end-2024 list minus point-in-time (total bias)"),
                    (SURV, R24, "  of which hindsight on the streak"),
                    (R24, PIT, "  of which members since removed")]:
    x = res[a] - res[b]
    print(f"{label:48}{x.mean() * 100:+6.2f} points a year, paired t {tstat(x):5.2f}, ahead {(x > 0).sum()}/25")
for a, b in [(2000, 2012), (2013, 2024)]:
    p = res.loc[a:b]
    print(f"  {a}-{b} a year: " + ", ".join(f"{k.split(',')[0]} {cagr(p[k]):.2%}" for k in books))
on = [held(aristo[y] & survivors, y)[0] for y in range(1999, 2024)]
off = [held(aristo[y] - survivors, y)[0] for y in range(1999, 2024)]
share = np.mean([len(aristo[y] & survivors) / len(aristo[y]) for y in range(1999, 2024)])
print(f"Inside each point-in-time list: {share:.1%} of names are on the end-2024 list (mean return "
      f"{np.mean(on):.2%}); the rest {np.nanmean(off):.2%}")

ever = set().union(*[aristo[y] for y in range(1999, 2024)])
fate = pd.Series({e: "on the end-2024 list" if e in survivors
                  else "in the S&P 500, streak ended" if e in member[2024]
                  else f"stopped trading ({delist.get(e, {}).get('event_type', 'no exit record')})" if e in ended.index
                  else "left the S&P 500, still trading" for e in ever})
alive = {e for e in ever if e in ended.index and streak.at[e, last[e].year - 1] >= 25}
broke = {e for e in ever if any(full.at[e, y] and streak.at[e, y] == 0 for y in range(2000, 2025)
                                if any(e in aristo[z] for z in range(1999, y)))}
print(f"\n{len(ever)} companies passed on a point-in-time roster, 1999-2023. At end-2024:")
for k, v in fate.value_counts().items():
    print(f"  {k:36}{v:4}")
print(f"  streak ended after passing: {len(broke)} ({len(broke) / len(ever):.1%}); "
      f"stopped trading with the streak intact: {len(alive)}")
c99 = aristo[1999]
print(f"December 1999 list: {len(c99)} names, {len(c99 & survivors)} on the end-2024 list; "
      f"{len(survivors - c99)} of the end-2024 list were not on it ({len(survivors - member[1999])} not yet S&P 500 members)")
print("Calendar-year dividend not raised (streak ends), tickers as on the previous December's roster:")
for y in range(2000, 2025):
    b = sorted(tick[y - 1][e] for e in aristo[y - 1] if full.at[e, y] and streak.at[e, y] == 0)
    if b:
        print(f"  {y} {len(b):3}  {' '.join(b)}")

# Chart: growth of $1, 2000-2024
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
style = {SURV: ("#d95926", "-"), R24: ("#199e70", "-"), PIT: ("#3987e5", "-"), EW: ("#a3a3a3", "--")}
for k, (color, ls) in style.items():
    g = pd.concat([pd.Series([1.0], index=[1999]), (1 + res[k]).cumprod()])
    ax.plot(g.index, g.values, color=color, ls=ls, lw=2, label=f"{k} ({cagr(res[k]):.1%} a year)")
    if k in (SURV, PIT):
        ax.annotate(f"${g.iloc[-1]:.2f}", (2024, g.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color="#e0e0e0", fontsize=9)
ax.set_yscale("log")
ax.set_yticks([1, 2, 4, 8, 16], ["$1", "$2", "$4", "$8", "$16"])
ax.minorticks_off()
ax.set_xlim(1999, 2026)
ax.set_xlabel("Year end")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("Dividend Aristocrats Backtest: Survivorship Bias Measured (S&P 500, 2000-2024)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("dividend-aristocrats-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
