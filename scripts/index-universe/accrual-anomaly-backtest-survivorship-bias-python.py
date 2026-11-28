# Full write-up: https://xfinlink.com/blog/accrual-anomaly-backtest-survivorship-bias-python
#
# Accrual anomaly backtest on two S&P 500 universes. At each December close from 2010 to 2024,
# non-financial companies are sorted into quintiles by accruals: net income minus operating cash
# flow, over average total assets, from the latest fiscal year on record by the sort date (as_of).
# Each quintile is held the next calendar year, equal weight, bought and held, with monthly total
# returns; a holding that stops trading takes its recorded delisting return (an exit with none
# keeps its last traded price), then sits in cash. Returns end on 2025-12-31. Two universes:
#   honest: the S&P 500 roster in force on the sort date
#   naive:  today's S&P 500 roster traced back (its companies trading on the sort date)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

RUN_DATE = pd.Timestamp("2026-11-28")  # the day today's roster was read
forms = pd.date_range("2010-12-01", "2024-12-31", freq="BYE")  # last business day of December
ymd = lambda d: d.strftime("%Y-%m-%d")

FIELDS = ["net_income", "operating_cash_flow", "total_assets", "revenue", "sic"]

def statements(d, tickers=None, issuers=None):
    """Annual statements as on file at the close of d (as_of), fiscal years ending up to three years back."""
    kw = dict(period_type="annual", as_of=ymd(d), start=ymd(d - pd.DateOffset(years=3)), end=ymd(d),
              fields=FIELDS, max_rows=50_000)
    if tickers is not None:  # tickers resolve on the as_of date; rows carry the company id
        return xfl.fundamentals(tickers, **kw)
    return pd.concat([xfl.fundamentals(entity_id=issuers[i:i + 500], **kw) for i in range(0, len(issuers), 500)])

# Today's roster: security ids for prices, company ids for statements
today = xfl.index("sp500").dropna(subset=["entity_id", "ticker"])
now = xfl.fundamentals(today["ticker"].tolist(), period_type="annual", fields=["net_income"])
sec_of = dict(zip(now["entity_id"], now["ticker"].map(dict(zip(today["ticker"], today["entity_id"])))))

F, rows = [], []
for d in forms:
    roster = xfl.index("sp500", as_of=ymd(d)).dropna(subset=["entity_id", "ticker"])
    a = statements(d, tickers=roster["ticker"].tolist())
    a["sec"] = a["ticker"].map(dict(zip(roster["ticker"], roster["entity_id"])))
    b = statements(d, issuers=sorted(set(sec_of) - set(a["entity_id"])))  # today's names outside it
    b["sec"] = b["entity_id"].map(sec_of)
    rows.append(pd.concat([a.assign(honest=True), b.assign(honest=False)])[["entity_id", "ticker", "entity_name", "sec", "honest"]]
                .drop_duplicates("entity_id").assign(form=d, roster=len(roster)))
    F.append(pd.concat([a, b]).drop_duplicates(["entity_id", "period_end"]).assign(form=d))
S = pd.concat(rows, ignore_index=True).dropna(subset=["sec"])
S["sec"] = S["sec"].astype(int)
S["naive"] = S["entity_id"].isin(set(sec_of))
F = pd.concat(F, ignore_index=True)

# Keep true fiscal years: the company's usual year-end month, keyed on rows that carry revenue;
# a 12-month window ending in another month within 300 days of a neighbouring year-end is dropped
F["m"] = F["period_end"].dt.month
usual = F.groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax())
usual.update(F[F["revenue"].notna()].groupby("entity_id")["m"].agg(lambda s: s.value_counts().idxmax()))
F["off"] = ((F["m"] - F["entity_id"].map(usual)) % 12).map(lambda x: min(x, 12 - x))
ends = F.drop_duplicates(["entity_id", "period_end"]).sort_values(["entity_id", "period_end"])
g = ends.groupby("entity_id")["period_end"]
ends["near"] = np.minimum((ends["period_end"] - g.shift()).dt.days.fillna(999),
                          (g.shift(-1) - ends["period_end"]).dt.days.fillna(999))
F = F.merge(ends[["entity_id", "period_end", "near"]], on=["entity_id", "period_end"])
F = F[~((F["off"] >= 2) & (F["near"] < 300))]

# Latest fiscal year on record at the sort date (ended within 18 months), and the year before it
cur = F[((F["form"] - F["period_end"]).dt.days <= 548) & F[["net_income", "operating_cash_flow", "total_assets"]].notna().all(axis=1)]
cur = cur.sort_values("period_end")
cur = cur.drop_duplicates(["form", "entity_id"], keep="last")
pri = F.dropna(subset=["total_assets"])[["form", "entity_id", "period_end", "total_assets"]].rename(columns={"period_end": "pe0", "total_assets": "ta0"})
A = cur.merge(pri, on=["form", "entity_id"])
A = A[(A["period_end"] - A["pe0"]).dt.days.between(300, 430)].sort_values("pe0").drop_duplicates(["form", "entity_id"], keep="last")
A["accruals"] = (A["net_income"] - A["operating_cash_flow"]) / ((A["total_assets"] + A["ta0"]) / 2)
S = S.merge(A[["form", "entity_id", "period_end", "accruals", "sic"]], on=["form", "entity_id"], how="left")

# Monthly total returns, December 2010 to December 2025, by security
ids = sorted(S["sec"].unique())
px = pd.concat(xfl.prices(entity_id=ids[i:i + 200], start="2010-12-01", end="2025-12-31", interval="1mo",
                          fields=["return_daily", "close"], max_rows=250_000) for i in range(0, len(ids), 200))
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").reindex(
    index=pd.period_range("2010-12", "2025-12", freq="M"), columns=ids)
traded = px.pivot(index="m", columns="entity_id", values="close").reindex(index=R.index, columns=ids).notna()

# Exits: recorded delisting return added to the last month (one exit year per call)
last = R.apply(pd.Series.last_valid_index).dropna()
ended = last[last < pd.Period("2025-12", "M")]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    g = g.index.tolist()
    for i in range(0, len(g), 300):
        d = xfl.prices(entity_id=g[i:i + 300], start=f"{y}-01-01", end="2025-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e in last.index and dr is not None and not np.isnan(dr):
        m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
        if m <= R.index[-1]:
            R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1

S["y"] = S["form"].dt.year
S = S[[traded.at[pd.Period(f"{y}-12", "M"), s] for y, s in zip(S["y"], S["sec"])]]
S["sic"] = pd.to_numeric(S["sic"], errors="coerce")
n_cover = S[S["honest"]].groupby("y").agg(lines=("roster", "first"), have=("accruals", lambda s: s.notna().sum()))
# Companies sharing a company record with a merger partner before the merger, set aside for those
# years: MeadWestvaco (WestRock's record) to 2014; Tyco and Johnson Controls Inc to 2015
SET_ASIDE = {8097: 2014, 4074: 2015}
SET_ASIDE_YEARS = {670: (2012, 2013)}  # AGL Resources 2012-2013: an annual record dated ahead of its fiscal year
S = S[~(S["y"] <= S["entity_id"].map(SET_ASIDE))]
S = S[[not (e in SET_ASIDE_YEARS and SET_ASIDE_YEARS[e][0] <= y <= SET_ASIDE_YEARS[e][1])
       for e, y in zip(S["entity_id"], S["y"])]]
S = S[S["accruals"].notna() & S["sic"].notna() & ~S["sic"].between(6000, 6799)].copy()  # financials (SIC 6000-6799) left out

def held(secs, y):
    """Equal weight at the start of y+1, bought and held to December; cash after an exit."""
    return (1 + R.loc[f"{y + 1}-01":f"{y + 1}-12", secs].fillna(0)).prod() - 1

for y, g in S.groupby("y"):
    S.loc[g.index, "r"] = g["sec"].map(held(sorted(set(g["sec"])), y)).values

def quintiles(u):
    """1 = lowest accruals (earnings backed by cash) ... 5 = highest accruals."""
    return u.groupby("y")["accruals"].transform(lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)

U = {"Roster in force": S[S["honest"]].copy(), "Today's roster": S[S["naive"]].copy()}
tab, spread = {}, {}
for k, u in U.items():
    u["q"] = quintiles(u)
    tab[k] = u.groupby(["y", "q"])["r"].mean().unstack()
    spread[k] = tab[k][1] - tab[k][5]
tstat = lambda s: s.mean() / s.std() * len(s) ** 0.5
H, N = U  # labels
Y = len(forms)

# ---------------- report ----------------
print(f"{Y} December sorts {forms[0].date()} to {forms[-1].date()}, holding years {forms[0].year + 1}-2025; "
      f"today's roster {len(today)} lines on {RUN_DATE.date()}; {len(ids):,} securities, {len(delist)} exit records")
print(f"Roster lines with accruals on the sort date: {(n_cover['have'] / n_cover['lines']).min():.1%}-"
      f"{(n_cover['have'] / n_cover['lines']).max():.1%} a year")
for k, u in U.items():
    n = u.groupby("y").size()
    print(f"{k}: {n.min()}-{n.max()} non-financial companies a year, {len(u):,} company-years")
print(f"\n{'universe':18}{'Q1 low':>8}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5 high':>9}{'Q1-Q5':>8}{'t':>6}{'Q1>Q5':>7}")
for k in U:
    t = tab[k]
    print(f"{k:18}" + "".join(f"{t[q].mean():{w}.1%}" for q, w in zip(range(1, 6), (8, 7, 7, 7, 9)))
          + f"{spread[k].mean() * 100:+8.2f}{tstat(spread[k]):6.2f}{(spread[k] > 0).sum():>4}/{Y}")
med = U[H].groupby("q")["accruals"].median()
print("Median accruals, roster in force: " + "  ".join(f"Q{q} {v:+.1%}" for q, v in med.items()))

print(f"\nSurvivorship effect (today's roster minus roster in force), points a year, paired over {Y} years")
for q in range(1, 6):
    x = tab[N][q] - tab[H][q]
    print(f"  Q{q}{'':12}{x.mean() * 100:+6.2f}  paired t {tstat(x):5.2f}  higher in {(x > 0).sum():>2}/{Y} years")
x = tab[N].mean(axis=1) - tab[H].mean(axis=1)
print(f"  Quintile average{x.mean() * 100:+6.2f}  paired t {tstat(x):5.2f}  higher in {(x > 0).sum():>2}/{Y} years")
x = spread[N] - spread[H]
print(f"  Q1-Q5 spread   {x.mean() * 100:+6.2f}  paired t {tstat(x):5.2f}  higher in {(x > 0).sum():>2}/{Y} years")

h = U[H].assign(other=~U[H]["naive"])  # in the index then, not on today's roster
n = U[N].assign(other=~U[N]["honest"])  # on today's roster, not yet in the index
for lab, u, col in [("Roster in force: members since removed", h, "removed"),
                    ("Today's roster: companies not yet in the index", n, "not yet in")]:
    print(f"\n{lab}: {u['other'].sum():,} of {len(u):,} company-years ({u['other'].mean():.1%})")
    print(f"  {'quintile':10}{'names':>6}{col:>12}{'share':>7}{'return, ' + col:>21}{'return, rest':>14}")
    for q, g in u.groupby("q"):
        print(f"  Q{q:<9}{len(g):6}{g['other'].sum():12}{g['other'].mean():7.1%}"
              f"{g.loc[g['other'], 'r'].mean():21.1%}{g.loc[~g['other'], 'r'].mean():14.1%}")

# Chart: quintile returns on both universes, and the share of each quintile that has left the index
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(5)
for off, k, color in [(-0.2, H, "#3b82f6"), (0.2, N, "#c2780a")]:
    bars = ax.bar(x + off, [tab[k][q].mean() * 100 for q in range(1, 6)], width=0.38, color=color, label=k)
    ax.bar_label(bars, fmt="%.1f", padding=2, color="#e0e0e0", fontsize=8.5)
ax.set_xticks(x, ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
ax.set_xlabel("Accrual quintile")
ax.set_ylabel(f"Average annual return, {forms[0].year + 1}-2025 (%)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
for off, u, color, lab in [(-0.2, h, "#3b82f6", "Roster in force:\nlater removed"),
                           (0.2, n, "#c2780a", "Today's roster:\nnot yet in the index")]:
    bars = bx.bar(x + off, u.groupby("q")["other"].mean().values * 100, width=0.38, color=color, label=lab)
    bx.bar_label(bars, fmt="%.0f", padding=2, color="#e0e0e0", fontsize=8.5)
bx.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"])
bx.set_xlabel("Accrual quintile")
bx.set_ylabel("Share of company-years (%)")
bx.set_ylim(0, 45)
bx.legend(frameon=False, loc="upper right", fontsize=8.5)
for a in (ax, bx):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Accrual Anomaly Backtest: Survivorship Bias in the S&P 500")
plt.tight_layout()
plt.savefig("accrual-anomaly-backtest-survivorship-bias-python.png", dpi=150, facecolor="#0a0a0a")
