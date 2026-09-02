# Full write-up: https://xfinlink.com/blog/earnings-vs-multiple-return-decomposition-python
"""Did earnings or the multiple drive the last decade of stock returns?

Splits each company's fiscal 2014 to fiscal 2024 total return into three
additive log pieces: growth in diluted earnings per share, change in the
price/earnings multiple, and the contribution of reinvested dividends.
Universe is the point-in-time S&P 500 roster, addressed by entity id.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST, LAST = 2014, 2024
PE_LO, PE_HI = 5.0, 150.0
# Earnings and the multiple carry a different meaning for lenders, insurers and
# property owners, so those two sectors sit outside this comparison.
EXCLUDE = {"Financials", "Real Estate"}

# 1. Point-in-time roster, carried by entity id so a rename does not split a
#    company's history into two shorter ones.
roster = xfl.index("sp500", as_of="2024-12-31")
ids = sorted(int(i) for i in roster["entity_id"].dropna().unique())

# 2. Diluted EPS at both ends of the decade.
fun = xfl.fundamentals(entity_id=ids, start="2013-12-01", end="2025-06-30",
                       period_type="annual",
                       fields=["eps_diluted", "net_income_available_to_common",
                               "weighted_avg_shares_diluted"],
                       max_rows=40000)
fun = fun[fun["fiscal_year"].isin([FIRST, LAST])].copy()
# A handful of filers carry a fiscal-year label that its own period end
# contradicts, so keep only the rows where the two agree.
fun = fun[(((fun.fiscal_year == FIRST) & fun.period_end.between("2014-01-01", "2015-06-30")) |
           ((fun.fiscal_year == LAST) & fun.period_end.between("2024-01-01", "2024-12-31")))]
fun = fun.sort_values("period_end").drop_duplicates(["entity_id", "fiscal_year"], keep="last")

a = fun[fun.fiscal_year == FIRST].set_index("entity_id")
b = fun[fun.fiscal_year == LAST].set_index("entity_id")
pair = pd.DataFrame({
    "eps0": a["eps_diluted"], "d0": a["period_end"],
    "ni0": a["net_income_available_to_common"], "sh0": a["weighted_avg_shares_diluted"],
    "eps1": b["eps_diluted"], "d1": b["period_end"],
    "ni1": b["net_income_available_to_common"], "sh1": b["weighted_avg_shares_diluted"],
    "ticker": b["ticker"], "name": b["entity_name"], "sector": b["gics_sector"],
}).dropna(subset=["eps0", "eps1", "d0", "d1"])
pair = pair[~pair["sector"].isin(EXCLUDE)]
pair = pair[(pair["eps0"] > 0) & (pair["eps1"] > 0)]  # a multiple on a loss has no reading
# Reported EPS has to sit in the same order of magnitude as net income to
# common over diluted shares.
chk0 = ((pair["ni0"] / pair["sh0"] - pair["eps0"]).abs() / pair["eps0"].abs()).fillna(0)
chk1 = ((pair["ni1"] / pair["sh1"] - pair["eps1"]).abs() / pair["eps1"].abs()).fillna(0)
pair = pair[(chk0 <= 2.0) & (chk1 <= 2.0)]

# 3. Monthly bars: total return, raw close, split-adjusted close.
keep_ids = sorted(int(i) for i in pair.index)
frames = []
for i in range(0, len(keep_ids), 50):
    frames.append(xfl.prices(entity_id=keep_ids[i:i + 50], start="2014-01-01", end="2025-06-30",
                             interval="1mo", fields=["close", "adj_close", "return_daily"],
                             max_rows=50000))
px = pd.concat(frames, ignore_index=True)
px["m"] = px["date"].dt.to_period("M")

rows = []
for eid, g in px.groupby("entity_id"):
    g = g.sort_values("date").reset_index(drop=True)
    r = pair.loc[eid]
    # A monthly bar is stamped with the month's first trading day and carries
    # that month's closing price, so match on the calendar month.
    m0, m1 = r["d0"].to_period("M"), r["d1"].to_period("M")
    h0, h1 = g.index[g["m"] == m0], g.index[g["m"] == m1]
    if len(h0) != 1 or len(h1) != 1:
        continue
    i0, i1 = int(h0[0]), int(h1[0])
    if i1 <= i0:
        continue
    b0, b1 = g.loc[i0], g.loc[i1]
    seg = g.loc[i0 + 1:i1]
    if not (b0["close"] > 0 and b1["close"] > 0 and b0["adj_close"] > 0 and b1["adj_close"] > 0):
        continue
    if seg["return_daily"].isna().any() or len(seg) < 100:
        continue
    # The multiple pairs a price with the earnings of the year it belongs to,
    # both on the share count of that date, so no split adjustment enters here.
    pe0, pe1 = b0["close"] / r["eps0"], b1["close"] / r["eps1"]
    years = (m1 - m0).n / 12.0
    log_total = float(np.log1p(seg["return_daily"]).sum())
    log_price = float(np.log(b1["adj_close"] / b0["adj_close"]))
    log_mult = float(np.log(pe1 / pe0))
    rows.append({
        "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
        "m0": str(m0), "m1": str(m1), "years": years, "pe0": pe0, "pe1": pe1,
        "earn": (log_price - log_mult) / years,   # per-share earnings growth
        "mult": log_mult / years,                 # re-rating
        "divs": (log_total - log_price) / years,  # reinvested dividends
        "total": log_total / years,
    })

raw = pd.DataFrame(rows).dropna(subset=["sector"])
keep = (raw["pe0"].between(PE_LO, PE_HI) & raw["pe1"].between(PE_LO, PE_HI)
        & (raw["divs"] >= -1e-4))
res = raw[keep].reset_index(drop=True)

print("Point-in-time S&P 500 roster at 2024-12-31, fiscal %d to fiscal %d" % (FIRST, LAST))
print("companies with a readable multiple at both ends: %d" % len(res))
print("window length in years, min/median/max: %.2f / %.2f / %.2f"
      % (res["years"].min(), res["years"].median(), res["years"].max()))
print("median multiple, start %.1f  end %.1f" % (res["pe0"].median(), res["pe1"].median()))
print("identity check, max |earnings + multiple + dividends - total|: %.1e"
      % (res["earn"] + res["mult"] + res["divs"] - res["total"]).abs().max())
print()

q = 100 * res[["total", "earn", "mult", "divs"]].quantile([.1, .25, .5, .75, .9])
q.index = ["10th pct", "25th pct", "median", "75th pct", "90th pct"]
q.columns = ["total", "earnings", "multiple", "dividends"]
print("annualised contribution to total return, percentage points")
print(q.round(2).to_string())
print("mean          %s" % "  ".join("%.2f" % (100 * res[c].mean())
                                     for c in ["total", "earn", "mult", "divs"]))
print()

print("multiple expanded over the decade:          %.1f%%" % (100 * (res["mult"] > 0).mean()))
print("multiple contributed more than earnings:    %.1f%%" % (100 * (res["mult"] > res["earn"]).mean()))
print("multiple was the largest of the three:      %.1f%%"
      % (100 * (res[["earn", "mult", "divs"]].idxmax(axis=1) == "mult").mean()))
print("multiple compressed and the stock still rose: %.1f%%"
      % (100 * ((res["mult"] < 0) & (res["total"] > 0)).mean()))
print("earnings per share fell and the stock still rose: %.1f%%"
      % (100 * ((res["earn"] < 0) & (res["total"] > 0)).mean()))
print()

sec = res.groupby("sector").agg(n=("ticker", "size"), earnings=("earn", "median"),
                                multiple=("mult", "median"), dividends=("divs", "median"),
                                total=("total", "median"))
for c in ["earnings", "multiple", "dividends", "total"]:
    sec[c] = (100 * sec[c]).round(2)
sec = sec.sort_values("total", ascending=False)
print("by sector, median annualised contribution in percentage points")
print(sec.to_string())
print()

cols = ["ticker", "name", "pe0", "pe1", "earn", "mult", "divs", "total"]
for label, frame in [("the eight largest re-ratings up", res.nlargest(8, "mult")),
                     ("the eight largest re-ratings down", res.nsmallest(8, "mult"))]:
    t = frame[cols].copy()
    for c in ["earn", "mult", "divs", "total"]:
        t[c] = (100 * t[c]).round(1)
    print(label + ", annualised percentage points")
    print(t.round(1).to_string(index=False))
    print()

# 4. Chart: stacked contributions by sector.
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 7))
s = sec.sort_values("total")
y = np.arange(len(s))
pos, neg = np.zeros(len(s)), np.zeros(len(s))
for col, colour, label in [("earnings", "#3b82f6", "Earnings growth"),
                           ("multiple", "#f59e0b", "Change in the P/E multiple"),
                           ("dividends", "#9ca3af", "Dividends")]:
    v = s[col].values
    ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=colour, label=label, height=0.62)
    pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
ax.plot(s["total"].values, y, "o", color="#e0e0e0", markersize=5, label="Median total return")
ax.set_yticks(y)
ax.set_yticklabels(s.index)
ax.axvline(0, color="#6b7280", linewidth=0.8)
ax.set_xlabel("Median annualised contribution to total return (percentage points)")
ax.set_title("Where a decade of S&P 500 returns came from, fiscal 2014 to fiscal 2024")
leg = ax.legend(loc="lower right", labelcolor="#e0e0e0", framealpha=1.0)
leg.get_frame().set_facecolor("#0a0a0a")
leg.get_frame().set_edgecolor("#3a3a3a")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig("earnings-vs-multiple-return-decomposition-python.png", dpi=150, facecolor="#0a0a0a")
