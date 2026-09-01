# Full write-up: https://xfinlink.com/blog/buyback-timing-dollar-weighted-price-python
"""Do companies buy back their own stock at good prices?

Compares the dollar-weighted average price a repurchase programme paid against
the price an equal-dollar-every-year programme would have paid over the same
fiscal years. A company that spends the same amount every year scores exactly
1.000 by construction, so the score measures only the timing of the spending.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FY_FIRST, FY_LAST = 2015, 2024
MIN_YEARS = 5
MIN_MONTHS = 10

# 1. Point-in-time membership: the S&P 500 as it stood at the end of 2015,
#    carried forward by entity id so a later rename or delisting keeps the name.
roster = xfl.index("sp500", as_of="2015-12-31")
ids = sorted(set(int(i) for i in roster["entity_id"].dropna()))

# 2. Annual repurchase spending.
fun = xfl.fundamentals(entity_id=ids, start="2014-06-30", end="2025-12-31",
                       period_type="annual", fields=["share_repurchases"],
                       max_rows=40000)
fun["period_end"] = pd.to_datetime(fun["period_end"])
fun = fun[(fun["fiscal_year"] >= FY_FIRST) & (fun["fiscal_year"] <= FY_LAST)]
fun = fun.dropna(subset=["share_repurchases", "period_end"])
fun = fun[fun["share_repurchases"] > 0]
fun = fun.sort_values(["entity_id", "fiscal_year"]).drop_duplicates(
    ["entity_id", "fiscal_year"], keep="last")

counts = fun.groupby("entity_id").size()
keep = counts[counts >= MIN_YEARS].index.tolist()
fun = fun[fun["entity_id"].isin(keep)]

# 3. Monthly closes. Split-adjusted throughout, which is the correct basis:
#    an adjusted price is the price per current-equivalent share, so dollars
#    divided by adjusted price gives shares on today's terms.
px = []
for i in range(0, len(keep), 100):
    px.append(xfl.prices(entity_id=keep[i:i + 100], start="2014-01-01",
                         end="2025-12-31", interval="1mo",
                         fields=["adj_close"], max_rows=200000))
px = pd.concat(px, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
px = px[px["adj_close"] > 0].dropna(subset=["adj_close"])

# 4. Per company: the two average prices.
rows, yearly = [], []
for eid, g in fun.groupby("entity_id"):
    p = px[px["entity_id"] == eid]
    recs = []
    for _, r in g.iterrows():
        end = r["period_end"]
        w = p[(p["date"] > end - pd.Timedelta(days=364)) & (p["date"] <= end)]
        if len(w) < MIN_MONTHS:
            continue
        recs.append((int(r["fiscal_year"]), float(r["share_repurchases"]),
                     float(w["adj_close"].mean())))
    if len(recs) < MIN_YEARS:
        continue
    d = np.array([x[1] for x in recs])
    pr = np.array([x[2] for x in recs])
    dollar_weighted = d.sum() / (d / pr).sum()
    equal_dollar = len(pr) / (1.0 / pr).sum()
    rows.append({"ticker": g["ticker"].iloc[-1], "name": g["entity_name"].iloc[-1],
                 "sector": g["gics_sector"].iloc[-1], "years": len(recs),
                 "dollars": d.sum(), "score": dollar_weighted / equal_dollar})
    for fy, dd, pp in recs:
        yearly.append({"fiscal_year": fy, "dollars": dd, "price": pp, "ticker": rows[-1]["ticker"]})

res = pd.DataFrame(rows)
yr = pd.DataFrame(yearly)

# ---- output -------------------------------------------------------------
print("S&P 500 members at 2015-12-31, fiscal years %d-%d" % (FY_FIRST, FY_LAST))
print("companies with >=%d repurchase years and usable prices: %d" % (MIN_YEARS, len(res)))
print("company-years %d   total repurchased $%.2f trillion"
      % (res["years"].sum(), res["dollars"].sum() / 1e6))
print()
print("timing score (1.000 = spent evenly; above 1 = paid more than even spending would)")
print("  median            %.4f" % res["score"].median())
print("  mean              %.4f" % res["score"].mean())
print("  share above 1.0   %.1f%%" % (100 * (res["score"] > 1).mean()))
print("  dollar-weighted   %.4f" % (res["dollars"].sum() / (res["dollars"] / res["score"]).sum()))
print("  quartiles         %.3f / %.3f / %.3f"
      % tuple(res["score"].quantile([0.25, 0.5, 0.75])))
print()
print("by sector")
s = res.groupby("sector").agg(n=("score", "size"), median=("score", "median"),
                              dollars=("dollars", "sum"))
s["dollars"] = (s["dollars"] / 1000).round(0)
print(s.sort_values("median", ascending=False).round(4).to_string())
print()
print("the twelve largest programmes")
big = res.nlargest(12, "dollars")[["ticker", "name", "years", "dollars", "score"]]
big["dollars"] = (big["dollars"] / 1000).round(1)
print(big.to_string(index=False))
print()
print("best and worst timers among programmes above $5bn")
sub = res[res["dollars"] >= 5000]
print("  best  ", ", ".join("%s %.3f" % (r.ticker, r.score)
                            for r in sub.nsmallest(6, "score").itertuples()))
print("  worst ", ", ".join("%s %.3f" % (r.ticker, r.score)
                            for r in sub.nlargest(6, "score").itertuples()))
print()
agg = yr.groupby("fiscal_year").agg(dollars=("dollars", "sum"), price=("price", "median"))
agg["dollars"] = (agg["dollars"] / 1000).round(0)
agg["price_index"] = (100 * agg["price"] / agg["price"].iloc[0]).round(1)
print("aggregate spending against the sample's median share price")
print(agg[["dollars", "price_index"]].to_string())

# ---- chart --------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7))

a1.hist(res["score"].clip(upper=2.5), bins=60, color="#3b82f6", edgecolor="none")
a1.axvline(1.0, color="#e0e0e0", lw=1.2, ls="--")
a1.axvline(res["score"].median(), color="#f59e0b", lw=1.6)
a1.set_title("Buyback timing score, %d S&P 500 companies, fiscal %d-%d"
             % (len(res), FY_FIRST, FY_LAST))
a1.set_xlabel("Price paid, relative to spending the same amount every year")
a1.set_ylabel("Companies")
a1.text(res["score"].median() + 0.03, a1.get_ylim()[1] * 0.82,
        "median %.3f" % res["score"].median(), color="#f59e0b")
a1.text(1.0 - 0.03, a1.get_ylim()[1] * 0.82, "even spending", color="#e0e0e0",
        ha="right")
for sp in ("top", "right"):
    a1.spines[sp].set_visible(False)

a2.bar(agg.index, agg["dollars"], color="#3b82f6")
a2.set_ylabel("Repurchases ($bn)")
a2.set_xlabel("Fiscal year")
a2.set_title("Spending rises and falls with the share price")
a3 = a2.twinx()
a3.plot(agg.index, agg["price_index"], color="#f59e0b", lw=2, marker="o", ms=4)
a3.set_ylabel("Median share price (2015 = 100)", color="#f59e0b")
a3.tick_params(axis="y", colors="#f59e0b")
for sp in ("top",):
    a2.spines[sp].set_visible(False)
    a3.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("buyback-timing-dollar-weighted-price-python.png", dpi=150,
            facecolor="#0a0a0a")
