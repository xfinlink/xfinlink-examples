# Full write-up: https://xfinlink.com/blog/does-leverage-explain-beta-unlevering-python
"""Does leverage explain beta?

Estimates equity betas against SPY, removes the effect of borrowing to get an
asset beta, and asks whether the leverage adjustment does the job theory says
it does: raise beta with debt, and leave a business-risk number behind.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2021-01-01", "2025-12-31"
FISCAL_YEAR = 2024
MIN_DAYS = 1000

# 1. Point-in-time membership, addressed by entity id.
roster = xfl.index("sp500", as_of="2024-12-31")
ids = sorted(set(int(i) for i in roster["entity_id"].dropna()))

# 2. Capital structure. Net debt comes from enterprise value minus market
#    capitalisation, so it needs no book-equity figure at all.
cap = []
for i in range(0, len(ids), 200):
    cap.append(xfl.metrics(entity_id=ids[i:i + 200], period_type="annual",
                           start="2024-01-01", end="2025-06-30",
                           fields=["market_cap", "enterprise_value"],
                           max_rows=20000))
cap = pd.concat(cap, ignore_index=True)
cap = cap[cap["fiscal_year"] == FISCAL_YEAR] if "fiscal_year" in cap else cap
cap = cap.dropna(subset=["market_cap", "enterprise_value"])
cap = cap[(cap["market_cap"] > 0) & (cap["enterprise_value"] > 0)]
cap = cap.sort_values("period_end").groupby("entity_id").tail(1)
cap["equity_share"] = cap["market_cap"] / cap["enterprise_value"]
# A company holding more cash than debt has an equity share above 1. Keep the
# ones where the adjustment stays interpretable.
cap = cap[(cap["equity_share"] > 0.3) & (cap["equity_share"] < 1.6)]

# 3. Daily returns against the market.
spy = xfl.prices("SPY", start=START, end=END, fields=["return_daily"])
spy["date"] = pd.to_datetime(spy["date"])
mkt = spy.set_index("date")["return_daily"].dropna()

eids = [int(i) for i in cap["entity_id"]]
px = []
for i in range(0, len(eids), 60):
    px.append(xfl.prices(entity_id=eids[i:i + 60], start=START, end=END,
                         fields=["return_daily"], max_rows=200000))
px = pd.concat(px, ignore_index=True)
px["date"] = pd.to_datetime(px["date"])
px = px.dropna(subset=["return_daily"])

rows = []
for eid, g in px.groupby("entity_id"):
    r = g.set_index("date")["return_daily"]
    j = pd.concat([r.rename("stock"), mkt.rename("mkt")], axis=1).dropna()
    if len(j) < MIN_DAYS:
        continue
    beta = j["stock"].cov(j["mkt"]) / j["mkt"].var()
    c = cap[cap["entity_id"] == eid].iloc[0]
    rows.append({"ticker": c["ticker"], "name": c["entity_name"],
                 "sector": c["gics_sector"], "days": len(j),
                 "equity_beta": beta, "equity_share": c["equity_share"],
                 "asset_beta": beta * c["equity_share"],
                 "net_debt_share": 1 - c["equity_share"]})

res = pd.DataFrame(rows).dropna(subset=["sector"])

# ---- output -------------------------------------------------------------
print("S&P 500 members at 2024-12-31, daily returns %s to %s" % (START, END))
print("companies with fiscal %d capital structure and >=%d trading days: %d"
      % (FISCAL_YEAR, MIN_DAYS, len(res)))
print()
print("                     mean   median      sd      IQR")
for col, label in (("equity_beta", "equity beta"), ("asset_beta", "asset beta")):
    q = res[col].quantile([0.25, 0.75])
    print("  %-14s %7.3f %7.3f %7.3f  %.2f-%.2f"
          % (label, res[col].mean(), res[col].median(), res[col].std(),
             q.iloc[0], q.iloc[1]))
print("  cross-sectional dispersion falls %.1f%%"
      % (100 * (1 - res["asset_beta"].std() / res["equity_beta"].std())))
print()
res["q"] = pd.qcut(res["net_debt_share"].rank(method="first"), 5,
                   labels=["Q1 least levered", "Q2", "Q3", "Q4", "Q5 most levered"])
tab = res.groupby("q", observed=True).agg(
    n=("equity_beta", "size"), net_debt=("net_debt_share", "median"),
    eq_beta=("equity_beta", "median"), as_beta=("asset_beta", "median"))
tab["net_debt"] = (100 * tab["net_debt"]).round(1)
print("quintiles of net debt as a share of enterprise value")
print(tab.round(3).to_string())
print("  spread Q5 minus Q1:  equity beta %+.3f   asset beta %+.3f"
      % (tab["eq_beta"].iloc[-1] - tab["eq_beta"].iloc[0],
         tab["as_beta"].iloc[-1] - tab["as_beta"].iloc[0]))
print()
print("within-sector standard deviation")
s = res.groupby("sector").agg(n=("equity_beta", "size"),
                              eq_sd=("equity_beta", "std"),
                              as_sd=("asset_beta", "std"),
                              net_debt=("net_debt_share", "median"))
s["net_debt"] = (100 * s["net_debt"]).round(1)
s["fall"] = (100 * (1 - s["as_sd"] / s["eq_sd"])).round(1)
print(s.sort_values("net_debt", ascending=False).round(3).to_string())
print()
print("the most levered names in the sample")
big = res.nlargest(8, "net_debt_share")[["ticker", "name", "sector",
                                         "net_debt_share", "equity_beta", "asset_beta"]]
big["net_debt_share"] = (100 * big["net_debt_share"]).round(1)
print(big.round(3).to_string(index=False))

# ---- chart --------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))

bins = np.linspace(0, 2.4, 45)
a1.hist(res["equity_beta"], bins=bins, color="#3b82f6", alpha=0.85, label="Equity beta")
a1.hist(res["asset_beta"], bins=bins, color="#f59e0b", alpha=0.65, label="Asset beta")
a1.set_xlabel("Beta against the market")
a1.set_ylabel("Companies")
a1.set_title("Removing debt narrows the spread")
a1.legend(frameon=False)

x = np.arange(len(tab))
a2.plot(x, tab["eq_beta"], color="#3b82f6", marker="o", lw=2, label="Equity beta")
a2.plot(x, tab["as_beta"], color="#f59e0b", marker="o", lw=2, label="Asset beta")
a2.set_xticks(x)
a2.set_xticklabels(["least\nlevered", "Q2", "Q3", "Q4", "most\nlevered"])
a2.set_ylabel("Median beta")
a2.set_xlabel("Net debt as a share of enterprise value")
a2.set_title("Beta against leverage")
a2.legend(frameon=False)
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("does-leverage-explain-beta-unlevering-python.png", dpi=150,
            facecolor="#0a0a0a")
