# Full write-up: https://xfinlink.com/blog/reverse-dcf-implied-growth-sp500-python
"""
Reverse DCF on the S&P 500: what free-cash-flow growth rate does each share
price already assume, and how does that compare with what each company
actually delivered over the past decade?

Model: 10 explicit years of growth at g, then a perpetuity growing at 2.5%,
discounted at 9%. Solve for the g that makes the model value equal the
market capitalisation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

HORIZON = 10          # explicit forecast years
TERMINAL = 0.025      # perpetuity growth after year 10
RATES = [0.08, 0.09, 0.10, 0.11]
BASE_RATE = 0.09
G_LO, G_HI = -0.40, 0.60   # range of growth rates the model can express
EXCLUDE = {"Financials", "Real Estate"}

# ── 1. Universe: current S&P 500, addressed by permanent entity id ──────
members = xfl.index("sp500")
ids = members["entity_id"].dropna().astype(int).tolist()

quarterly, annual, market = [], [], []
for i in range(0, len(ids), 50):
    batch = ids[i:i + 50]
    quarterly.append(xfl.fundamentals(entity_id=batch, period_type="quarterly",
                                      start="2025-01-01", fields=["free_cash_flow"],
                                      max_rows=50000))
    annual.append(xfl.fundamentals(entity_id=batch, period_type="annual",
                                   start="2014-06-01",
                                   fields=["free_cash_flow", "fiscal_year"],
                                   max_rows=50000))
    market.append(xfl.prices(entity_id=batch, start="2026-05-15",
                             fields=["close", "market_cap"], max_rows=200000))
q = pd.concat(quarterly, ignore_index=True)
a = pd.concat(annual, ignore_index=True)
p = pd.concat(market, ignore_index=True)

# ── 2. Base cash flow: sum of the four most recent quarterly filings ────
q = q.dropna(subset=["free_cash_flow"]).sort_values("period_end")
ttm = q.groupby("entity_id").tail(4).groupby("entity_id").agg(
    fcf_ttm=("free_cash_flow", "sum"),
    n_q=("free_cash_flow", "size"),
    last_q=("period_end", "max"))
ttm = ttm[(ttm["n_q"] == 4) & (ttm["last_q"] >= "2026-01-01")]

# ── 3. Price: median market cap over the last 60 trading days ───────────
cap = p.dropna(subset=["market_cap"]).sort_values("date")
cap = cap.groupby("entity_id").tail(60).groupby("entity_id")["market_cap"].median() / 1e6

# ── 4. Delivered growth: fiscal 2015 to fiscal 2025 free-cash-flow CAGR ─
labels = a.sort_values("period_end").groupby("entity_id")[
    ["ticker", "entity_name", "gics_sector"]].last()
fy = a[a["fiscal_year"].isin([2015, 2025])].pivot_table(
    index="entity_id", columns="fiscal_year", values="free_cash_flow", aggfunc="last")
fy.columns = [f"fcf_{c}" for c in fy.columns]

d = ttm.join(labels).join(cap.rename("mcap")).join(fy)
d = d[~d["gics_sector"].isin(EXCLUDE) & d["gics_sector"].notna()]
d = d[(d["fcf_ttm"] > 0) & (d["mcap"] > 0)].copy()
d["pfcf"] = d["mcap"] / d["fcf_ttm"]


# ── 5. Invert the DCF: solve for the growth rate that justifies the price
def model_multiple(g, r):
    """Value of $1 of current free cash flow, as a multiple, at growth g."""
    t = np.arange(1, HORIZON + 1)
    explicit = (((1 + g) ** t) / ((1 + r) ** t)).sum()
    terminal = (1 + g) ** HORIZON * (1 + TERMINAL) / ((r - TERMINAL) * (1 + r) ** HORIZON)
    return explicit + terminal


def implied_growth(multiple, r):
    f = lambda g: model_multiple(g, r) - multiple
    if f(G_LO) > 0 or f(G_HI) < 0:
        return np.nan          # outside the range the model can express
    return brentq(f, G_LO, G_HI, xtol=1e-9)


for r in RATES:
    d[f"g_{int(r * 100)}"] = d["pfcf"].apply(lambda m, r=r: implied_growth(m, r))
d = d.dropna(subset=[f"g_{int(BASE_RATE * 100)}"])

have_record = d["fcf_2015"].gt(0) & d["fcf_2025"].gt(0)
d.loc[have_record, "delivered"] = (d.loc[have_record, "fcf_2025"] /
                                   d.loc[have_record, "fcf_2015"]) ** (1 / 10) - 1

g9 = f"g_{int(BASE_RATE * 100)}"
d["quintile"] = pd.qcut(d["pfcf"], 5, labels=False) + 1

# ── 6. Results ─────────────────────────────────────────────────────────
print(f"sample: {len(d)} S&P 500 members outside financials and real estate")
print(f"model: {HORIZON} explicit years, {TERMINAL:.1%} terminal growth, "
      f"{BASE_RATE:.1%} discount rate\n")

print("implied 10-year free-cash-flow growth, whole sample")
for label, value in [("10th percentile", d[g9].quantile(0.10)),
                     ("median", d[g9].median()),
                     ("90th percentile", d[g9].quantile(0.90))]:
    print(f"  {label:<16} {value:>7.1%}")

print("\nby price-to-free-cash-flow quintile")
print(f"{'quintile':<10}{'n':>5}{'median P/FCF':>15}{'implied growth':>17}{'delivered growth':>19}")
tab = d.groupby("quintile").agg(n=("pfcf", "size"), pfcf=("pfcf", "median"),
                                implied=(g9, "median"), delivered=("delivered", "median"))
for k, row in tab.iterrows():
    name = {1: "1 cheapest", 5: "5 dearest"}.get(k, str(k))
    print(f"{name:<10}{int(row['n']):>5}{row['pfcf']:>14.1f}x"
          f"{row['implied']:>16.1%}{row['delivered']:>19.1%}")

print("\ndiscount rate sensitivity (median implied growth)")
med = {r: d[f"g_{int(r * 100)}"].median() for r in RATES}
for r in RATES:
    print(f"  {r:>5.1%}{med[r]:>10.1%}")
slope = (med[RATES[-1]] - med[RATES[0]]) / (RATES[-1] - RATES[0])
print(f"  1 point of discount rate moves implied growth by {slope:.2f} points")

both = d.dropna(subset=["delivered"])
print(f"\nimplied vs delivered, {len(both)} names with a full 10-year record")
print(f"  median implied growth   {both[g9].median():>7.1%}")
print(f"  median delivered growth {both['delivered'].median():>7.1%}")
print(f"  Spearman rank correlation {both[g9].corr(both['delivered'], method='spearman'):>6.2f}")
above = (both[g9] > both["delivered"]).sum()
print(f"  priced above own delivered growth: {above} of {len(both)} "
      f"({above / len(both):.0%})")

# ── 7. Chart ───────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

lim = (-0.30, 0.60)
ax1.scatter(both["delivered"], both[g9], s=18, color="#3b82f6", alpha=0.65,
            edgecolors="none")
ax1.plot(lim, lim, color="#9ca3af", lw=1, ls="--")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Free cash flow growth delivered, 2015 to 2025 (per year)")
ax1.set_ylabel("Growth the price implies")
ax1.set_title("What the price assumes vs what the company delivered", loc="left", pad=8)
ax1.text(0.02, 0.93, "above the dashed line: priced for faster growth than the "
                     "past decade produced", transform=ax1.transAxes,
         color="#9ca3af", fontsize=9)
ax1.text(0.02, 0.86, f"Spearman rank correlation "
                     f"{both[g9].corr(both['delivered'], method='spearman'):.2f}",
         transform=ax1.transAxes, color="#9ca3af", fontsize=9)
for ax in (ax1, ax2):
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

shades = {8: "#bfdbfe", 9: "#60a5fa", 10: "#3b82f6", 11: "#1d4ed8"}
order = d.sort_values("pfcf")
for r in RATES:
    k = int(r * 100)
    ax2.plot(order["pfcf"], order[f"g_{k}"], color=shades[k], lw=1.8,
             label=f"{r:.0%} discount rate")
mid = d["pfcf"].median()
ax2.axvline(mid, color="#6b7280", lw=1, ls=":")
ax2.annotate(f"at the sample median of {mid:.0f}x, the four assumptions span "
             f"{med[RATES[0]]:.1%} to {med[RATES[-1]]:.1%}",
             xy=(mid * 1.15, -0.15), color="#9ca3af", fontsize=9)
ax2.set_xscale("log")
ax2.set_xlim(4, 700)
ax2.set_xticks([5, 10, 20, 40, 80, 160, 320, 640])
ax2.set_xticklabels(["5x", "10x", "20x", "40x", "80x", "160x", "320x", "640x"])
ax2.set_xlabel("Price to trailing free cash flow")
ax2.set_ylabel("Growth the price implies")
ax2.set_title("The same price, four discount rate assumptions", loc="left", pad=8)
ax2.legend(frameon=False, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("reverse-dcf-implied-growth-sp500-python.png", dpi=150,
            facecolor="#0a0a0a")
