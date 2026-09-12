# Full write-up: https://xfinlink.com/blog/quarterly-earnings-vs-revenue-forecastability-python
#
# Are earnings harder to forecast than revenue?
# Foster's seasonal autoregressive model fitted firm by firm on quarterly
# S&P 500 revenue and net income, with a held-out forecast comparison.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SPLIT = pd.Timestamp("2021-12-31")

idx = xfl.index("sp500")
tickers = sorted(idx["ticker"].dropna().unique().tolist())
fund = xfl.fundamentals(tickers, period_type="quarterly",
                        fields=["revenue", "net_income"],
                        start="2011-01-01", end="2026-09-12", max_rows=200000)


def fit_firm(y):
    """Foster model vs seasonal naive on one quarterly series."""
    y = y.dropna()
    if len(y) < 50:
        return None
    gaps = y.index.to_series().diff().dt.days.dropna()
    if gaps.max() > 100 or gaps.min() < 75:
        return None
    d = y.diff(4)
    f = pd.DataFrame({"y": y, "d": d, "dlag": d.shift(1), "y1": y.shift(1),
                      "y4": y.shift(4), "y5": y.shift(5)}).dropna()
    ins, oos = f[f.index <= SPLIT], f[f.index > SPLIT]
    if len(ins) < 25 or len(oos) < 8:
        return None
    x = np.column_stack([np.ones(len(ins)), ins["dlag"].values])
    delta, phi = np.linalg.lstsq(x, ins["d"].values, rcond=None)[0]
    naive = (oos["y"] - oos["y4"]).abs().mean()
    foster = (oos["y"] - (oos["y4"] + delta + phi * (oos["y1"] - oos["y5"]))).abs().mean()
    scale = oos["y"].abs().mean()
    return {"phi": phi, "rel_naive": naive / scale, "ratio": foster / naive}


rows = []
for eid, g in fund.groupby("entity_id"):
    g = g.drop_duplicates("period_end", keep="last").sort_values("period_end")
    rev = fit_firm(g.set_index("period_end")["revenue"].astype(float))
    ni = fit_firm(g.set_index("period_end")["net_income"].astype(float))
    if rev is None or ni is None:
        continue
    rows.append({"ticker": g["ticker"].iloc[-1],
                 **{f"{k}_rev": v for k, v in rev.items()},
                 **{f"{k}_ni": v for k, v in ni.items()}})

res = pd.DataFrame(rows)

print(f"Sample: {len(res)} S&P 500 firms, quarterly data from 2011, "
      f"forecasts from 2022 Q1 onward\n")
print(f"{'':<28}{'Revenue':>10}{'Net income':>13}")
for label, a, b in [
    ("Median phi", res["phi_rev"].median(), res["phi_ni"].median()),
    ("Share with phi > 0", (res["phi_rev"] > 0).mean(), (res["phi_ni"] > 0).mean()),
    ("Seasonal-naive error", res["rel_naive_rev"].median(), res["rel_naive_ni"].median()),
    ("Foster / seasonal-naive", res["ratio_rev"].median(), res["ratio_ni"].median()),
    ("Share Foster beats naive", (res["ratio_rev"] < 1).mean(), (res["ratio_ni"] < 1).mean()),
]:
    fmt = "{:>10.1%}{:>13.1%}" if "Share" in label or "error" in label else "{:>10.3f}{:>13.3f}"
    print(f"{label:<28}" + fmt.format(a, b))

print(f"\n{'Firm':<8}{'phi rev':>9}{'phi NI':>8}{'naive err rev':>15}{'naive err NI':>14}")
for t in ["AAPL", "KO", "MSFT", "NVDA", "WMT"]:
    r = res[res["ticker"] == t].iloc[0]
    print(f"{t:<8}{r['phi_rev']:>9.3f}{r['phi_ni']:>8.3f}"
          f"{r['rel_naive_rev']:>14.1%}{r['rel_naive_ni']:>14.1%}")

# ---- chart ----
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
bins = np.linspace(-0.6, 1.2, 31)
axes[0].hist(res["phi_rev"].clip(-0.6, 1.2), bins=bins, color="#3b82f6", alpha=0.85,
             label="Revenue")
axes[0].hist(res["phi_ni"].clip(-0.6, 1.2), bins=bins, color="#f59e0b", alpha=0.65,
             label="Net income")
axes[0].axvline(0, color="#e0e0e0", lw=0.8)
axes[0].set_xlabel("Persistence of the year-over-year change (phi)")
axes[0].set_ylabel("Number of firms")
axes[0].set_title("How much last quarter's change carries over", fontsize=11)
axes[0].legend(frameon=False)

bins2 = np.linspace(0.2, 1.6, 29)
axes[1].hist(res["ratio_rev"].clip(0.2, 1.6), bins=bins2, color="#3b82f6", alpha=0.85,
             label="Revenue")
axes[1].hist(res["ratio_ni"].clip(0.2, 1.6), bins=bins2, color="#f59e0b", alpha=0.65,
             label="Net income")
axes[1].axvline(1.0, color="#e0e0e0", lw=0.8, ls="--")
axes[1].set_xlabel("Forecast error vs same quarter last year (ratio)")
axes[1].set_ylabel("Number of firms")
axes[1].set_title("Held-out forecast error, 2022 Q1 onward", fontsize=11)
axes[1].legend(frameon=False)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Quarterly revenue is far more forecastable than quarterly earnings",
             fontsize=13, color="#e0e0e0")
plt.tight_layout()
plt.savefig("quarterly-earnings-vs-revenue-forecastability-python.png", dpi=150,
            facecolor="#0a0a0a")
