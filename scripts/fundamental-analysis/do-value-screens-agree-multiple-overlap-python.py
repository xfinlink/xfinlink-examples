# Full write-up: https://xfinlink.com/blog/do-value-screens-agree-multiple-overlap-python
"""Do five standard valuation multiples pick the same cheap stocks?

Ranks the S&P 500 on P/E, P/B, P/S, P/FCF and EV/EBIT from one daily
valuation snapshot, then measures how much the cheapest quintiles overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SNAPSHOT = "2026-07-31"
MULTIPLES = ["pe_ratio", "pb_ratio", "ps_ratio", "price_to_fcf", "ev_ebit"]
LABELS = {"pe_ratio": "P/E", "pb_ratio": "P/B", "ps_ratio": "P/S",
          "price_to_fcf": "P/FCF", "ev_ebit": "EV/EBIT"}
EXTRA = ["roe", "net_margin", "debt_to_equity", "asset_turnover", "market_cap"]
OUT_PNG = "do-value-screens-agree-multiple-overlap-python.png"

# ── Data ──────────────────────────────────────────────────────────────
tickers = sorted(set(xfl.index("sp500")["ticker"].dropna()))

frames = []
for i in range(0, len(tickers), 100):
    frames.append(
        xfl.metrics(
            tickers[i:i + 100],
            period_type="daily",
            start="2026-07-27",
            end=SNAPSHOT,
            fields=MULTIPLES + EXTRA,
        )
    )
snap = (
    pd.concat(frames, ignore_index=True)
    .sort_values("period_end")
    .groupby("ticker", as_index=False)
    .last()
)

# A multiple with a non-positive denominator carries no ranking information,
# so the screen keeps only names priced on all five at once.
d = snap.dropna(subset=MULTIPLES + EXTRA)
d = d[(d[MULTIPLES] > 0).all(axis=1)].reset_index(drop=True)
n_cheap = int(round(len(d) * 0.2))

# ── Agreement between the multiples ───────────────────────────────────
spearman = d[MULTIPLES].corr(method="spearman")

cheap = {c: set(d.nsmallest(n_cheap, c)["ticker"]) for c in MULTIPLES}
overlap = pd.DataFrame(
    [[100 * len(cheap[a] & cheap[b]) / n_cheap for b in MULTIPLES] for a in MULTIPLES],
    index=MULTIPLES, columns=MULTIPLES,
)

all_five = sorted(set.intersection(*cheap.values()))
any_one = set.union(*cheap.values())

# ── What each cheap bucket is made of ─────────────────────────────────
profile = pd.DataFrame(
    [
        {
            "screen": LABELS[c],
            "roe": d.nsmallest(n_cheap, c)["roe"].median(),
            "net_margin": d.nsmallest(n_cheap, c)["net_margin"].median(),
            "debt_to_equity": d.nsmallest(n_cheap, c)["debt_to_equity"].median(),
            "asset_turnover": d.nsmallest(n_cheap, c)["asset_turnover"].median(),
        }
        for c in MULTIPLES
    ]
)
full = {
    "screen": "all names",
    "roe": d["roe"].median(),
    "net_margin": d["net_margin"].median(),
    "debt_to_equity": d["debt_to_equity"].median(),
    "asset_turnover": d["asset_turnover"].median(),
}
profile = pd.concat([profile, pd.DataFrame([full])], ignore_index=True)

# ── Output ────────────────────────────────────────────────────────────
print(f"Snapshot {SNAPSHOT} | {len(tickers)} tickers | {len(d)} priced on all five multiples")
print(f"Cheapest quintile = {n_cheap} names per screen\n")

print("Spearman rank correlation between multiples")
print(spearman.rename(index=LABELS, columns=LABELS).round(2).to_string(), "\n")

print("Share of one screen's cheapest quintile that also appears in another (%)")
print(overlap.rename(index=LABELS, columns=LABELS).round(0).astype(int).to_string(), "\n")

print("Median profile of each cheapest quintile")
print(profile.assign(
    roe=lambda x: (100 * x["roe"]).round(1),
    net_margin=lambda x: (100 * x["net_margin"]).round(1),
    debt_to_equity=lambda x: x["debt_to_equity"].round(2),
    asset_turnover=lambda x: x["asset_turnover"].round(2),
).to_string(index=False), "\n")

print(f"Cheapest quintile on all five multiples: {len(all_five)} names")
print("  " + ", ".join(all_five))
print(f"Cheapest quintile on at least one multiple: {len(any_one)} names "
      f"({len(any_one) / n_cheap:.1f}x the {n_cheap} slots a single screen fills)")

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "xfl", ["#0a0a0a", "#1e3a5f", "#3b82f6"]
)
labels = [LABELS[c] for c in MULTIPLES]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

grid = overlap.to_numpy()
ax1.imshow(grid, cmap=cmap, vmin=0, vmax=100)
ax1.set_xticks(range(len(labels)), labels)
ax1.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax1.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                 color="#e0e0e0" if grid[i, j] < 70 else "#0a0a0a", fontsize=10)
ax1.set_title(f"Shared names in the cheapest {n_cheap} (%)", pad=10)
for spine in ax1.spines.values():
    spine.set_visible(False)

y = np.arange(len(labels))
bars = profile.iloc[:len(labels)]
ax2.barh(y + 0.19, 100 * bars["roe"], height=0.36, color="#3b82f6", label="Return on equity")
ax2.barh(y - 0.19, 100 * bars["net_margin"], height=0.36, color="#6b7280", label="Net margin")
ax2.axvline(100 * full["roe"], color="#e0e0e0", linestyle="--", linewidth=0.9)
ax2.text(100 * full["roe"] + 0.3, len(labels) - 0.45, "all names (ROE)", fontsize=8, color="#e0e0e0")
ax2.set_yticks(y, labels)
ax2.set_ylim(-1.05, len(labels) - 0.35)
ax2.set_xlabel("Median for the cheapest quintile (%)")
ax2.set_title("What each cheap bucket is made of", pad=10)
ax2.legend(frameon=False, loc="lower right", fontsize=8)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

fig.suptitle("Five value screens, five different cheap lists (S&P 500, 31 July 2026)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")
