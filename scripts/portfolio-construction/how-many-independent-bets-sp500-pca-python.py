# Full write-up: https://xfinlink.com/blog/how-many-independent-bets-sp500-pca-python
"""How many independent bets does the S&P 500 contain?

Year-by-year eigendecomposition of the correlation matrix of daily returns for
point-in-time S&P 500 members, 2015-2025, with a permutation null so the
eigenvalue spectrum can be read against what pure noise produces.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2015, 2026))
CHUNK = 100
rng = np.random.default_rng(0)


def year_panel(year):
    """Daily returns for the S&P 500 as it stood on 1 January of `year`."""
    roster = xfl.index("sp500", as_of=f"{year}-01-01")
    eids = sorted(roster["entity_id"].tolist())
    frames = [xfl.prices(entity_id=eids[i:i + CHUNK],
                         start=f"{year}-01-01", end=f"{year}-12-31",
                         fields=["return_daily"], max_rows=200_000)
              for i in range(0, len(eids), CHUNK)]
    px = pd.concat(frames, ignore_index=True)
    piv = px.pivot_table(index="date", columns="entity_id", values="return_daily")
    piv = piv[piv.notna().sum(axis=1) >= 100]          # real trading days only
    return len(eids), piv[piv.columns[piv.notna().sum() == len(piv)]]


def spectrum(matrix):
    corr = np.corrcoef(matrix, rowvar=False)
    return corr, np.sort(np.linalg.eigvalsh(corr))[::-1]


def effective_bets(eig, n):
    """exp(entropy) of the normalised eigenvalues: bets on the position scale."""
    p = np.clip(eig / n, 1e-15, None)
    return float(np.exp(-(p * np.log(p)).sum()))


rows, spectra = [], {}
for year in YEARS:
    roster_size, panel = year_panel(year)
    A = panel.values
    days, n = A.shape
    corr, eig = spectrum(A)

    # permutation null: each name keeps its own returns, the dates are shuffled,
    # so any co-movement between names is destroyed and the rest is unchanged
    _, eig_null = spectrum(np.column_stack([rng.permutation(A[:, j]) for j in range(n)]))

    ew = A.mean(axis=1)
    rows.append({
        "year": year, "roster": roster_size, "names": n, "days": days,
        "trace": float(eig.sum()),
        "pc1": eig[0] / n, "pc2": eig[1] / n,
        "bets": effective_bets(eig, n),
        "bets_null": effective_bets(eig_null, n),
        "real_factors": int((eig > eig_null[0]).sum()),
        "avg_corr": float(corr[np.triu_indices(n, 1)].mean()),
        "port_vol": float(ew.std(ddof=1) * np.sqrt(252)),
        "stock_vol": float((A.std(axis=0, ddof=1) * np.sqrt(252)).mean()),
    })
    spectra[year] = eig

res = pd.DataFrame(rows).set_index("year")
res["vol_cut"] = 1 - res["port_vol"] / res["stock_vol"]

print("Point-in-time S&P 500 members, daily returns, one correlation matrix per year")
print(f"Check: eigenvalues sum to the number of stocks in every year "
      f"(largest gap {max(abs(res['trace'] - res['names'])):.2e})")

print("\nHOW MUCH OF THE CROSS-SECTION IS ONE COMMON FACTOR")
print(f"{'year':>6s}{'stocks':>8s}{'days':>6s}{'PC1':>8s}{'PC2':>7s}{'avg corr':>10s}"
      f"{'bets':>8s}{'factors':>9s}{'port vol':>10s}")
for year, r in res.iterrows():
    print(f"{year:>6d}{int(r['names']):>8d}{int(r['days']):>6d}{r['pc1'] * 100:>7.1f}%"
          f"{r['pc2'] * 100:>6.1f}%{r['avg_corr']:>10.2f}{r['bets']:>8.1f}"
          f"{int(r['real_factors']):>9d}{r['port_vol'] * 100:>9.1f}%")

calm, tight = res["pc1"].idxmin(), res["pc1"].idxmax()
print(f"\nWidest dispersion : {calm}  PC1 {res.loc[calm, 'pc1'] * 100:.1f}%  "
      f"{res.loc[calm, 'bets']:.1f} bets from {int(res.loc[calm, 'names'])} stocks")
print(f"Tightest year     : {tight}  PC1 {res.loc[tight, 'pc1'] * 100:.1f}%  "
      f"{res.loc[tight, 'bets']:.1f} bets from {int(res.loc[tight, 'names'])} stocks")
print(f"Range of the effective bet count: {res['bets'].min():.1f} to {res['bets'].max():.1f}"
      f"   median {res['bets'].median():.1f}")

print(f"\nPermutation null (same stocks, same days, co-movement shuffled out)")
print(f"  the estimator returns {res['bets_null'].min():.0f} to {res['bets_null'].max():.0f} "
      f"bets when the names are genuinely independent")
print(f"  components rising above that noise floor: "
      f"{res['real_factors'].min()} to {res['real_factors'].max()} per year")

print(f"\nPC1 share against the equal-weight portfolio's volatility: "
      f"correlation {res['pc1'].corr(res['port_vol']):.2f} across {len(res)} years")
print(f"Volatility an equal-weight book removes versus the average single stock: "
      f"{res['vol_cut'].min() * 100:.0f}% ({res['vol_cut'].idxmin()}) to "
      f"{res['vol_cut'].max() * 100:.0f}% ({res['vol_cut'].idxmax()})")

# ---- chart -------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
x = res.index.to_numpy()

ax1.bar(x, res["pc1"] * 100, color="#3b82f6", width=0.62)
for xi, v in zip(x, res["pc1"] * 100):
    ax1.text(xi, v + 1.0, f"{v:.0f}", ha="center", color="#9ca3af", fontsize=9)
ax1.set_ylabel("Variance in one common factor (%)")
ax1.set_ylim(0, max(res["pc1"] * 100) + 8)
ax1.set_title("One factor moves the S&P 500 cross-section, and its grip varies by year",
              color="#e0e0e0", fontsize=13, pad=10)

ax2.bar(x, res["bets"], color="#3b82f6", width=0.62)
for xi, v in zip(x, res["bets"]):
    ax2.text(xi, v + 0.8, f"{v:.0f}", ha="center", color="#9ca3af", fontsize=9)
ax2.set_ylabel("Effective independent bets")
ax2.set_xlabel("Year")
ax2.set_ylim(0, max(res["bets"]) + 14)
ax2.text(x[-1] + 0.4, max(res["bets"]) + 7,
         f"out of roughly {int(res['names'].median())} stocks held",
         ha="right", color="#9ca3af", fontsize=9)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
plt.tight_layout()
plt.savefig("how-many-independent-bets-sp500-pca-python.png", dpi=150, facecolor="#0a0a0a")
