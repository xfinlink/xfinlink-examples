# Full write-up: https://xfinlink.com/blog/benfords-law-reported-financials-python
#
# Do the figures S&P 500 companies report on their financial statements follow
# Benford's Law? Chi-square goodness-of-fit and the mean absolute deviation
# (MAD) statistic, line by line, on a point-in-time index roster.
#
# A weak Benford fit is not evidence of manipulation. It is a property of how a
# number is generated: figures confined to a narrow band of magnitudes cannot
# produce a logarithmic first-digit distribution no matter how honest they are.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF = "2015-12-31"
START, END = "2010-01-01", "2025-12-31"

LINES = ["revenue", "total_assets", "net_income",
         "operating_cash_flow", "capital_expenditures", "eps_diluted"]
FIELDS = LINES + ["weighted_avg_shares_diluted"]

LABELS = {"revenue": "Revenue", "total_assets": "Total assets",
          "net_income": "Net income", "operating_cash_flow": "Operating cash flow",
          "capital_expenditures": "Capital expenditures", "eps_diluted": "Diluted EPS"}

# Benford's Law: P(first digit = d) = log10(1 + 1/d)
DIGITS = np.arange(1, 10)
BENFORD = np.log10(1 + 1 / DIGITS)
assert abs(BENFORD.sum() - 1.0) < 1e-12


def first_digit(x):
    """Leading significant digit of |x|, for any non-zero magnitude."""
    return int(f"{abs(x):.15e}"[0])


# ------------------------------------------------------------------ data pull
roster = xfl.index("sp500", as_of=AS_OF)
entity_ids = roster["entity_id"].dropna().astype(int).tolist()

frames = [
    xfl.fundamentals(entity_id=entity_ids[i:i + 50], period_type="annual",
                     start=START, end=END, fields=FIELDS)
    for i in range(0, len(entity_ids), 50)
]
raw = pd.concat(frames, ignore_index=True).drop_duplicates(["entity_id", "period_end"])

# Keep company-years that report all six figures, so every line is measured on
# exactly the same rows, and where the per-share figure reconciles with the
# income statement (diluted EPS within 25% of net income / diluted shares).
d = raw.dropna(subset=FIELDS).copy()
for c in LINES:
    d = d[d[c] != 0]
d = d[d["weighted_avg_shares_diluted"] > 0]
ratio = d["eps_diluted"] / (d["net_income"] / d["weighted_avg_shares_diluted"])
sample = d[(ratio >= 0.80) & (ratio <= 1.25)]

print(f"Roster as of {AS_OF}: {len(entity_ids)} companies")
print(f"Annual filings {START[:4]}-{END[:4]}: {len(raw):,} company-years retrieved")
print(f"Complete on all six lines:  {len(d):,}")
print(f"After per-share reconciliation: {len(sample):,} company-years, "
      f"{sample['entity_id'].nunique()} companies")
print(f"Digits tested: {len(sample) * len(LINES):,}")
print()

# ----------------------------------------------------------- per-line results
rows, counts = [], {}
for col in LINES:
    v = sample[col].abs()
    obs = np.array([(v.map(first_digit) == k).sum() for k in DIGITS], dtype=float)
    n = int(obs.sum())
    assert n == len(v), "observed digit counts must sum to N"

    chi2, p = stats.chisquare(obs, BENFORD * n)
    mad = np.abs(obs / n - BENFORD).mean()
    logs = np.log10(v)
    span = logs.quantile(0.95) - logs.quantile(0.05)  # orders of magnitude, middle 90%

    counts[col] = obs
    rows.append({"line": LABELS[col], "n": n, "mad": mad,
                 "chi2": chi2, "p": p, "span": span})

res = pd.DataFrame(rows).sort_values("mad").reset_index(drop=True)

print(f"{'Statement line':<22}{'N':>7}{'MAD':>9}{'chi-square':>12}{'p-value':>11}{'log10 span':>12}")
print("-" * 73)
for _, r in res.iterrows():
    print(f"{r['line']:<22}{int(r['n']):>7,}{r['mad']:>9.4f}"
          f"{r['chi2']:>12,.1f}{r['p']:>11.2e}{r['span']:>12.2f}")

rho, rho_p = stats.spearmanr(res["mad"], res["span"])
print(f"\nSpearman rank correlation, MAD vs magnitude span: "
      f"rho = {rho:.2f} (p = {rho_p:.3f}, n = {len(res)})")
print()

# ------------------------------------------- monetary lines pooled vs Benford
MONETARY = [c for c in LINES if c != "eps_diluted"]
pooled = sum(counts[c] for c in MONETARY)
n_pool = int(pooled.sum())
chi2_pool, p_pool = stats.chisquare(pooled, BENFORD * n_pool)
mad_pool = np.abs(pooled / n_pool - BENFORD).mean()

print(f"Five monetary lines pooled (N = {n_pool:,}):  MAD = {mad_pool:.4f}   "
      f"chi-square = {chi2_pool:,.1f}   p = {p_pool:.2e}")
print(f"{'Digit':<7}{'Observed':>10}{'Benford':>10}{'Diff':>9}")
for i, k in enumerate(DIGITS):
    o = pooled[i] / n_pool
    print(f"{k:<7}{o:>9.2%}{BENFORD[i]:>10.2%}{o - BENFORD[i]:>+9.2%}")
print()

# ------------------------- how large MAD gets on chance alone at company scale
# Draw digits from the exact Benford distribution at the size of one company's
# full filing history, 5,000 times, and look at the resulting MAD.
per_company = int(np.median(sample.groupby("entity_id").size()) * len(MONETARY))
rng = np.random.default_rng(0)
draws = rng.multinomial(per_company, BENFORD, size=5000) / per_company
null_mad = np.abs(draws - BENFORD).mean(axis=1)

print(f"Median monetary figures available per company: {per_company}")
print(f"MAD of clean Benford data at N = {per_company}: "
      f"median {np.median(null_mad):.4f}, 95th percentile {np.quantile(null_mad, 0.95):.4f}")
print("Nigrini first-digit thresholds: 0.0060 close, 0.0120 acceptable, 0.0150 nonconformity")

# -------------------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

w = 0.38
eps_freq = counts["eps_diluted"] / counts["eps_diluted"].sum()
ax1.bar(DIGITS - w / 2, pooled / n_pool, w, color="#3b82f6", label="Monetary lines")
ax1.bar(DIGITS + w / 2, eps_freq, w, color="#8b5cf6", label="Diluted EPS")
ax1.plot(DIGITS, BENFORD, "o--", color="#e0e0e0", lw=1.4, ms=5, label="Benford's Law")
ax1.set_xticks(DIGITS)
ax1.set_yticks(np.arange(0, 0.31, 0.05))
ax1.set_yticklabels([f"{t:.0%}" for t in np.arange(0, 0.31, 0.05)])
ax1.set_xlabel("Leading digit")
ax1.set_ylabel("Share of reported figures")
ax1.set_title("First-digit frequencies, S&P 500 annual filings", fontsize=11)
ax1.legend(frameon=False, fontsize=9)

y = np.arange(len(res))
ax2.barh(y, res["mad"], color=["#3b82f6" if m < 0.012 else "#8b5cf6" for m in res["mad"]])
ax2.set_yticks(y)
ax2.set_yticklabels(res["line"], fontsize=9)
ax2.invert_yaxis()
ax2.axvline(0.006, color="#e0e0e0", ls=":", lw=1)
ax2.axvline(0.015, color="#e0e0e0", ls="--", lw=1)
ax2.text(0.0056, 2.5, "close conformity", fontsize=8, color="#9a9a9a",
         rotation=90, va="center", ha="right")
ax2.text(0.0146, 2.5, "nonconformity", fontsize=8, color="#9a9a9a",
         rotation=90, va="center", ha="right")
ax2.set_xlim(0, 0.016)
ax2.set_xlabel("Mean absolute deviation from Benford")
ax2.set_title("Fit by statement line, 2010-2025", fontsize=11)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("benfords-law-reported-financials-python.png", dpi=150)
