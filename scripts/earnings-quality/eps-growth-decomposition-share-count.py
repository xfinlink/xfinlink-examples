# Full write-up: https://xfinlink.com/blog/eps-growth-decomposition-share-count
#
# EPS growth is an identity: in logs, EPS growth = net income growth - share count growth.
# This splits a decade of large-cap EPS growth into the part that came from earning more
# and the part that came from dividing by fewer shares.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

BASE = (pd.Timestamp("2014-06-01"), pd.Timestamp("2015-05-31"))
END = (pd.Timestamp("2024-06-01"), pd.Timestamp("2025-05-31"))
FIELDS = ["net_income", "eps_diluted", "weighted_avg_shares_diluted", "gics_sector"]

# Point-in-time S&P 500 roster, so the universe is the one an investor actually
# faced in 2014 rather than the survivors picked with hindsight. Entity ids are
# used instead of tickers because tickers get reassigned over a decade.
roster = xfl.index("sp500", as_of="2014-06-30")
ids = sorted(roster["entity_id"].unique().tolist())

frames = []
for i in range(0, len(ids), 50):
    frames.append(xfl.fundamentals(entity_id=ids[i:i + 50], period_type="annual",
                                   start="2013-06-01", end="2025-12-31", fields=FIELDS))
df = pd.concat([f for f in frames if len(f)], ignore_index=True)
df = df.dropna(subset=["period_end"]).sort_values(["entity_id", "period_end"])


def pick(g, window):
    rows = g[(g.period_end >= window[0]) & (g.period_end <= window[1])]
    return rows.iloc[-1] if len(rows) else None


records, funnel = [], {}


def step(label):
    funnel[label] = funnel.get(label, 0) + 1


for eid, g in df.groupby("entity_id"):
    step("companies in the 2014 roster")
    b, e = pick(g, BASE), pick(g, END)
    if b is None or e is None:
        continue
    step("both endpoint fiscal years reported")
    if not 3600 <= (e.period_end - b.period_end).days <= 3700:
        continue
    step("endpoints exactly ten fiscal years apart")

    vals = {}
    for tag, row in (("b", b), ("e", e)):
        ni, eps, sh = row.net_income, row.eps_diluted, row.weighted_avg_shares_diluted
        if pd.isna(ni) or pd.isna(eps) or pd.isna(sh):
            break
        vals[tag] = (float(ni), float(eps), float(sh))
    if len(vals) < 2:
        continue
    step("earnings, per-share and share count all present")

    if not all(v > 0 for pair in vals.values() for v in pair):
        continue
    step("profitable in both endpoint years")

    # The identity only holds if EPS, net income and the share count describe the
    # same earnings base. Where reported EPS times reported shares does not
    # reproduce reported net income, the per-share figure rests on a different
    # base (preferred dividends, minority interests, discontinued operations).
    artic = max(abs(v[1] * v[2] - v[0]) / v[0] for v in vals.values())
    if artic > 0.01:
        continue
    step("endpoint rows reconcile within 1%")

    win = g[(g.period_end >= b.period_end) & (g.period_end <= e.period_end)]
    sh = win["weighted_avg_shares_diluted"].astype(float)
    if sh.isna().any() or len(win) < 9:
        continue
    step("continuous annual share-count history")

    # A large-cap share count moves by single-digit percentages through ordinary
    # issuance and repurchase. A jump beyond a quarter means the two years are not
    # on one basis: a stock split, or a share-funded acquisition.
    ratio = (sh / sh.shift(1)).dropna()
    if ((ratio > 1.25) | (ratio < 0.80)).any():
        continue
    step("share count on a single basis throughout")

    records.append(dict(ticker=e.ticker, name=e.entity_name, sector=e.gics_sector,
                        base_end=b.period_end.date(), end_end=e.period_end.date(),
                        ni0=vals["b"][0], ni1=vals["e"][0], eps0=vals["b"][1],
                        eps1=vals["e"][1], sh0=vals["b"][2], sh1=vals["e"][2]))

r = pd.DataFrame(records)
r["g_eps"] = np.log(r.eps1 / r.eps0)          # total EPS growth, in logs
r["g_profit"] = np.log(r.ni1 / r.ni0)         # the profit leg
r["g_shares"] = -np.log(r.sh1 / r.sh0)        # the share count leg
r["residual"] = r.g_eps - (r.g_profit + r.g_shares)

ann = lambda x: (np.exp(x / 10) - 1) * 100    # ten years of log growth to % a year
for col in ("g_eps", "g_profit", "g_shares"):
    r[col + "_a"] = ann(r[col])

print("SAMPLE CONSTRUCTION")
for label in ["companies in the 2014 roster", "both endpoint fiscal years reported",
              "endpoints exactly ten fiscal years apart",
              "earnings, per-share and share count all present",
              "profitable in both endpoint years", "endpoint rows reconcile within 1%",
              "continuous annual share-count history",
              "share count on a single basis throughout"]:
    print(f"  {funnel[label]:>4}  {label}")

print(f"\nIDENTITY CHECK  largest residual across {len(r)} companies: "
      f"{r.residual.abs().max():.5f} log points")

# Log points add exactly; annualised percentages do not, so the additive view
# is reported in logs and the annualised view is given separately for scale.
print("\nDECOMPOSITION OF TEN-YEAR EPS GROWTH (cross-sectional mean, log points)")
print(f"  profit leg        {r.g_profit.mean():+.4f}")
print(f"  share count leg   {r.g_shares.mean():+.4f}")
print(f"  sum of the legs   {r.g_profit.mean() + r.g_shares.mean():+.4f}")
print(f"  EPS growth        {r.g_eps.mean():+.4f}")
print(f"  share of EPS growth supplied by the share count: "
      f"{100 * r.g_shares.mean() / r.g_eps.mean():.1f}%")
print(f"\nFOR SCALE, MEDIAN COMPANY, ANNUALISED: EPS {r.g_eps_a.median():.2f}%   "
      f"profit {r.g_profit_a.median():.2f}%   share count {r.g_shares_a.median():.2f}%")

grew = r[r.g_eps_a >= 2]
print(f"\n  share count retired: {(r.g_shares > 0).sum()} of {len(r)} companies")
print(f"  EPS rose while profit fell: {((r.g_eps > 0) & (r.g_profit < 0)).sum()} companies")
print(f"  median share of EPS growth from the share count, among the {len(grew)} "
      f"companies growing EPS at 2% a year or better: "
      f"{(100 * grew.g_shares / grew.g_eps).median():.1f}%")

top = r.sort_values("g_shares", ascending=False).head(12)
print("\nLARGEST SHARE COUNT LEG (annualised %)")
print(f"{'':<6}{'company':<30}{'EPS':>7}{'profit':>9}{'shares':>9}")
for _, x in top.iterrows():
    print(f"{x.ticker:<6}{x['name'][:28]:<30}{x.g_eps_a:>7.1f}{x.g_profit_a:>9.1f}"
          f"{x.g_shares_a:>9.1f}")

sec = (r.groupby("sector").agg(n=("ticker", "size"), eps=("g_eps_a", "median"),
                               profit=("g_profit_a", "median"),
                               shares=("g_shares_a", "median")))
sec = sec[sec.n >= 10].sort_values("shares", ascending=False)
print("\nBY SECTOR, MEDIAN ANNUALISED % (sectors with at least 10 companies)")
print(f"{'sector':<26}{'n':>4}{'EPS':>8}{'profit':>9}{'shares':>9}")
for name, x in sec.iterrows():
    print(f"{name:<26}{int(x.n):>4}{x.eps:>8.1f}{x.profit:>9.1f}{x.shares:>9.1f}")

# ---- chart ----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
PROFIT, SHARES = "#3b82f6", "#f59e0b"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

# Left: every company as a point. The two legs are the coordinates, so a company
# sitting above the zero line grew EPS faster than it grew profit. The dashed
# curve is exact flat EPS: (1+profit)(1+shares) = 1.
ax1.axhline(0, color="#5a5a5a", lw=0.8)
grid = np.linspace(-20, 32, 200)
ax1.plot(grid, 100 * (1 / (1 + grid / 100) - 1), "--", color="#8a8a8a", lw=0.9)
ax1.scatter(r.g_profit_a, r.g_shares_a, s=18, c=np.where(r.g_shares > 0, SHARES, PROFIT),
            alpha=0.8, linewidths=0)
for _, x in pd.concat([r.nlargest(2, "g_shares"), r.nsmallest(2, "g_shares")]).iterrows():
    ax1.annotate(x.ticker, (x.g_profit_a, x.g_shares_a), fontsize=7.5, color="#e0e0e0",
                 xytext=(5, 2), textcoords="offset points")
ax1.set_xlim(-20, 32)
ax1.set_ylim(r.g_shares_a.min() - 1.5, r.g_shares_a.max() + 1.5)
ax1.set_xlabel("Profit leg: net income growth, % a year")
ax1.set_ylabel("Share count leg: reduction in shares, % a year")
ax1.set_title(f"Two legs of EPS growth, {len(r)} large caps, 2014-2024",
              color="#e0e0e0", fontsize=11)
ax1.text(-18.5, r.g_shares_a.min() - 0.9, "dashed curve: EPS flat",
         fontsize=7.5, color="#8a8a8a")

# Right: sector medians as side-by-side bars. Not stacked, because the two legs
# carry opposite signs in some sectors and medians do not sum to the median.
s = sec.iloc[::-1]
ys = np.arange(len(s))
ax2.barh(ys + 0.20, s.profit, height=0.38, color=PROFIT, label="profit leg")
ax2.barh(ys - 0.20, s.shares, height=0.38, color=SHARES, label="share count leg")
ax2.set_yticks(ys, [i.replace(" ", "\n", 1) if len(i) > 14 else i for i in s.index])
ax2.axvline(0, color="#5a5a5a", lw=0.8)
ax2.set_xlabel("Median annualised growth, % a year")
ax2.set_title("By sector", color="#e0e0e0", fontsize=11)
ax2.legend(frameon=False, loc="upper right", fontsize=8)

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("eps-growth-decomposition-share-count.png", dpi=150, facecolor="#0a0a0a")
