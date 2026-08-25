# Full write-up: https://xfinlink.com/blog/cash-conversion-cycle-analysis-python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# -- Universe: 38 large non-financial US companies across six sectors -------
# Financials and real estate are excluded: a bank has no inventory and no
# cost of goods sold, so the cycle does not describe how the business works.
TICKERS = [
    "COST", "TGT", "KR", "SYY", "PG", "PEP", "CL", "GIS",
    "HD", "LOW", "ORLY", "AZO", "GPC", "NKE", "TSLA",
    "AAPL", "DELL", "CSCO", "NVDA", "TXN", "INTC",
    "MCK", "CVS", "JNJ", "PFE", "MRK", "LLY", "ABBV", "AMGN",
    "CAT", "HON", "LMT", "EMR", "ETN", "GWW",
    "CVX", "MPC", "VLO",
]

FIELDS = [
    "ticker", "gics_sector", "period_end", "revenue", "cost_of_revenue",
    "accounts_receivable", "inventory", "accounts_payable",
]

# -- Pull the two most recent annual filings per company --------------------
df = xfl.fundamentals(
    TICKERS, period_type="annual", start="2023-01-01", fields=FIELDS
).sort_values(["ticker", "period_end"])

# -- Build the cycle from average balances ----------------------------------
rows = []
for ticker, group in df.groupby("ticker"):
    group = group.tail(2)
    cur, prev = group.iloc[-1], group.iloc[-2]
    receivables = (cur.accounts_receivable + prev.accounts_receivable) / 2
    stock = (cur.inventory + prev.inventory) / 2
    payables = (cur.accounts_payable + prev.accounts_payable) / 2

    dso = 365 * receivables / cur.revenue
    dio = 365 * stock / cur.cost_of_revenue
    dpo = 365 * payables / cur.cost_of_revenue

    rows.append({
        "ticker": ticker,
        "sector": cur.gics_sector,
        "fy_end": cur.period_end.date(),
        "dso": dso,
        "dio": dio,
        "dpo": dpo,
        "ccc": dso + dio - dpo,
    })

cycle = pd.DataFrame(rows).sort_values("ccc").reset_index(drop=True)

# -- Print the company table ------------------------------------------------
print("Cash conversion cycle, latest annual filing (days)")
print("DSO = receivable days, DIO = inventory days, DPO = payable days")
print()
header = f"{'ticker':<7}{'sector':<24}{'fy end':<12}{'DSO':>7}{'DIO':>8}{'DPO':>8}{'CCC':>9}"
print(header)
print("-" * len(header))
for _, r in cycle.iterrows():
    print(f"{r.ticker:<7}{r.sector:<24}{str(r.fy_end):<12}"
          f"{r.dso:>7.1f}{r.dio:>8.1f}{r.dpo:>8.1f}{r.ccc:>9.1f}")

# -- Print the sector summary -----------------------------------------------
summary = (cycle.groupby("sector")["ccc"]
           .agg(["count", "median", "min", "max"])
           .sort_values("median"))
print()
print("Sector medians")
sub = f"{'sector':<24}{'n':>4}{'median':>10}{'min':>9}{'max':>9}"
print(sub)
print("-" * len(sub))
for sector, r in summary.iterrows():
    print(f"{sector:<24}{int(r['count']):>4}{r['median']:>10.1f}"
          f"{r['min']:>9.1f}{r['max']:>9.1f}")

# -- Chart ------------------------------------------------------------------
BG, FG = "#0a0a0a", "#e0e0e0"
COLOURS = {
    "Consumer Staples": "#3b82f6",
    "Consumer Discretionary": "#22d3ee",
    "Energy": "#a3e635",
    "Information Technology": "#a78bfa",
    "Industrials": "#fbbf24",
    "Health Care": "#f87171",
}

fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
ax.set_facecolor(BG)

plot = cycle.iloc[::-1]
positions = range(len(plot))
ax.barh(list(positions), plot["ccc"],
        color=[COLOURS[s] for s in plot["sector"]], height=0.72)
ax.set_yticks(list(positions))
ax.set_yticklabels(plot["ticker"], fontsize=8, color=FG)
ax.axvline(0, color=FG, linewidth=0.9, alpha=0.6)

ax.set_xlabel("Days of cash tied up in operations", color=FG, fontsize=10)
ax.set_title("How long cash stays tied up: cash conversion cycle by company",
             color=FG, fontsize=12, pad=12)
ax.tick_params(axis="x", colors=FG, labelsize=9)
ax.tick_params(axis="y", length=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#333333")

handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[s])
           for s in summary.index]
ax.legend(handles, list(summary.index), fontsize=8, loc="upper right",
          facecolor=BG, edgecolor="#333333", labelcolor=FG, framealpha=1.0)

plt.tight_layout()
plt.savefig("cash-conversion-cycle-analysis-python.png",
            dpi=150, facecolor=BG)
