# Full write-up: https://xfinlink.com/blog/does-revenue-growth-require-borrowing-python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "does-revenue-growth-require-borrowing-python"
FIRST_FY, LAST_FY = 2019, 2024
EXCLUDED_SECTORS = ("Financials", "Real Estate")
FIELDS = [
    "revenue", "operating_cash_flow", "capital_expenditures",
    "dividends_paid_common", "share_repurchases", "total_debt", "gics_sector",
]


def load_panel():
    """Six fiscal years of cash-flow and debt data for the 2019 S&P 500 roster."""
    roster = xfl.index("sp500", as_of="2019-12-31").drop_duplicates("entity_id")
    ids = sorted(int(e) for e in roster["entity_id"].dropna())
    frames = [xfl.fundamentals(entity_id=ids[i:i + 50], period_type="annual",
                               start="2018-06-01", end="2025-12-31", fields=FIELDS)
              for i in range(0, len(ids), 50)]
    df = pd.concat(frames, ignore_index=True)
    # 52/53-week filers can report two rows under one fiscal_year label; keep the
    # later close so every company contributes exactly one row per fiscal year.
    df = df.sort_values(["entity_id", "fiscal_year", "period_end"])
    df = df.drop_duplicates(["entity_id", "fiscal_year"], keep="last")
    return len(ids), df


def build(df):
    """One row per company: growth, cash generated, and the change in debt."""
    panel = df[df["fiscal_year"].between(FIRST_FY, LAST_FY)].copy()
    panel = panel[~panel["gics_sector"].isin(EXCLUDED_SECTORS)]
    for col in ("dividends_paid_common", "share_repurchases"):
        panel[col] = panel[col].fillna(0.0)
    panel = panel.dropna(subset=["revenue", "operating_cash_flow",
                                 "capital_expenditures", "total_debt"])
    # A non-financial business cannot convert more cash than it books in sales,
    # so a year failing that test does not describe a fundable operation.
    panel = panel[panel["operating_cash_flow"] <= panel["revenue"]]
    years = panel.groupby("entity_id")["fiscal_year"].nunique()
    panel = panel[panel["entity_id"].isin(years[years == LAST_FY - FIRST_FY + 1].index)]

    flows = panel[panel["fiscal_year"] > FIRST_FY]
    firms = flows.groupby("entity_id").agg(
        ticker=("ticker", "first"), name=("entity_name", "first"),
        sector=("gics_sector", "first"), cum_revenue=("revenue", "sum"),
        cum_ocf=("operating_cash_flow", "sum"), cum_capex=("capital_expenditures", "sum"),
        cum_dividends=("dividends_paid_common", "sum"),
        cum_buybacks=("share_repurchases", "sum"))
    first = panel[panel["fiscal_year"] == FIRST_FY].set_index("entity_id")
    last = panel[panel["fiscal_year"] == LAST_FY].set_index("entity_id")
    firms["revenue_first"] = first["revenue"]
    firms["revenue_last"] = last["revenue"]
    firms["debt_change"] = last["total_debt"] - first["total_debt"]

    firms = firms[(firms["revenue_first"] > 0) & (firms["revenue_last"] > 0)]
    span = LAST_FY - FIRST_FY
    firms["cagr"] = (firms["revenue_last"] / firms["revenue_first"]) ** (1 / span) - 1
    # Cash the business kept after reinvesting and paying the dividend, and the
    # debt it raised, both as a share of the revenue booked over the same years.
    firms["retained"] = ((firms["cum_ocf"] - firms["cum_capex"] - firms["cum_dividends"])
                         / firms["cum_revenue"] * 100)
    firms["borrowed"] = firms["debt_change"] / firms["cum_revenue"] * 100
    firms["buybacks"] = firms["cum_buybacks"] / firms["cum_revenue"] * 100
    firms["quintile"] = pd.qcut(firms["cagr"], 5, labels=[1, 2, 3, 4, 5])
    return panel, firms


def summarise(firms):
    return firms.groupby("quintile", observed=True).agg(
        companies=("cagr", "size"), cagr=("cagr", "median"),
        retained=("retained", "median"), borrowed=("borrowed", "median"),
        buybacks=("buybacks", "median"),
        deficit=("retained", lambda s: (s < 0).mean() * 100))


def make_chart(table, sectors):
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
        "axes.edgecolor": "#333333", "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0", "font.size": 9,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))

    pos = np.arange(len(table))
    left.bar(pos - 0.2, table["retained"], width=0.4, color="#3b82f6",
             label="Cash kept after capex and dividends")
    left.bar(pos + 0.2, table["borrowed"], width=0.4, color="#9ca3af",
             label="Increase in total debt")
    left.axhline(0, color="#4b5563", linewidth=0.8)
    left.set_xticks(pos)
    left.set_xticklabels([f"Q{q}\n{c:.0%}" for q, c in zip(table.index, table["cagr"])])
    left.set_xlabel("Revenue growth quintile, median annual growth")
    left.set_ylabel("Percent of five-year revenue")
    left.set_title("Faster growers keep more cash", fontsize=10, fontweight="bold")
    left.legend(frameon=False, fontsize=8, loc="upper left")

    order = sectors.sort_values("borrowed")
    ypos = np.arange(len(order))
    right.barh(ypos, order["borrowed"], color="#3b82f6", height=0.6)
    right.axvline(0, color="#4b5563", linewidth=0.8)
    right.set_yticks(ypos)
    right.set_yticklabels(order.index)
    right.set_xlabel("Increase in total debt, percent of five-year revenue")
    right.set_title("Borrowing is a sector trait", fontsize=10, fontweight="bold")

    for axis in (left, right):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
    image = Image.open(f"{SLUG}.png").convert("RGB").quantize(colors=128)
    image.save(f"{SLUG}.png", optimize=True)


def main():
    roster_size, df = load_panel()
    panel, firms = build(df)
    assert not panel.duplicated(["entity_id", "fiscal_year"]).any()
    table = summarise(firms)

    print("=" * 74)
    print("DOES FAST REVENUE GROWTH FORCE A COMPANY TO BORROW?")
    print(f"S&P 500 roster at 31 Dec 2019, fiscal {FIRST_FY}-{LAST_FY}, "
          "ex Financials and Real Estate")
    print("=" * 74)
    print(f"entities on the point-in-time roster            {roster_size:>6}")
    print(f"complete six-year records in scope              {len(firms):>6}")
    print(f"company-years                                   {len(panel):>6}")

    print("\nMEDIANS BY REVENUE GROWTH QUINTILE, PERCENT OF FIVE-YEAR REVENUE")
    print(f"{'quintile':<10}{'companies':>11}{'rev CAGR':>11}{'cash kept':>12}"
          f"{'debt added':>13}{'buybacks':>11}{'in deficit':>13}")
    for q, row in table.iterrows():
        print(f"Q{q:<9}{row['companies']:>11.0f}{row['cagr'] * 100:>10.1f}%"
              f"{row['retained']:>11.1f}%{row['borrowed']:>12.1f}%"
              f"{row['buybacks']:>10.1f}%{row['deficit']:>12.1f}%")

    print("\nRANK CORRELATION WITH REVENUE GROWTH")
    for label, col in (("cash kept", "retained"), ("debt added", "borrowed")):
        rho, pval = stats.spearmanr(firms["cagr"], firms[col])
        within = stats.spearmanr(firms.groupby("sector")["cagr"].rank(pct=True),
                                 firms.groupby("sector")[col].rank(pct=True))
        print(f"{label:<12} all companies rho {rho:>+.3f} (p={pval:.4f})"
              f"   within sector rho {within[0]:>+.3f} (p={within[1]:.4f})")

    print("\nROBUSTNESS: SAME TABLE WITHOUT THE 20 LARGEST BY REVENUE")
    small = firms.drop(firms.nlargest(20, "cum_revenue").index).copy()
    small["quintile"] = pd.qcut(small["cagr"], 5, labels=[1, 2, 3, 4, 5])
    for q, row in summarise(small).iterrows():
        print(f"Q{q:<9}{row['companies']:>11.0f}{row['cagr'] * 100:>10.1f}%"
              f"{row['retained']:>11.1f}%{row['borrowed']:>12.1f}%"
              f"{row['buybacks']:>10.1f}%{row['deficit']:>12.1f}%")

    sectors = firms.groupby("sector").agg(
        companies=("cagr", "size"), cagr=("cagr", "median"),
        retained=("retained", "median"), borrowed=("borrowed", "median"))
    print("\nBY SECTOR, MEDIANS")
    print(f"{'sector':<26}{'companies':>11}{'rev CAGR':>11}{'cash kept':>12}{'debt added':>13}")
    for name, row in sectors.sort_values("borrowed", ascending=False).iterrows():
        print(f"{name:<26}{row['companies']:>11.0f}{row['cagr'] * 100:>10.1f}%"
              f"{row['retained']:>11.1f}%{row['borrowed']:>12.1f}%")

    print("\nFASTEST-GROWING QUINTILE, COMPANIES THAT STILL RAN A CASH DEFICIT")
    hungry = firms[(firms["quintile"] == 5) & (firms["retained"] < 0)]
    for _, row in hungry.sort_values("retained").iterrows():
        print(f"{row['ticker']:<6}{row['sector']:<24}growth {row['cagr'] * 100:>5.1f}%"
              f"   cash kept {row['retained']:>6.1f}%   debt added {row['borrowed']:>5.1f}%")

    make_chart(table, sectors)


if __name__ == "__main__":
    main()
