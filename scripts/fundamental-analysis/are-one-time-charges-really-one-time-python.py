# Full write-up: https://xfinlink.com/blog/are-one-time-charges-really-one-time-python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLUG = "are-one-time-charges-really-one-time-python"
FIRST_FY, LAST_FY = 2015, 2023
YEARS = LAST_FY - FIRST_FY + 1
MATERIALITY = 0.005  # charge must reach 0.5% of that year's revenue to count
FIELDS = ["restructuring_charges", "impairment_charges", "operating_income", "revenue"]


def load_panel():
    """Nine fiscal years of charge data for the point-in-time S&P 500 roster."""
    roster = xfl.index("sp500", as_of="2014-12-31")
    roster = roster.drop_duplicates("entity_id")
    ids = sorted(int(e) for e in roster["entity_id"].dropna())

    frames = [
        xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                         start="2014-06-01", end="2024-06-30", fields=FIELDS)
        for i in range(0, len(ids), 100)
    ]
    df = pd.concat(frames, ignore_index=True)

    # Label each filing by fiscal year, then keep one filing per company-year.
    df["fiscal"] = df["period_end"].dt.year - (df["period_end"].dt.month <= 5).astype(int)
    df = df.sort_values("period_end").drop_duplicates(["entity_id", "fiscal"], keep="last")

    panel = df[(df["fiscal"] >= FIRST_FY) & (df["fiscal"] <= LAST_FY)]
    panel = panel.dropna(subset=["revenue", "operating_income"])
    panel = panel[panel["revenue"] > 0]

    complete = panel.groupby("entity_id").size()
    panel = panel[panel["entity_id"].isin(complete[complete == YEARS].index)].copy()
    panel["roster_ticker"] = panel["entity_id"].map(
        roster.set_index("entity_id")["ticker"])
    return panel


def score(panel):
    """Count material charge years and total the charges per company."""
    restructuring = panel["restructuring_charges"].clip(lower=0).fillna(0)
    impairment = panel["impairment_charges"].clip(lower=0).fillna(0)

    # The larger of the two, never the sum: some filers report one combined
    # figure under both labels, and adding them would count it twice.
    panel["charge"] = np.maximum(restructuring, impairment)
    panel["charge_pct_revenue"] = panel["charge"] / panel["revenue"]
    panel["material"] = panel["charge_pct_revenue"] > MATERIALITY

    firms = panel.groupby("entity_id").agg(
        ticker=("roster_ticker", "last"),
        name=("entity_name", "last"),
        sector=("gics_sector", "last"),
        charge_years=("material", "sum"),
        total_charges=("charge", "sum"),
        total_operating_income=("operating_income", "sum"),
        total_revenue=("revenue", "sum"),
    )
    pre_charge = firms["total_operating_income"] + firms["total_charges"]
    firms["drag"] = np.where(pre_charge > 0, firms["total_charges"] / pre_charge, np.nan)
    return firms


def make_chart(firms, buckets):
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
        "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#3a3a3a", "font.size": 10,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))

    counts = firms["charge_years"].value_counts().reindex(range(YEARS + 1), fill_value=0)
    left.bar(counts.index, counts.values, color="#3b82f6", width=0.75)
    left.set_xlabel("Years with a charge, out of nine")
    left.set_ylabel("Number of companies")
    left.set_title("How often companies take a charge", fontsize=11, pad=10)
    left.set_xticks(range(YEARS + 1))
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)

    right.bar(range(len(buckets)), buckets["median_drag"] * 100,
              color="#3b82f6", width=0.65)
    right.set_xticks(range(len(buckets)))
    right.set_xticklabels(buckets.index)
    right.set_xlabel("Years with a charge, out of nine")
    right.set_ylabel("Share of operating profit charged off (%)")
    right.set_title("What the charges cost over nine years", fontsize=11, pad=10)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
    plt.close()


def main():
    panel = load_panel()
    firms = score(panel)
    charged = firms[firms["charge_years"] > 0]

    labels = ["0", "1-2", "3-4", "5-6", "7-9"]
    ranges = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 9)]
    buckets = pd.DataFrame([
        {"companies": len(sub),
         "median_drag": sub["drag"].median(),
         "median_charge_pct_revenue": (sub["total_charges"] / sub["total_revenue"]).median()}
        for lo, hi in ranges
        for sub in [firms[(firms["charge_years"] >= lo) & (firms["charge_years"] <= hi)]]
    ], index=labels)

    print("=" * 74)
    print("HOW OFTEN DO S&P 500 COMPANIES TAKE A 'ONE-TIME' CHARGE?")
    print(f"Roster at 31 Dec 2014, fiscal {FIRST_FY}-{LAST_FY}, "
          f"charge counted at >{MATERIALITY:.1%} of revenue")
    print("=" * 74)
    print(f"companies with a complete nine-year record   {len(firms):>6}")
    print(f"company-years                                {len(panel):>6}")
    print(f"took at least one charge                     {len(charged):>6}"
          f"  ({len(charged) / len(firms):.0%})")
    print(f"median charge years among those              {charged['charge_years'].median():>6.0f}")
    print(f"charged in five or more of nine years        {(charged['charge_years'] >= 5).sum():>6}")
    print(f"charged in seven or more of nine years       {(charged['charge_years'] >= 7).sum():>6}")
    print(f"charged in all nine years                    {(charged['charge_years'] == 9).sum():>6}")

    print("\nCHARGE FREQUENCY VERSUS WHAT THE CHARGES COST")
    print(f"{'years charged':<16}{'companies':>11}{'median cost':>14}{'as % revenue':>15}")
    for label, row in buckets.iterrows():
        print(f"{label:<16}{row['companies']:>11.0f}{row['median_drag'] * 100:>13.1f}%"
              f"{row['median_charge_pct_revenue'] * 100:>14.2f}%")

    print("\nBY SECTOR")
    sector = firms.groupby("sector").agg(
        companies=("charge_years", "size"),
        ever_charged=("charge_years", lambda s: (s > 0).sum()),
        median_years=("charge_years", lambda s: s[s > 0].median()),
        median_cost=("drag", "median"),
    ).sort_values("median_cost", ascending=False)
    print(f"{'sector':<26}{'companies':>10}{'ever':>7}{'median yrs':>12}{'median cost':>13}")
    for name, row in sector.iterrows():
        print(f"{name:<26}{row['companies']:>10.0f}{row['ever_charged']:>7.0f}"
              f"{row['median_years']:>12.0f}{row['median_cost'] * 100:>12.1f}%")

    print("\nSENSITIVITY TO THE MATERIALITY FLOOR")
    print(f"{'floor, % of revenue':<22}{'ever charged':>14}{'median years':>14}{'seven or more':>15}")
    for floor in (0.005, 0.01, 0.02):
        years = (panel["charge_pct_revenue"] > floor).groupby(panel["entity_id"]).sum()
        hit = years[years > 0]
        print(f"{floor:<22.1%}{len(hit):>14}{hit.median():>14.0f}{(hit >= 7).sum():>15}")

    print("\nCHARGED IN ALL NINE YEARS")
    every_year = firms[firms["charge_years"] == 9].sort_values("total_charges", ascending=False)
    print(", ".join(every_year["ticker"].dropna().tolist()))

    print("\nNEVER TOOK A MATERIAL CHARGE, LARGEST BY NINE-YEAR REVENUE")
    never = firms[firms["charge_years"] == 0].sort_values("total_revenue", ascending=False)
    print(", ".join(never["ticker"].dropna().head(12).tolist()))

    make_chart(firms, buckets)


if __name__ == "__main__":
    main()
