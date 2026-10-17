# Full write-up: https://xfinlink.com/blog/how-long-after-ipo-sp500-python
#
# How long had each company been trading when it joined the S&P 500?
# Every addition effective 1990-2025, measured from its security's first daily bar,
# with each young listing classified by the route it took to the market.

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
warnings.filterwarnings("ignore", message="xfinlink: index_events")  # the loop below pages on purpose

# Second share classes of companies already in the index (Alphabet C, Under Armour C, Fox B, ...)
SECOND_CLASS = {10003590, 10032892, 10006698, 10003102, 10030753, 10004763, 10029970}
EARLIER_CLASS = {10025221: 10006133}  # Berkshire's index line (class B) -> class A, listed earlier
COVERAGE_START = {"1962-01-02", "1962-07-02", "1972-12-14"}  # daily bars begin on these dates

# Hand classification of every first entry listed under five years (ticker:year of addition).
# Default: under 0.1 years = spin-off, 0.1-5 years = own IPO or direct listing.
ROUTE = {
    "carve-out IPO": """SFR:1990 KRB:1992 DWD:1993 ATI:1994 FDC:1994 GLD:1994 ALL:1995 CSE:1996
        LU:1996 UPR:1996 GDT:1996 AFS:1998 COF:1998 DPH:1999 COC:1999 TSG:2000 A:2000 CVG:2000
        CIT:2000 PALM:2000 MIR:2001 PBG:2001 MON:2002 TAPB:2002 COH:2004 CIT:2004 FSLB:2004
        GNW:2005 AIZ:2007 MJN:2009 ZTS:2013 SYF:2015 CFG:2016 KVUE:2023""",
    "spin-off": """AGN:1992 LEH:1998 NCR:2000 CNXT:2000 RJR:2002 EXPE:2007 DPS:2008 DISCA:2010
        MNK:2014 FBHS:2016 KEYS:2018 LW:2018 SNDK:2025""",
    "other": "FRE:1992 VIA:1994 NDAQ:2008 TWC:2009",
    "reorganisation": """CVN:1990 X:1991 TMC:1995 FCX:1995 UMG:1995 PNU:1995 FD:1995 GRA:1996
        ROK:1996 TEN:1996 SEE:1998 MAR:1998 UMG:1998 SRE:1998 EDS:1998 PCS:1998 MCK:1999 WLP:1999
        DVN:2000 DYN:2000 AET:2000 FLR:2000 XL:2001 CMCSA:2002 NWSA:2004 SHLD:2005 HOT:2006
        DTV:2006 NYX:2007 IVZ:2008 ESV:2012 LYB:2012 PNR:2012 KHC:2015 FTI:2017 DWDP:2017
        LIN:2018 AMCR:2019 VTRS:2020 WBD:2022 KDP:2022 SW:2024 TPL:2024 EXE:2025 PSKY:2025 CRH:2025""",
}
ROUTE_OF = {k: r for r, keys in ROUTE.items() for k in keys.split()}


def events(kind):
    pages = [xfl.index_events("sp500", event_type=kind, start="1925-12-31")]
    while len(pages[-1]) == 1000:  # one call returns at most 1,000 rows
        pages.append(xfl.index_events("sp500", event_type=kind, start="1925-12-31", offset=1000 * len(pages)))
    return pd.concat(pages, ignore_index=True).dropna(subset=["entity_id"]).astype({"entity_id": int})


def first_bar(entity_id):
    p = xfl.prices(entity_id=EARLIER_CLASS.get(entity_id, entity_id), start="1962-01-01", limit=1, fields=["close"])
    return p["date"].iloc[0].strftime("%Y-%m-%d")


adds, rems = events("added"), events("removed")

# each addition's last day in the index; a stay under 30 days is a placement, not a membership
adds["last_day"] = [rems.loc[(rems["entity_id"] == r.entity_id) & (rems["effective_date"] >= r.effective_date),
                             "effective_date"].min() for r in adds.itertuples()]
adds["days_in"] = (pd.to_datetime(adds["last_day"]) - pd.to_datetime(adds["effective_date"])).dt.days
adds["placement"] = adds["days_in"] < 30
placement_exits = set(zip(adds.loc[adds["placement"], "entity_id"], adds.loc[adds["placement"], "last_day"]))
real_adds = adds[~adds["placement"]]
real_exits = rems[[(e, d) not in placement_exits for e, d in zip(rems["entity_id"], rems["effective_date"])]]

window = adds[adds["effective_date"].between("1990-01-01", "2025-12-31")].copy()
window["second_class"] = window["entity_id"].isin(SECOND_CLASS)
window["returning"] = [((real_adds["entity_id"] == r.entity_id) & (real_adds["effective_date"] < r.effective_date)).any()
                       or ((real_exits["entity_id"] == r.entity_id) & (real_exits["effective_date"] < r.effective_date)).any()
                       for r in window.itertuples()]
first = window[~window["second_class"] & ~window["placement"] & ~window["returning"]].copy()

with ThreadPoolExecutor(8) as ex:
    first["first_bar"] = list(ex.map(first_bar, first["entity_id"]))
first["years"] = ((pd.to_datetime(first["effective_date"]) - pd.to_datetime(first["first_bar"])).dt.days / 365.25).clip(lower=0)
first["before_start"] = first["first_bar"].isin(COVERAGE_START)  # listed before daily bars begin: age is a floor
first["key"] = first["ticker"] + ":" + first["effective_date"].str[:4]
young = (first["years"] < 5) & ~first["before_start"]
default = np.where(first["years"] < 0.1, "spin-off", "own IPO")
first["route"] = np.where(young, [ROUTE_OF.get(k, d) for k, d in zip(first["key"], default)], "listed 5y+")
first["decade"] = first["effective_date"].str[:3] + "0s"

reorg = first["route"] == "reorganisation"
sample = first[~reorg]
print(f"S&P 500 additions effective 1990-2025: {len(window)}")
print(f"  second share classes: {window['second_class'].sum()}; stays under 30 days: {window['placement'].sum()}; "
      f"securities returning to the index: {(window['returning'] & ~window['second_class'] & ~window['placement']).sum()}")
print(f"  first entries: {len(first)}; new securities from mergers and reorganisations set aside: {reorg.sum()}")
print(f"  analysed: {len(sample)}; first bar on a coverage start date (listed before it): {sample['before_start'].sum()}\n")

rows = []
for dec, g in sample.groupby("decade"):
    u2 = g[g["years"] < 2]
    rows.append({"decade": dec, "adds": len(g), "median_yrs": g["years"].median(),
                 "median_ex_spinoffs": g.loc[g["route"] != "spin-off", "years"].median(),
                 "min_floor": g.loc[g["before_start"], "years"].min(),
                 "under_2y": len(u2), "spin_off": (u2["route"] == "spin-off").sum(),
                 "carve_out": (u2["route"] == "carve-out IPO").sum(), "own_ipo": (u2["route"] == "own IPO").sum(),
                 "own_ipo_under_1y": ((g["years"] < 1) & (g["route"] == "own IPO")).sum(),
                 "20y_plus": ((g["years"] >= 20) | g["before_start"]).sum()})
table = pd.DataFrame(rows).set_index("decade")
print(table.round(1).to_string())
print(f"all decades: median {sample['years'].median():.1f} years, under 2 years {(sample['years'] < 2).mean():.1%}\n")

print("Own IPO or direct listing, joined within 12 months:")
print(sample[(sample["route"] == "own IPO") & (sample["years"] < 1)][["effective_date", "ticker", "first_bar", "years"]].round(2).to_string(index=False))
print("\nFastest own listings by decade:")
fast = sample[sample["route"] == "own IPO"].sort_values("years").groupby("decade").head(2)
print(fast.sort_values(["decade", "years"])[["decade", "effective_date", "ticker", "first_bar", "years"]].round(2).to_string(index=False))
print("\nJoined within 12 months, any route, since 2003:")
late = sample[(sample["effective_date"] >= "2003-01-01") & (sample["years"] < 1)]
print(late["route"].value_counts().to_string())

# ---- chart: share of each decade's additions by listing age at entry ----
import matplotlib.pyplot as plt

old = (sample["years"] >= 20) | sample["before_start"]
bands = [("Spin-off, under 2 years", (sample["years"] < 2) & (sample["route"] == "spin-off")),
         ("IPO or carve-out, under 2 years", (sample["years"] < 2) & (sample["route"] != "spin-off")),
         ("2-5 years", sample["years"].between(2, 5, inclusive="left")),
         ("5-10 years", sample["years"].between(5, 10, inclusive="left")),
         ("10-20 years", sample["years"].between(10, 20, inclusive="left") & ~old),
         ("20+ years, or listed before daily data", old)]
share = pd.DataFrame({name: mask.groupby(sample["decade"]).mean() * 100 for name, mask in bands})
print("\nShare of additions by listing age at entry (%):")
print(share.round(1).to_string())

colors = ["#3b82f6", "#93c5fd", "#6b7280", "#9ca3af", "#d1d5db", "#f3f4f6"]
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444"})
fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros(len(share))
labels = [f"{d}\n({n} adds)" for d, n in zip(share.index, table["adds"])]
for (name, _), c in zip(bands, colors):
    ax.barh(labels, share[name], left=left, color=c, label=name, height=0.6)
    for y, (l, v) in enumerate(zip(left, share[name])):
        if v >= 6:
            ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color="#0a0a0a" if c not in ("#3b82f6", "#6b7280") else "#ffffff")
    left += share[name].values
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of the decade's first-time additions (%)")
ax.set_title("How long after listing companies joined the S&P 500, 1990-2025")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("how-long-after-ipo-sp500-python.png", dpi=150)
