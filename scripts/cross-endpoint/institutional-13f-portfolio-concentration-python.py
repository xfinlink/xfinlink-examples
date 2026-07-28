# Full write-up: https://xfinlink.com/blog/institutional-13f-portfolio-concentration-python
"""How concentrated are the largest institutional equity portfolios?

Top-ten weight and the effective number of positions, computed from reported
Form 13F portfolios at 30 June of every year from 2011 to 2025, for four
index-tracking managers and four managers that pick stocks.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# display name -> (managers() search term, exact manager_name, style)
MANAGERS = {
    "Vanguard":        ("vanguard group", "VANGUARD GROUP INC", "index"),
    "BlackRock":       ("blackrock inc", "BLACKROCK INC", "index"),
    "State Street":    ("state street", "STATE STREET CORP", "index"),
    "Geode":           ("geode", "GEODE CAPITAL MANAGEMENT, LLC", "index"),
    "Fidelity":        ("fmr", "FMR LLC", "stock picker"),
    "Wellington":      ("wellington management", "WELLINGTON MANAGEMENT GROUP LLP", "stock picker"),
    "Capital Research": ("capital research", "Capital Research Global Investors", "stock picker"),
    "Berkshire":       ("berkshire hathaway", "Berkshire Hathaway Inc", "stock picker"),
}
QUARTERS = [f"{y}-06-30" for y in range(2011, 2026)]

# ── Resolve manager ids ───────────────────────────────────────────────
ids = {}
for display, (term, exact, _style) in MANAGERS.items():
    found = xfl.managers(search=term)
    row = found[found["manager_name"] == exact].iloc[0]
    ids[display] = int(row["manager_id"])

# ── Concentration of each reported portfolio, year by year ────────────
def concentration(df):
    """Issuer-level weights for one manager and quarter."""
    eq = df[df["put_call"].isna()]              # common stock, not option lines
    # One issuer can be held through several securities (two Alphabet classes,
    # two Berkshire classes). Group on entity_id, never on ticker.
    value = eq.groupby("entity_id")["value_usd"].sum().sort_values(ascending=False)
    w = value / value.sum()
    hhi = float((w ** 2).sum())
    return dict(issuers=len(w), total_usd=float(value.sum()),
                top1=float(w.iloc[0]), top10=float(w.head(10).sum()),
                hhi=hhi * 10000, eff_n=1.0 / hhi)

rows = []
for display, mid in ids.items():
    for q in QUARTERS:
        df = xfl.manager_holdings(manager_id=mid, quarter=q, max_rows=20000)
        if df.empty:
            continue
        rows.append(dict(manager=display, style=MANAGERS[display][2],
                         year=int(q[:4]), **concentration(df)))

panel = pd.DataFrame(rows)
top10 = panel.pivot(index="year", columns="manager", values="top10")
effn = panel.pivot(index="year", columns="manager", values="eff_n")
order = list(MANAGERS)
top10, effn = top10[order], effn[order]

index_mgrs = [m for m, v in MANAGERS.items() if v[2] == "index"]
picker_mgrs = [m for m, v in MANAGERS.items() if v[2] == "stock picker"]

# ── Output ────────────────────────────────────────────────────────────
print("Reported Form 13F portfolios, 30 June 2011-2025, issuer-level weights")
print(f"Managers: {len(ids)}   quarters: {len(QUARTERS)}   "
      f"portfolio-quarters: {len(panel)}\n")

print("SIZE AND BREADTH AT 2025-06-30")
print(f"{'manager':17s} {'style':13s} {'issuers':>8s} {'reported $bn':>13s} {'largest position':>18s}")
last = panel[panel["year"] == 2025].set_index("manager")
for m in order:
    r = last.loc[m]
    print(f"{m:17s} {r['style']:13s} {r['issuers']:8.0f} {r['total_usd']/1e9:13,.0f} "
          f"{r['top1']*100:17.1f}%")

print("\nTOP-TEN WEIGHT (% of reported portfolio value)")
print(f"{'year':5s}" + "".join(f"{m[:12]:>13s}" for m in order))
for y, r in top10.iterrows():
    print(f"{y:5d}" + "".join(f"{r[m]*100:12.1f}%" for m in order))

print("\nEFFECTIVE NUMBER OF POSITIONS (1 / sum of squared weights)")
print(f"{'year':5s}" + "".join(f"{m[:12]:>13s}" for m in order))
for y, r in effn.iterrows():
    print(f"{y:5d}" + "".join(f"{r[m]:13.1f}" for m in order))

print("\nCHANGE 2011 -> 2025")
print(f"{'manager':17s} {'top-10 2011':>12s} {'top-10 2025':>12s} {'change':>9s} "
      f"{'eff. N 2011':>12s} {'eff. N 2025':>12s}")
for m in order:
    print(f"{m:17s} {top10.loc[2011, m]*100:11.1f}% {top10.loc[2025, m]*100:11.1f}% "
          f"{(top10.loc[2025, m]-top10.loc[2011, m])*100:+8.1f} "
          f"{effn.loc[2011, m]:12.1f} {effn.loc[2025, m]:12.1f}")

g_idx = top10[index_mgrs].mean(axis=1)
g_pick = top10[picker_mgrs].mean(axis=1)
print(f"\nGroup mean top-ten weight   index-tracking: {g_idx.loc[2011]*100:.1f}% "
      f"-> {g_idx.loc[2025]*100:.1f}%   stock pickers: {g_pick.loc[2011]*100:.1f}% "
      f"-> {g_pick.loc[2025]*100:.1f}%")
print(f"Gap between the two groups: {(g_pick.loc[2011]-g_idx.loc[2011])*100:+.1f} pts "
      f"in 2011, {(g_pick.loc[2025]-g_idx.loc[2025])*100:+.1f} pts in 2025")
ex_brk = [m for m in picker_mgrs if m != "Berkshire"]
g_pick2 = top10[ex_brk].mean(axis=1)
print(f"Stock pickers excluding Berkshire: {g_pick2.loc[2011]*100:.1f}% "
      f"-> {g_pick2.loc[2025]*100:.1f}%")

# ── Chart ─────────────────────────────────────────────────────────────
BG, FG, ACCENT = "#0a0a0a", "#e0e0e0", "#3b82f6"
PICKER_COLOUR = {"Fidelity": "#f59e0b", "Wellington": "#22c55e",
                 "Capital Research": "#a855f7", "Berkshire": "#ef4444"}
fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
for col, ax, title, ylab in [
        (top10, axes[0], "Weight in the ten largest holdings",
         "Share of reported portfolio value (%)"),
        (effn, axes[1], "Effective number of positions",
         "Positions (equal-weight equivalent)")]:
    ax.set_facecolor(BG)
    for m in order:
        y = col[m] * 100 if col is top10 else col[m]
        if MANAGERS[m][2] == "index":
            # The four trackers sit almost on top of each other, so they share
            # one colour and one legend entry.
            first = col is top10 and m == index_mgrs[0]
            ax.plot(col.index, y, color=ACCENT, linewidth=2.4, alpha=0.55,
                    label="Index-tracking (4 firms)" if first else None)
        else:
            ax.plot(col.index, y, color=PICKER_COLOUR[m], linewidth=1.6,
                    linestyle="--", label=m if col is top10 else None)
    ax.set_yscale("log")
    ax.set_title(title, color=FG, fontsize=11)
    ax.set_ylabel(ylab, color=FG, fontsize=9)
    ax.set_xlabel("Portfolio as reported at 30 June", color=FG, fontsize=9)
    ax.tick_params(colors=FG, labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for spine in ax.spines.values():
        spine.set_color("#333333")
axes[0].set_yticks([15, 20, 30, 50, 100])
axes[1].set_yticks([5, 10, 30, 100, 250])
axes[0].legend(fontsize=7, facecolor=BG, edgecolor="#333333", labelcolor=FG,
               loc="center left", ncol=2, framealpha=0.95)
fig.suptitle("Concentration roughly doubled, for index funds and stock pickers alike",
             color=FG, fontsize=13)
plt.tight_layout()
plt.savefig("institutional-13f-portfolio-concentration-python.png", dpi=150,
            facecolor=BG)
