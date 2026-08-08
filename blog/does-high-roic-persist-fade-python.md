**Do High Returns on Capital Persist? ROIC Fade Analysis in Python**

August 8, 2026 · PROFITABILITY-ANALYSIS

**What's the question?**

Return on invested capital measures the operating profit a company earns for each dollar of debt and equity put to work. A business earning 25% on capital is doing something its competitors cannot copy, at least for now. Economic theory says that will not last: high returns attract entry, entry competes the returns away, and profitability drifts back toward the cost of capital. The rate at which this happens is called fade.

Fade is not an academic curiosity. Every discounted cash flow model contains an assumption about it, usually buried in the terminal value, and an analyst who assumes a company holds 25% returns forever values it at a multiple of one who assumes decay to 10% over a decade. Quality-factor strategies make the same bet, buying high-return businesses on the premise that the return survives long enough to be paid for.

So the question is quantitative rather than directional. Everyone agrees returns fade. How fast, how completely, and how reliably is what a valuation actually depends on.

**The approach**

1. Take the S&P 500 as constituted on 31 December 2015, using membership as of that date rather than today's roster.
2. Sort the companies into quintiles by return on invested capital in 2015, then follow the median of each quintile forward through 2024 without re-sorting.
3. Rank the companies again on 2024 returns and build a transition matrix, which shows where each starting quintile ended up.

The point-in-time roster is what makes the answer trustworthy. Ranking today's index members and looking backward keeps only companies that survived the decade as large-cap public entities, disproportionately the ones whose returns held up, and the fade would look far gentler than it was.

Medians rather than means throughout, since return on invested capital has a long tail in both directions and a handful of extreme values would otherwise drive the result.

**Code**

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

members = xfl.index("sp500", as_of="2015-12-31")
ids = sorted(members["entity_id"].dropna().astype(int).unique())

CHUNK = 150
m = pd.concat([
    xfl.metrics(entity_id=ids[i:i + CHUNK], period_type="annual",
                start="2015-01-01", end="2024-12-31", fields=["roic"],
                max_rows=50000)
    for i in range(0, len(ids), CHUNK)], ignore_index=True)

m["year"] = pd.to_datetime(m["period_end"]).dt.year
panel = m.pivot_table(index="entity_id", columns="year", values="roic")

base = panel[2015].dropna()
q = pd.qcut(base, 5, labels=[1, 2, 3, 4, 5])
for lab in [5, 4, 3, 2, 1]:
    ent = q[q == lab].index
    path = [panel.loc[panel.index.intersection(ent), y].median() * 100
            for y in range(2015, 2025)]
    print(f"Q{lab}: " + "".join(f"{v:6.1f}%" for v in path))

# where each 2015 quintile sat in 2024
end = panel[2024].dropna()
both = base.index.intersection(end.index)
trans = pd.crosstab(pd.qcut(base[both], 5, labels=[1, 2, 3, 4, 5]),
                    pd.qcut(end[both], 5, labels=[1, 2, 3, 4, 5]),
                    normalize="index") * 100
```

Full script with formatting and visualisation: [does-high-roic-persist-fade-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/fundamental-analysis/does-high-roic-persist-fade-python.py)

**Output**

<CHART>

```
495 companies in the index at 2015-12-31
479 companies with at least one annual figure, 2015 to 2024
465 ranked on 2015 return on invested capital

quintile     n    2015   2016   2017   2018   2019   2020   2021   2022   2023   2024
Q5          93   24.5%  21.8%  19.6%  22.9%  20.7%  15.6%  19.3%  18.7%  17.5%  15.6%
Q4          93   14.2%  13.7%  13.4%  14.8%  14.9%  12.2%  14.7%  15.9%  13.7%  13.2%
Q3          93   10.1%   9.7%  10.2%  11.3%  11.2%   8.2%  10.8%  12.1%  11.3%  10.7%
Q2          93    6.6%   6.9%   6.9%   7.7%   7.1%   5.9%   8.1%   7.5%   7.3%   6.8%
Q1          93    1.9%   3.8%   5.7%   6.7%   5.7%   3.6%   7.1%   6.9%   9.7%   7.1%

median return on invested capital, %, by 2015 quintile

top-minus-bottom spread: 22.6pp in 2015, 8.5pp in 2024 (37% of the original)
top quintile median fell 8.9pp; bottom quintile median rose 5.2pp

398 companies ranked in both 2015 and 2024 (86% of the original cross-section)

where each 2015 quintile sat in 2024 (row %, quintile 5 = highest)
2015            Q1      Q2      Q3      Q4      Q5
Q1           27.5%   30.0%   18.8%   17.5%    6.2%
Q2           25.3%   34.2%   22.8%   10.1%    7.6%
Q3           12.5%   22.5%   21.2%   25.0%   18.8%
Q4           20.3%    6.3%   22.8%   25.3%   25.3%
Q5           15.0%    6.2%   15.0%   21.2%   42.5%

42.5% of the 2015 top quintile was still top quintile in 2024
rank correlation 2015 vs 2024: 0.332
```

**What this tells us**

The spread between the best and worst quintiles fell from 22.6 points to 8.5 over nine years, leaving 37% of the original gap. Fade is real and substantial. It is also incomplete: the top quintile still earned 15.6% in 2024 against 7.1% for the bottom, so a ranking made in 2015 retained genuine information about profitability nine years later.

Most of the convergence came from the top falling rather than the bottom rising: the top quintile median dropped 8.9 points against 5.2 for the bottom. Part of that bottom-quintile improvement is composition rather than recovery, since 86% of the starting cross-section still reported in 2024 and companies that struggle for a decade are the ones most likely to be acquired or taken private.

Persistence at the top beats chance and falls well short of a guarantee. Of the companies starting in the highest quintile, 42.5% were still there in 2024, more than double the 20% random reshuffling would produce. The rank correlation across the full cross-section is 0.332.

Deterioration is bimodal rather than gradual. Companies leaving the top quintile either slipped one place, to the fourth quintile at 21.2%, or fell the whole way to the bottom at 15.0%. Landing in the second quintile, the intermediate outcome, happened to only 6.2%. Businesses tend to hold roughly their position or to break, and the break is severe when it comes: Boeing went from 35.8% in 2015 to -19.4% in 2024, Gilead from 47.1% to 2.8% as its hepatitis C franchise ran off, and Fossil from 19.5% to -29.0%.

The 2020 column shows what a shared shock looks like against this pattern. Every quintile fell together and every quintile recovered by 2021, which is the signature of a cyclical hit rather than competitive erosion. The fade in the top quintile is visible before 2020 and continues after it.

**So what?**

Anchor terminal-value assumptions to the observed rate rather than to a company's current return. Nine years took the top quintile from 24.5% to 15.6%, roughly a point a year, and the surviving advantage settled at about double the bottom quintile rather than at parity. A model that fades to the cost of capital within five years is too aggressive on this evidence; one that holds the current return indefinitely is far more wrong in the other direction.

For quality strategies, the transition matrix argues for rebalancing rather than buying and holding. A 42.5% retention rate over nine years means a static high-return portfolio has more than half its positions in businesses that no longer qualify, and the shape of the exits matters as much as the rate: roughly a quarter of the companies that left the top quintile went straight to the bottom.

The practical screen follows from that asymmetry: track the change in return on capital rather than its level alone, since companies leave the top quickly enough that an annual re-rank catches them while a five-year holding period does not. Running this panel on any starting universe takes one metrics pull and gives a fade rate specific to the sector being valued.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*
