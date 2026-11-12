# Full write-up: https://xfinlink.com/blog/portfolio-optimization-survivorship-bias-python
#
# Does a portfolio optimizer make survivorship bias bigger or smaller than equal weighting?
# Annual rebalances on the first trading day of each January, 2000-2024. Each rule (equal weight,
# long-only minimum variance, long-only maximum Sharpe, Ledoit-Wolf covariance, 5% cap per name)
# runs on the S&P 500 roster of the rebalance date and on the end-2024 list used at every date.
# Buy and hold for 12 months on monthly total returns; a holding that stops trading gets its
# recorded delisting return, then the proceeds follow SPY to the year end.
import multiprocessing as mp
import os
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"   # one thread per worker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

HIST, CAP = 60, 0.05            # months of history a name needs; per-name weight cap
YEARS = range(2000, 2025)
CHART = "portfolio-optimization-survivorship-bias-python.png"

# ---- data ---------------------------------------------------------------------------------
spy = xfl.prices("SPY", start="1994-12-01", end="2024-12-31", interval="1mo", fields=["return_daily"])
jan = spy.loc[(spy["date"].dt.month == 1) & spy["date"].dt.year.between(2000, 2024), "date"]
roster = {d.year: set(xfl.index("sp500", as_of=d.date().isoformat())["entity_id"].dropna().astype(int))
          for d in jan}                                                   # members on each rebalance date
end24 = set(xfl.index("sp500", as_of="2024-12-31")["entity_id"].dropna().astype(int))

ids = sorted(set().union(*roster.values(), end24))
px = pd.concat([xfl.prices(entity_id=ids[i:i + 150], start="1995-01-01", end="2024-12-31", interval="1mo",
                           fields=["return_daily"], max_rows=400_000) for i in range(0, len(ids), 150)])
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").sort_index()
last = px.dropna(subset=["return_daily"]).groupby("entity_id")["m"].max()   # last priced month
spy_r = spy.assign(m=spy["date"].dt.to_period("M")).set_index("m")["return_daily"]

# Exit records (meta.delisting) for series that end before December 2024, one call per exit year
ended = last[(last >= pd.Period("2000-01", "M")) & (last < pd.Period("2024-12", "M"))]
delist = {}
for y, g in ended.groupby(ended.dt.year):
    g = g.index.tolist()
    for i in range(0, len(g), 150):
        d = xfl.prices(entity_id=g[i:i + 150], start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v.get("delisting_return") for k, v in (d.attrs["meta"].get("delisting") or {}).items()})


# ---- estimation and optimisation ----------------------------------------------------------
def ledoit_wolf(X):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity."""
    X = X - X.mean(0)
    T, N = X.shape
    S = X.T @ X / T
    m = np.trace(S) / N
    d2 = ((S - m * np.eye(N)) ** 2).sum() / N
    b2 = min(((np.einsum("ti,ti->t", X, X) ** 2).sum() - T * (S ** 2).sum()) / (T * T * N), d2)
    return b2 / d2 * m * np.eye(N) + (1 - b2 / d2) * S


def project(v, cap):
    """Euclidean projection onto {0 <= w <= cap, sum(w) = 1}, by bisection on the shift."""
    lo, hi = v.min() - 1, v.max()
    for _ in range(50):
        t = (lo + hi) / 2
        lo, hi = (t, hi) if np.clip(v - t, 0, cap).sum() > 1 else (lo, t)
    return np.clip(v - (lo + hi) / 2, 0, cap)


def frontier(S, mu, lam, cap, iters=3000):
    """Accelerated projected gradient for max mu'w - lam/2 w'Sw on the capped simplex."""
    step = 1 / (lam * np.linalg.eigvalsh(S)[-1])
    w = z = np.ones(len(S)) / len(S)
    t = 1.0
    for _ in range(iters):
        w1 = project(z - step * (lam * S @ z - mu), cap)
        t1 = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z, w, t = w1 + (t - 1) / t1 * (w1 - w), w1, t1
    return w


def min_var(S, cap):
    return frontier(S, np.zeros(len(S)), 1.0, cap)


def max_sharpe(mu, S, cap):
    """Highest mu'w / sqrt(w'Sw) on the capped-simplex frontier: grid, then golden section on log(lam)."""
    f = lambda x: (lambda w: (mu @ w) / np.sqrt(w @ S @ w))(frontier(S, mu, np.exp(x), cap, 1500))
    grid = np.linspace(-2, 9, 23)
    i = int(np.argmax([f(x) for x in grid]))
    assert 0 < i < len(grid) - 1, "optimum at the edge of the search grid"
    a, b, g = grid[i - 1], grid[i + 1], (5 ** 0.5 - 1) / 2
    for _ in range(14):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if f(c) > f(d) else (c, b)
    return frontier(S, mu, np.exp((a + b) / 2), cap)


def hold(ids, w, y):
    """Buy and hold for 12 months. A holding that stops trading gets its delisting return,
    then its proceeds follow SPY; a month with no recorded close carries it at its last close."""
    months = pd.period_range(f"{y}-01", f"{y}-12", freq="M")
    H = R.reindex(index=months, columns=ids)
    V, info = [], []
    for e in ids:
        k = (last[e] - months[0]).n if last[e] < months[-1] else 11       # last priced month
        v = np.cumprod(1 + np.nan_to_num(H[e].to_numpy()[:k + 1]))
        dr = delist.get(e) if k < 11 else None
        if k < 11:
            v[-1] *= 1 + (dr if dr is not None and not np.isnan(dr) else 0.0)
            v = np.r_[v, v[-1] * np.cumprod(1 + spy_r.reindex(months[k + 1:]).to_numpy())]
        V.append(v)
        info.append((e, k < 11, dr, v[-1] - 1))
    P = np.r_[1.0, w @ np.array(V)]
    return P[1:] / P[:-1] - 1, pd.DataFrame(info, columns=["id", "exit", "dr", "yr"]).assign(w=w)


UNIV = {"Point-in-time": lambda y: roster[y], "End-2024 list": lambda y: end24,
        "Members then, in end-2024 list": lambda y: roster[y] & end24}
RULES = ["Equal weight", "Minimum variance", "Maximum Sharpe"]


def run(task):
    y, u = task
    est = R.loc[pd.Period(f"{y - 1}-12", "M") - HIST + 1: pd.Period(f"{y - 1}-12", "M")]
    ids = sorted(e for e in UNIV[u](y) if e in est.columns and est[e].notna().all()
                 and last[e] >= pd.Period(f"{y}-01", "M"))                # full history, trades in January
    X = est[ids].to_numpy()
    S, mu = ledoit_wolf(X), X.mean(0)
    book = {"Equal weight": np.ones(len(ids)) / len(ids), "Minimum variance": min_var(S, CAP),
            "Maximum Sharpe": max_sharpe(mu, S, CAP)}
    return [((u, rule, y), len(ids), *hold(ids, w, y)) for rule, w in book.items()]


if __name__ == "__main__":
    tasks = [(y, u) for y in YEARS for u in UNIV]
    if "fork" in mp.get_all_start_methods():                              # workers share the data above
        with mp.get_context("fork").Pool() as pool:
            parts = pool.map(run, tasks)
    else:                                                                 # Windows: one process, slower
        parts = map(run, tasks)
    res = [o for part in parts for o in part]
    monthly = {k: r for k, _, r, _ in res}
    size = pd.DataFrame([(k[0], n) for k, n, _, _ in res], columns=["u", "n"])
    Hd = pd.concat([h.assign(u=k[0], rule=k[1], year=k[2]) for k, _, _, h in res], ignore_index=True)
    Hd["left"] = ~Hd["id"].isin(end24)                                    # not in the end-2024 list
    Hd["notyet"] = [e not in roster[y] for e, y in zip(Hd["id"], Hd["year"])]

    def stats(u, rule):
        r = np.concatenate([monthly[(u, rule, y)] for y in YEARS])
        yr = np.array([np.prod(1 + monthly[(u, rule, y)]) - 1 for y in YEARS])
        return np.prod(1 + r) ** (12 / len(r)) - 1, r.std(ddof=1) * 12 ** 0.5, r.mean() / r.std(ddof=1) * 12 ** 0.5, yr

    tstat = lambda d: d.mean() / d.std(ddof=1) * len(d) ** 0.5
    S = {(u, rule): stats(u, rule) for u in UNIV for rule in RULES}
    gap = {rule: S[("End-2024 list", rule)][3] - S[("Point-in-time", rule)][3] for rule in RULES}
    bias = {rule: S[("End-2024 list", rule)][0] - S[("Point-in-time", rule)][0] for rule in RULES}

    # ---- output -----------------------------------------------------------------------------
    rng = lambda u: size[size.u == u]["n"]
    print(f"Rebalances: first trading day of January, {YEARS[0]}-{YEARS[-1]}; {HIST} monthly returns required; "
          f"cap {CAP:.0%} a name; Sharpe at a zero risk-free rate")
    print(f"Names per rebalance: point-in-time {rng('Point-in-time').min()}-{rng('Point-in-time').max()}, "
          f"end-2024 list {rng('End-2024 list').min()}-{rng('End-2024 list').max()}\n")
    print(f"{'':18}{'point-in-time rosters':>24}{'end-2024 list':>24}   {'bias, points a year':>27}")
    print(f"{'':18}" + f"{'return':>9}{'vol':>8}{'Sharpe':>7}" * 2 + f"{'annual':>10}{'mean gap':>10}{'t':>6}{'ahead':>7}")
    for rule in RULES:
        a, b = S[("Point-in-time", rule)], S[("End-2024 list", rule)]
        print(f"{rule:18}{a[0]:9.2%}{a[1]:8.2%}{a[2]:7.3f}{b[0]:9.2%}{b[1]:8.2%}{b[2]:7.3f}"
              f"{bias[rule] * 100:+10.2f}{gap[rule].mean() * 100:+10.2f}{tstat(gap[rule]):6.2f}{(gap[rule] > 0).sum():>4}/25")
    print("Bias against equal weight's, year by year: " + ", ".join(
        f"{r[0].lower() + r[1:]} {(gap[r] - gap['Equal weight']).mean() * 100:+.2f} (t = {tstat(gap[r] - gap['Equal weight']):.2f})"
        for r in RULES[1:]))
    print("\nBias split, points a year: dropping members later removed / adding companies not yet members")
    for rule in RULES:
        mid = S[("Members then, in end-2024 list", rule)][0]
        print(f"  {rule:18}{(mid - S[('Point-in-time', rule)][0]) * 100:+6.2f} / {(S[('End-2024 list', rule)][0] - mid) * 100:+.2f}")
    print("\nWeight on each slice (mean over 25 rebalances) and its weighted mean 12-month return")
    for rule in RULES:
        cells = []
        for u, col in [("Point-in-time", "left"), ("End-2024 list", "notyet")]:
            d = Hd[(Hd.u == u) & (Hd.rule == rule) & Hd[col]]
            w = d.groupby("year")["w"].sum().reindex(YEARS, fill_value=0)
            cells.append(f"{w.mean():6.1%}  {(d.w * d.yr).sum() / d.w.sum():+6.1%}")
        print(f"  {rule:18}point-in-time, later removed {cells[0]}   end-2024 list, not yet members {cells[1]}")
    ex = Hd[(Hd.u == "Point-in-time") & Hd.exit].drop_duplicates(["id", "year"])
    print(f"\nPoint-in-time holdings that stopped trading inside their year: {len(ex)}, "
          f"{ex.dr.notna().sum()} with a recorded delisting return")

    # ---- chart: annualised return per rule, both universes ----------------------------------
    plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                         "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                         "axes.edgecolor": "#444444", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(RULES))
    for off, u, colour, label in [(-0.2, "Point-in-time", "#3b82f6", "S&P 500 roster on each rebalance date"),
                                  (0.2, "End-2024 list", "#d97706", "End-2024 list used at every date")]:
        v = [S[(u, r)][0] * 100 for r in RULES]
        ax.bar(x + off, v, width=0.38, color=colour, label=label, edgecolor="#0a0a0a", linewidth=2)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.2, f"{vi:.1f}%", ha="center", va="bottom", fontsize=10)
    for xi, r in zip(x, RULES):
        ax.text(xi, S[("End-2024 list", r)][0] * 100 + 1.4, f"bias +{bias[r] * 100:.1f} pts", ha="center",
                color="#e0e0e0", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Equal weight", "Minimum variance", "Maximum Sharpe"])
    ax.set_ylabel("Annualised total return, 2000-2024 (%)")
    ax.set_ylim(0, 18)
    ax.set_title("Portfolio optimization backtest: survivorship bias by weighting rule")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHART, dpi=150)
