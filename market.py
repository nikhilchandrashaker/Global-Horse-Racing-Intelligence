"""Module E — Odds & Market Efficiency Lab (final-odds edition).

What this can and cannot do with the data on hand
- Both race files carry FINAL win odds only, no timestamps. So: calibration, overround and
  simple return simulations yes; market-movement or "closing line value" work no.
- Only races whose odds cover (nearly) the whole field are used (`races.market_complete`).
  The HKJC results file is a partial sample, so few of its races qualify.
- Every model comparison is chronological: fit on earlier race days, score on later ones,
  and report a bootstrap interval so a handful of lucky races isn't mistaken for an edge.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import DB_PATH, REPORT_DIR

RNG = np.random.default_rng(7)
BINS = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
ODDS_BANDS = [(1.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 30.0), (30.0, 10_000.0)]


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def load_market(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.sql(
        """SELECT o.*, r.finish_pos
           FROM odds_final o JOIN race_results r USING (race_id, horse_key)
           WHERE o.market_complete"""
    ).df()


def calibration(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in m.groupby("source"):
        g = g.assign(bin=pd.cut(g.implied_prob_norm, BINS, include_lowest=True))
        for b, h in g.groupby("bin", observed=True):
            lo, hi = wilson(h.won.sum(), len(h))
            rows.append(
                {"source": src, "prob_bin": str(b), "runners": len(h), "mean_implied_prob": h.implied_prob_norm.mean(),
                 "actual_win_rate": h.won.mean(), "ci_low": float(lo), "ci_high": float(hi)}
            )
    return pd.DataFrame(rows)


def _boot_roi(race_ids: np.ndarray, stake: np.ndarray, ret: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    codes, uniq = pd.factorize(race_ids)
    s = np.bincount(codes, weights=stake, minlength=len(uniq))
    r = np.bincount(codes, weights=ret, minlength=len(uniq))
    roi = r.sum() / s.sum() - 1
    idx = RNG.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = r[idx].sum(axis=1) / s[idx].sum(axis=1) - 1
    return roi, *np.percentile(boots, [2.5, 97.5])


def returns(m: pd.DataFrame) -> pd.DataFrame:
    """Flat 1-unit bets. With takeout T, betting everything returns about -T in expectation."""
    rows = []
    for src, g in m.groupby("source"):
        ret_all = g.won.astype(float) * g.final_odds
        strategies = {"back every runner": g.index == g.index}
        strategies["back the favourite"] = (g.fav_rank == 1).to_numpy()
        for lo, hi in ODDS_BANDS:
            strategies[f"odds {lo:g}-{hi:g}" if hi < 1000 else f"odds {lo:g}+"] = g.final_odds.between(lo, hi, inclusive="left").to_numpy()
        for name, mask in strategies.items():
            mask = np.asarray(mask)
            if mask.sum() < 20:
                continue
            roi, lo_ci, hi_ci = _boot_roi(g.race_id.to_numpy()[mask], np.ones(mask.sum()), ret_all.to_numpy()[mask])
            rows.append({"source": src, "strategy": name, "bets": int(mask.sum()), "roi": roi,
                         "ci_low": lo_ci, "ci_high": hi_ci})
    return pd.DataFrame(rows)


def market_summary(m: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in m.groupby("source"):
        w = g[g.won]
        ll = -np.log(w.groupby("race_id").implied_prob_norm.mean().clip(1e-6))
        field = g.groupby("race_id").size()
        uni = -np.log(1.0 / field.reindex(ll.index))
        brier = ((g.implied_prob_norm - g.won.astype(float)) ** 2).mean()
        fav = g[g.fav_rank == 1]
        rows.append(
            {"source": src, "races": g.race_id.nunique(), "runners": len(g),
             "median_overround": races[(races.source == src) & races.market_complete].overround.median(),
             "market_logloss": ll.mean(), "uniform_logloss": uni.mean(), "market_brier": brier,
             "favourite_win_rate": fav.won.mean(), "favourite_mean_implied": fav.implied_prob_norm.mean()}
        )
    return pd.DataFrame(rows)


def _softmax_by_race(df: pd.DataFrame, score: np.ndarray, temp: float) -> np.ndarray:
    z = score / temp
    z = z - pd.Series(z, index=df.index).groupby(df.race_id).transform("max").to_numpy()
    e = np.exp(z)
    return e / pd.Series(e, index=df.index).groupby(df.race_id).transform("sum").to_numpy()


def _race_logloss(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    w = df.assign(p=np.clip(p, 1e-6, 1))[df.won]
    return -np.log(w.groupby("race_id").p.mean())


def _top1(df: pd.DataFrame, p: np.ndarray) -> float:
    # The raw files list runners in finishing order, so ties must be broken at random,
    # otherwise "pick the first row" silently picks the winner.
    d = df.assign(p=p + RNG.random(len(p)) * 1e-9)
    top = d.loc[d.groupby("race_id").p.idxmax()]
    return float(top.groupby("race_id").won.max().mean())


def baselines(con: duckdb.DuckDBPyConnection, m: pd.DataFrame, test_frac: float = 0.3) -> pd.DataFrame:
    """Chronological holdout: fit temperatures on early race days, score on later ones."""
    hist = con.sql("SELECT race_id, horse_key, elo_pre FROM rating_history").df()
    rtg = con.sql("SELECT race_id, horse_key, pre_race_rating FROM race_entries").df()
    d = m.merge(hist, on=["race_id", "horse_key"], how="left").merge(rtg, on=["race_id", "horse_key"], how="left")
    d["elo_pre"] = d.elo_pre.fillna(1500.0)
    rows = []
    for src, g in d.groupby("source"):
        g = g.sort_values("race_date").reset_index(drop=True)
        dates = np.sort(g.race_date.unique())
        cut = dates[int(len(dates) * (1 - test_frac))]
        tr, te = g[g.race_date < cut].reset_index(drop=True), g[g.race_date >= cut].reset_index(drop=True)
        n_tr, n_te = tr.race_id.nunique(), te.race_id.nunique()

        cands: dict[str, tuple[np.ndarray, str]] = {}
        field = te.groupby("race_id").race_id.transform("size").to_numpy()
        cands["uniform (1/field)"] = (1.0 / field, "")
        cands["market (final odds)"] = (te.implied_prob_norm.to_numpy(), "")

        score_sets = {"Elo softmax": ("elo_pre", [50, 100, 200, 400, 800, 1e6])}
        if src == "hkjc" and g.pre_race_rating.notna().mean() > 0.9:
            score_sets["official rating softmax"] = ("pre_race_rating", [2, 4, 8, 16, 32, 1e6])
        for name, (col, grid) in score_sets.items():
            best = min(grid, key=lambda t: _race_logloss(tr, _softmax_by_race(tr, tr[col].fillna(tr[col].median()).to_numpy(), t)).mean())
            p = _softmax_by_race(te, te[col].fillna(tr[col].median()).to_numpy(), best)
            cands[name] = (p, f"T={best:g}")

        mk_ll = _race_logloss(te, cands["market (final odds)"][0])
        for name, (p, note) in cands.items():
            ll = _race_logloss(te, p)
            diff = (ll - mk_ll.reindex(ll.index)).to_numpy()
            boots = RNG.choice(diff, size=(2000, len(diff))).mean(axis=1)
            rows.append(
                {"source": src, "model": name, "fit": note, "train_races": n_tr, "test_races": n_te,
                 "test_logloss": ll.mean(), "test_top1": _top1(te, p),
                 "logloss_vs_market": diff.mean(), "vs_market_ci_low": np.percentile(boots, 2.5),
                 "vs_market_ci_high": np.percentile(boots, 97.5)}
            )
    return pd.DataFrame(rows)


def build_market(db_path=DB_PATH) -> dict[str, int]:
    con = duckdb.connect(str(db_path))
    races = con.sql("SELECT * FROM races").df()
    m = load_market(con)
    out = {
        "market_calibration": calibration(m),
        "market_returns": returns(m),
        "market_summary": market_summary(m, races),
        "baseline_eval": baselines(con, m),
    }
    for name, df in out.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()
    return {k: len(v) for k, v in out.items()}


def export_reports(db_path=DB_PATH, out_dir=REPORT_DIR) -> list[str]:
    """CSV research summaries (also downloadable from the dashboard)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    files = {
        "horse_ratings.csv": "SELECT * FROM horse_ratings ORDER BY source, rank_in_source",
        "ecuador_rankings.csv": "SELECT * FROM ecuador_rankings ORDER BY rank",
        "market_calibration.csv": "SELECT * FROM market_calibration",
        "market_returns.csv": "SELECT * FROM market_returns",
        "market_summary.csv": "SELECT * FROM market_summary",
        "baseline_eval.csv": "SELECT * FROM baseline_eval",
        "data_audit.csv": "SELECT * FROM data_audit",
    }
    for fname, q in files.items():
        con.sql(q).df().to_csv(out_dir / fname, index=False)
    con.close()
    return list(files)


if __name__ == "__main__":
    print(build_market())
    print(export_reports())
