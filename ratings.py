"""Module A — Universal Horse Ratings.

Race-level sources (JRA, HKJC): a chronological, multi-runner Elo plus shrunk summaries.
Career-summary source (Ecuador): Bayesian-shrunk win/podium rates plus earnings per start.

Three separate ratings are kept, as the spec asks:
  overall_rating  ability, shrunk toward the pool average so thin records can't top the table
  consistency     how stable a horse's within-field performance is (needs >= 3 starts)
  recent_form     exponentially weighted recent within-field performance

Elo ratings are only comparable *within* a source. JRA and HKJC pools never meet, so there is
no valid way to put them on one scale; the dashboard shows them side by side, never merged.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import DB_PATH

START, K_FACTOR = 1500.0, 32.0
SHRINK_STARTS = 3.0  # starts at which a horse's own evidence gets 50% weight
EMA_ALPHA = 0.5


def compute_elo(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Chronological multi-runner Elo. Returns one row per (race, finisher) with the PRE-race rating."""
    df = results.merge(races[["race_id", "race_date", "race_no"]], on="race_id")
    df = df[df.finish_pos.notna()].sort_values(["source", "race_date", "race_no", "race_id"])
    rating: dict[str, float] = {}
    n_prior: dict[str, int] = {}
    out: list[tuple] = []
    for race_id, g in df.groupby("race_id", sort=False):
        f = len(g)
        if f < 2:
            continue
        keys = g.horse_key.to_numpy()
        pos = g.finish_pos.to_numpy(dtype=float)
        pre = np.array([rating.get(k, START) for k in keys])
        # pairwise expected / actual scores
        diff = (pre[None, :] - pre[:, None]) / 400.0  # [i, j] = (r_j - r_i)/400
        expected = 1.0 / (1.0 + 10.0**diff)
        actual = (pos[None, :] > pos[:, None]) + 0.5 * (pos[None, :] == pos[:, None])
        np.fill_diagonal(expected, 0.0)
        np.fill_diagonal(actual, 0.0)
        delta = K_FACTOR / (f - 1) * (actual - expected).sum(axis=1)
        avg_rank = rankdata(pos, method="average")
        perf = (f - avg_rank) / (f - 1)
        date = g.race_date.iloc[0]
        for k, p, pr, d, pf in zip(keys, pos, pre, delta, perf):
            out.append((race_id, k, date, p, f, n_prior.get(k, 0), pr, pr + d, pf))
            rating[k] = pr + d
            n_prior[k] = n_prior.get(k, 0) + 1
    return pd.DataFrame(
        out,
        columns=["race_id", "horse_key", "race_date", "finish_pos", "finishers", "n_prior_starts",
                 "elo_pre", "elo_post", "perf_score"],
    )


def summarise_horses(hist: pd.DataFrame, horses: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """One row per horse with ability, consistency and form, plus confidence."""
    hist = hist.sort_values(["horse_key", "race_date"])
    rows = []
    for key, g in hist.groupby("horse_key"):
        n = len(g)
        perf = g.perf_score.to_numpy()
        ema = perf[0]
        for p in perf[1:]:
            ema = EMA_ALPHA * p + (1 - EMA_ALPHA) * ema
        w = n / (n + SHRINK_STARTS)
        elo_raw = g.elo_post.iloc[-1]
        rows.append(
            {
                "horse_key": key,
                "starts_observed": n,
                "wins": int((g.finish_pos == 1).sum()),
                "places": int((g.finish_pos <= 3).sum()),
                "mean_perf": perf.mean(),
                "elo_raw": elo_raw,
                "overall_rating": START + (elo_raw - START) * w,
                "consistency": (1 - min(1.0, perf.std(ddof=0) / 0.5)) if n >= 3 else np.nan,
                "recent_form": 0.5 + (ema - 0.5) * (n / (n + 2.0)),
                "last_race_date": g.race_date.iloc[-1],
            }
        )
    out = pd.DataFrame(rows)
    out["win_rate"] = out.wins / out.starts_observed
    out["place_rate"] = out.places / out.starts_observed
    out["confidence"] = pd.cut(out.starts_observed, [0, 2, 4, 10_000], labels=["thin", "moderate", "solid"])
    # official HK rating (pre-race) at last start, for validation
    last_rtg = (
        entries.dropna(subset=["pre_race_rating"])
        .merge(hist[["race_id", "horse_key", "race_date"]], on=["race_id", "horse_key"])
        .sort_values("race_date").groupby("horse_key").tail(1)[["horse_key", "pre_race_rating"]]
        .rename(columns={"pre_race_rating": "official_rating_last"})
    )
    out = out.merge(last_rtg, on="horse_key", how="left")
    cols = ["horse_key", "source", "name", "sire", "dam"]
    out = out.merge(horses[cols], on="horse_key", how="left")
    out["rank_in_source"] = out.groupby("source").overall_rating.rank(ascending=False, method="min")
    return out


def ecuador_rankings(stats: pd.DataFrame, prior_strength: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bayesian-shrunk ranking for the seasonal Ecuador leaderboards. Returns (overall, by_season)."""

    def score(df: pd.DataFrame) -> pd.DataFrame:
        p_win = df.wins.sum() / df.starts.sum()
        p_pod = (df.wins + df.seconds + df.thirds).sum() / df.starts.sum()
        df = df.copy()
        df["win_rate_shrunk"] = (df.wins + prior_strength * p_win) / (df.starts + prior_strength)
        df["podium_rate_shrunk"] = (df.wins + df.seconds + df.thirds + prior_strength * p_pod) / (df.starts + prior_strength)
        df["earnings_per_start"] = df.earnings / df.starts
        z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)
        df["score"] = (
            0.5 * z(df.win_rate_shrunk) + 0.3 * z(df.podium_rate_shrunk) + 0.2 * z(np.log(df.earnings_per_start))
        )
        df["rank"] = df.score.rank(ascending=False, method="min").astype(int)
        return df.sort_values("rank")

    agg = (
        stats.groupby(["horse_key", "name"], as_index=False)
        .agg(seasons_listed=("season", "nunique"), starts=("starts", "sum"), wins=("wins", "sum"),
             seconds=("seconds", "sum"), thirds=("thirds", "sum"), earnings=("earnings", "sum"),
             classics_won=("classics_won", "sum"))
    )
    overall = score(agg)
    by_season = pd.concat([score(g.assign(season=s)) for s, g in stats.groupby("season")], ignore_index=True)
    return overall, by_season


def build_ratings(db_path=DB_PATH) -> dict[str, int]:
    con = duckdb.connect(str(db_path))
    results = con.sql("SELECT * FROM race_results").df()
    races = con.sql("SELECT race_id, race_date, race_no, source FROM races").df()
    horses = con.sql("SELECT * FROM horses").df()
    entries = con.sql("SELECT * FROM race_entries").df()
    stats = con.sql("SELECT * FROM horse_career_stats").df()

    hist = compute_elo(results, races)
    summary = summarise_horses(hist, horses, entries)
    ecu_all, ecu_season = ecuador_rankings(stats)

    out = {"rating_history": hist, "horse_ratings": summary, "ecuador_rankings": ecu_all,
           "ecuador_rankings_by_season": ecu_season}
    for name, df in out.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    counts = {n: len(df) for n, df in out.items()}
    con.close()
    return counts


if __name__ == "__main__":
    print(build_ratings())
