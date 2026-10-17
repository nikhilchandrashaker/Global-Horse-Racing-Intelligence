"""Load, clean and standardise the three source datasets into related DuckDB tables.

Design rules
- Never merge everything into one giant table. Related tables, source-specific IDs.
- horse_key = '<source>:<source id>' so IDs from different countries can never collide.
  Nothing here matches horses across countries; that needs a separate identity step.
- Keep provenance: every table carries `source`; `data_audit` records what was dropped and why.
- Pre-race facts (entries) are kept apart from post-race facts (results).
"""

from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import DB_PATH, RAW_DIR
from .parse import (
    hk_finish,
    hk_surface,
    jra_finish,
    parse_decimal_comma,
    parse_distance,
    parse_margin,
    parse_race_no,
    parse_time,
)

# Overround window used to decide whether a race's odds cover (nearly) the whole field.
# Real win pools carry takeout, so a complete field sums to well above 1.
OVERROUND_LO, OVERROUND_HI = 1.05, 1.45

HK_CLASS_MAP = {
    "CLASS 1": "1", "CLASS 2": "2", "CLASS 3": "3", "CLASS 4": "4", "CLASS 5": "5",
    "GROUP ONE": "G1", "GROUP TWO": "G2", "GROUP THREE": "G3", "4 YEAR OLDS": "4YO",
}


class Audit:
    """Collects data-quality facts so they end up in the database and the README."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, source: str, check: str, value) -> None:
        self.rows.append((source, check, str(value)))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["source", "check", "value"])


# --------------------------------------------------------------------------- JRA
def load_jra(raw: Path, audit: Audit) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(raw / "jra_race_results_2024.csv", parse_dates=["race_date"])
    audit.add("jra", "rows_raw", len(df))
    audit.add("jra", "date_range", f"{df.race_date.min().date()} to {df.race_date.max().date()}")

    df["race_id"] = (
        "jra:" + df.race_date.dt.strftime("%Y-%m-%d") + ":" + df.race_course.str.title() + ":"
        + df.race_no.astype(int).astype(str).str.zfill(2)
    )
    df["horse_key"] = "jra:" + df.horse_id.astype(str)

    fin = df.final_position.map(jra_finish)
    df["finish_pos"] = [f[0] for f in fin]
    df["status"] = [f[1] for f in fin]
    audit.add("jra", "rows_without_finish_position", int((df.status != "FIN").sum()))
    audit.add("jra", "rows_missing_pedigree_or_connections", int(df.sire.isna().sum()))

    df["surface"] = df.race_track.str.upper()
    df["finish_time_s"] = df.finish_time.map(parse_time)
    df["margin_lengths"] = df.margin.map(parse_margin)

    races = (
        df.groupby("race_id")
        .agg(
            source=("race_id", lambda s: "jra"),
            race_date=("race_date", "first"),
            course=("race_course", "first"),
            race_no=("race_no", "first"),
            surface=("surface", "first"),
            distance_m=("race_distance", "first"),
            race_class=("race_type", "first"),
            going=("race_going", "first"),
            race_name=("race_name", "first"),
            prize_value=("value_of_race", "first"),
            observed_runners=("horse_key", "size"),
        )
        .reset_index()
    )
    races["track_detail"] = None

    entries = pd.DataFrame(
        {
            "race_id": df.race_id,
            "horse_key": df.horse_key,
            "source": "jra",
            "draw": df.bracket,
            "horse_number": df.horse_number,
            "jockey": df.jockey,
            "trainer": df.trainer,
            "pre_race_rating": np.nan,  # JRA gives no official rating in this file
            "carried_wt": df.horse_weight,
            "carried_wt_unit": "kg",
            "body_wt": np.nan,
            "age": df.horse_age,
            "sex": df.horse_sex,
            "gear": None,
            "has_result": True,
        }
    )
    results = pd.DataFrame(
        {
            "race_id": df.race_id,
            "horse_key": df.horse_key,
            "source": "jra",
            "finish_pos": df.finish_pos,
            "status": df.status,
            "margin_lengths": df.margin_lengths,
            "finish_time_s": df.finish_time_s,
            "final_odds": df.win_odds,
            "fav_rank": df.win_fav,
            "running_position": df.running_position,
            "incident_report": None,
        }
    )
    last = df.sort_values("race_date").groupby("horse_key").tail(1)
    horses = pd.DataFrame(
        {
            "horse_key": last.horse_key,
            "source": "jra",
            "source_horse_id": last.horse_id.astype(str),
            "name": last.horse_name,
            "sex": last.horse_sex,
            "sire": last.sire,
            "dam": last.dam,
            "dam_sire": last.dam_sire,
            "dam_dam": last.dam_dam,
        }
    )
    return dict(races=races, entries=entries, results=results, horses=horses)


# -------------------------------------------------------------------------- HKJC
def load_hkjc(raw: Path, audit: Audit) -> dict[str, pd.DataFrame]:
    rec = pd.read_csv(raw / "race_record_2024.csv", dtype=str)
    audit.add("hkjc", "race_record_rows_raw", len(rec))

    overseas = rec.race_course.eq("Meydan") | rec.race_index.eq("Overseas")
    audit.add("hkjc", "overseas_rows_dropped", int(overseas.sum()))
    rec = rec[~overseas].copy()

    rec["race_date"] = pd.to_datetime(rec.race_date, format="%Y/%m/%d")
    rec["race_no_i"] = rec.race_no.map(parse_race_no).astype(int)
    rec["race_id"] = (
        "hkjc:" + rec.race_date.dt.strftime("%Y-%m-%d") + ":" + rec.race_course + ":"
        + rec.race_no_i.astype(str).str.zfill(2)
    )
    rec["horse_key"] = "hkjc:" + rec.horse_id
    audit.add("hkjc", "race_record_date_range",
              f"{rec.race_date.min().date()} to {rec.race_date.max().date()} (file name says 2024; dates are 2025)")
    dup = rec.duplicated(["race_id", "horse_key"]).sum()
    audit.add("hkjc", "duplicate_race_horse_rows", int(dup))
    rec = rec.drop_duplicates(["race_id", "horse_key"])

    fin = rec.result_position.map(hk_finish)
    rec["finish_pos"] = [f[0] for f in fin]
    rec["status"] = [f[1] for f in fin]
    surf = rec.race_track.map(hk_surface)
    rec["surface"] = [s[0] for s in surf]
    rec["track_detail"] = [s[1] for s in surf]
    rec["distance_m"] = rec.distance.map(parse_distance)
    rec["win_odds_n"] = pd.to_numeric(rec.win_odds, errors="coerce")
    rec["finish_time_s"] = rec.finish_time.map(parse_time)
    rec["margin_lengths"] = rec.LBW.map(parse_margin)
    for c in ["rtg", "draw", "act_wt", "declar_horse_wt"]:
        rec[c] = pd.to_numeric(rec[c], errors="coerce")

    # ---- racecards: pre-race entries, including runners absent from the results file
    cards = pd.concat(
        [pd.read_csv(f, dtype=str).assign(card_file=Path(f).name) for f in sorted(glob.glob(str(raw / "hkjc_racecard_*.csv")))],
        ignore_index=True,
    )
    audit.add("hkjc", "racecard_rows_raw", len(cards))
    nonrunner = cards.rtg.eq("-") & cards.draw.eq("-")
    audit.add("hkjc", "racecard_nonrunner_rows_dropped", int(nonrunner.sum()))
    cards = cards[~nonrunner].copy()
    cards["race_date"] = pd.to_datetime(cards.race_date, format="%Y/%m/%d")
    cards["race_no_i"] = cards.race_no.map(parse_race_no).astype(int)
    cards["race_id"] = (
        "hkjc:" + cards.race_date.dt.strftime("%Y-%m-%d") + ":" + cards.race_course + ":"
        + cards.race_no_i.astype(str).str.zfill(2)
    )
    cards["horse_key"] = "hkjc:" + cards.horse_id
    cards["distance_m"] = cards.distance.map(parse_distance)
    cards["race_class_n"] = cards.race_class.str.upper().map(HK_CLASS_MAP).fillna(cards.race_class)
    surf = cards.race_track.map(hk_surface)
    cards["surface"] = [s[0] for s in surf]
    cards["track_detail"] = [s[1] for s in surf]
    for c in ["rtg", "draw", "act_wt", "declar_horse_wt"]:
        cards[c] = pd.to_numeric(cards[c], errors="coerce")
    cards = cards.drop_duplicates(["race_id", "horse_key"])

    # coverage of the results file against the racecards
    matched = cards.merge(rec[["race_id", "horse_key"]], on=["race_id", "horse_key"], how="left", indicator=True)
    audit.add("hkjc", "racecard_runners_found_in_results_pct", round(100 * (matched._merge == "both").mean(), 1))

    card_counts = cards.groupby("race_id").size().rename("card_runners")

    # ---- races: record first, racecard-only races appended
    r_rec = (
        rec.groupby("race_id")
        .agg(
            race_date=("race_date", "first"), course=("race_course", "first"), race_no=("race_no_i", "first"),
            surface=("surface", "first"), track_detail=("track_detail", "first"), distance_m=("distance_m", "first"),
            race_class=("race_class", "first"), going=("going", "first"), observed_runners=("horse_key", "size"),
        )
        .reset_index()
    )
    only_card = cards[~cards.race_id.isin(r_rec.race_id)]
    r_card = (
        only_card.groupby("race_id")
        .agg(
            race_date=("race_date", "first"), course=("race_course", "first"), race_no=("race_no_i", "first"),
            surface=("surface", "first"), track_detail=("track_detail", "first"), distance_m=("distance_m", "first"),
            race_class=("race_class_n", "first"), going=("going", "first"),
        )
        .reset_index()
    )
    r_card["observed_runners"] = 0
    races = pd.concat([r_rec, r_card], ignore_index=True)
    races["source"] = "hkjc"
    races["race_name"] = None
    races["prize_value"] = np.nan
    races = races.merge(card_counts, on="race_id", how="left")
    audit.add("hkjc", "races_in_results", len(r_rec))
    audit.add("hkjc", "races_only_on_racecards", len(r_card))

    # ---- entries
    e_rec = pd.DataFrame(
        {
            "race_id": rec.race_id, "horse_key": rec.horse_key, "source": "hkjc", "draw": rec.draw,
            "horse_number": np.nan, "jockey": rec.jockey, "trainer": rec.trainer, "pre_race_rating": rec.rtg,
            "carried_wt": rec.act_wt, "carried_wt_unit": "lb", "body_wt": rec.declar_horse_wt,
            "age": np.nan, "sex": None, "gear": rec.gear, "has_result": True,
        }
    )
    c_only = cards.merge(rec[["race_id", "horse_key"]], on=["race_id", "horse_key"], how="left", indicator=True)
    c_only = c_only[c_only._merge == "left_only"]
    e_card = pd.DataFrame(
        {
            "race_id": c_only.race_id, "horse_key": c_only.horse_key, "source": "hkjc", "draw": c_only.draw,
            "horse_number": np.nan, "jockey": c_only.jockey, "trainer": c_only.trainer,
            "pre_race_rating": c_only.rtg, "carried_wt": c_only.act_wt, "carried_wt_unit": "lb",
            "body_wt": c_only.declar_horse_wt, "age": np.nan, "sex": None, "gear": c_only.gear, "has_result": False,
        }
    )
    entries = pd.concat([e_rec, e_card], ignore_index=True)

    results = pd.DataFrame(
        {
            "race_id": rec.race_id, "horse_key": rec.horse_key, "source": "hkjc", "finish_pos": rec.finish_pos,
            "status": rec.status, "margin_lengths": rec.margin_lengths, "finish_time_s": rec.finish_time_s,
            "final_odds": rec.win_odds_n, "fav_rank": np.nan, "running_position": rec.running_position,
            "incident_report": rec.incident_report,
        }
    )
    # favourite rank from final odds within race
    results["fav_rank"] = results.groupby("race_id").final_odds.rank(method="min")

    # ---- barrier trials
    bt = pd.read_csv(raw / "barrier_trail_2024.csv", parse_dates=["barrier_trial_date"])
    audit.add("hkjc", "barrier_trial_rows", len(bt))
    surf = bt.race_track.map(hk_surface)
    trials = pd.DataFrame(
        {
            "horse_key": "hkjc:" + bt.horse_id, "trial_date": bt.barrier_trial_date, "course": bt.race_course,
            "batch_no": bt.batch_no, "surface": [s[0] for s in surf], "distance_m": bt.distance, "going": bt.going,
            "draw": bt.draw, "jockey": bt.jockey, "trainer": bt.trainer, "gear": bt.gear,
            "lengths_behind": bt.LBW.map(parse_margin), "finish_time_s": bt.finish_time_s,
            "result": bt.result, "comment": bt.comment,
        }
    )

    names = rec.sort_values("race_date").groupby("horse_key").tail(1)[["horse_key", "horse_id", "horse_name"]]
    all_keys = pd.Series(
        pd.concat([entries.horse_key, trials.horse_key]).unique(), name="horse_key"
    ).to_frame()
    horses = all_keys.merge(names, on="horse_key", how="left")
    horses["source"] = "hkjc"
    horses["source_horse_id"] = horses.horse_key.str.split(":").str[1]
    horses = horses.rename(columns={"horse_name": "name"})[["horse_key", "source", "source_horse_id", "name"]]
    for c in ["sex", "sire", "dam", "dam_sire", "dam_dam"]:
        horses[c] = None

    return dict(races=races, entries=entries, results=results, horses=horses, trials=trials)


# ---------------------------------------------------------------------- Ecuador
def load_ecuador(raw: Path, audit: Audit) -> dict[str, pd.DataFrame]:
    e = pd.read_csv(raw / "analisis_total_rendimiento_caballos_hipodromomsd_ecuador.csv", sep=";", encoding="cp1252")
    audit.add("ecuador", "rows_raw", len(e))
    n_dup = int(e.duplicated().sum())
    audit.add("ecuador", "exact_duplicate_rows_removed", n_dup)
    e = e.drop_duplicates()
    audit.add("ecuador", "rows_after_dedupe", len(e))
    audit.add("ecuador", "rows_per_season", e.groupby("year").size().to_dict())
    audit.add("ecuador", "not_finish_all_zero", bool((e.not_finish == 0).all()))
    audit.add("ecuador", "interpretation",
              "Seasonal top-50 leaderboards (not full career records); earnings currency not stated; 2026 is a partial season")
    stats = pd.DataFrame(
        {
            "horse_key": "ecu:" + e.id_horse.astype(str), "source": "ecuador", "name": e.name, "season": e.year,
            "starts": e.num_carreras, "wins": e["first"], "seconds": e.second, "thirds": e.third,
            "earnings": e.ganancias, "classics_won": e.clasicos_ganados,
            "win_pct": parse_decimal_comma(e.eficiencia), "podium_pct": parse_decimal_comma(e.efectividad),
            "ef_first_places": e.ef_first_places,
        }
    )
    horses = (
        e.sort_values("year").groupby("id_horse").tail(1)
        .assign(horse_key=lambda d: "ecu:" + d.id_horse.astype(str), source="ecuador",
                source_horse_id=lambda d: d.id_horse.astype(str))
        [["horse_key", "source", "source_horse_id", "name"]]
    )
    for c in ["sex", "sire", "dam", "dam_sire", "dam_dam"]:
        horses[c] = None
    return dict(stats=stats, horses=horses)


# ------------------------------------------------------------------------- build
def flag_market_complete(races: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """A race is usable for market work only if a winner and odds cover (nearly) the whole field."""
    g = results.dropna(subset=["final_odds"]).assign(ip=lambda d: 1 / d.final_odds).groupby("race_id")
    ov = g.ip.sum().rename("overround")
    n_odds = g.size().rename("n_with_odds")
    won = results.groupby("race_id").finish_pos.min().eq(1).rename("has_winner")
    out = races.merge(ov, on="race_id", how="left").merge(n_odds, on="race_id", how="left").merge(
        won, on="race_id", how="left")
    out["has_winner"] = out.has_winner.fillna(False)
    full_odds = out.n_with_odds.eq(out.observed_runners)
    out["market_complete"] = (
        out.has_winner & full_odds & out.overround.between(OVERROUND_LO, OVERROUND_HI)
    ).fillna(False)
    return out


def build(raw: Path = RAW_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    audit = Audit()
    jra = load_jra(raw, audit)
    hk = load_hkjc(raw, audit)
    ecu = load_ecuador(raw, audit)

    races = pd.concat([jra["races"], hk["races"]], ignore_index=True)
    results = pd.concat([jra["results"], hk["results"]], ignore_index=True)
    entries = pd.concat([jra["entries"], hk["entries"]], ignore_index=True)
    horses = pd.concat([jra["horses"], hk["horses"], ecu["horses"]], ignore_index=True)

    races = flag_market_complete(races, results)
    races["field_size"] = races.observed_runners
    for src in ["jra", "hkjc"]:
        sub = races[races.source == src]
        audit.add(src, "races_total", len(sub))
        audit.add(src, "races_with_winner", int(sub.has_winner.sum()))
        audit.add(src, "races_market_complete", int(sub.market_complete.sum()))
        audit.add(src, "median_overround", round(float(sub.overround.median()), 3))

    # results only for runners with a result; derived flags
    results["won"] = results.finish_pos.eq(1)
    results["placed"] = results.finish_pos.le(3)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    tables = {
        "horses": horses, "races": races, "race_entries": entries, "race_results": results,
        "barrier_trials": hk["trials"], "horse_career_stats": ecu["stats"], "data_audit": audit.frame(),
    }
    for name, df in tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    # predictions are written by the model stage; the schema exists so the dashboard can rely on it
    con.execute(
        """CREATE TABLE predictions (
               race_id VARCHAR, horse_key VARCHAR, model VARCHAR, win_prob DOUBLE,
               predicted_rank INTEGER, made_at TIMESTAMP)"""
    )
    # market view: normalised implied probabilities, usable races only
    con.execute(
        """CREATE VIEW odds_final AS
           SELECT r.race_id, r.horse_key, r.final_odds, 1.0 / r.final_odds AS implied_prob_raw,
                  (1.0 / r.final_odds) / SUM(1.0 / r.final_odds) OVER (PARTITION BY r.race_id) AS implied_prob_norm,
                  r.fav_rank, r.won, ra.source, ra.race_date, ra.market_complete
           FROM race_results r JOIN races ra USING (race_id)
           WHERE r.final_odds IS NOT NULL"""
    )
    counts = {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in list(tables) + ["predictions"]}
    con.close()
    return counts


if __name__ == "__main__":
    print(build())
