"""Tests for the parts that are easy to get silently wrong."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hri import DB_PATH  # noqa: E402
from hri.parse import (  # noqa: E402
    hk_finish, hk_surface, jra_finish, parse_decimal_comma, parse_distance, parse_margin, parse_race_no, parse_time,
)
from hri.ratings import compute_elo, ecuador_rankings  # noqa: E402


def test_margins():
    assert parse_margin("1-1/4") == 1.25
    assert parse_margin("1 3/4") == 1.75
    assert parse_margin("3/4") == 0.75
    assert parse_margin("HD") == 0.2
    assert parse_margin("11") == 11.0
    assert math.isnan(parse_margin("---"))
    assert math.isnan(parse_margin(None))


def test_times():
    assert parse_time("1.39.58") == pytest.approx(99.58)
    assert parse_time("1:56.0") == pytest.approx(116.0)
    assert parse_time("57.38") == pytest.approx(57.38)
    assert math.isnan(parse_time("junk"))


def test_misc_parsers():
    assert parse_distance("1650M") == 1650 and parse_distance(1200) == 1200
    assert parse_race_no("Race 7") == 7 and math.isnan(parse_race_no("oversea"))
    assert hk_surface('TURF - "A+3" Course') == ("TURF", "A+3")
    assert hk_surface("All Weather Track")[0] == "AWT"
    assert hk_finish("03") == (3.0, "FIN") and hk_finish("WV")[1] == "WV"
    assert jra_finish("3rd") == (3.0, "FIN") and jra_finish(np.nan)[1] == "NF"
    assert parse_decimal_comma(pd.Series(["81,82", "100"])).tolist() == [81.82, 100.0]


def _toy():
    races = pd.DataFrame({"race_id": ["r1", "r2"], "race_date": pd.to_datetime(["2025-01-01", "2025-01-08"]),
                          "race_no": [1, 1], "source": "x"})
    res = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 3, "horse_key": ["a", "b", "c"] * 2,
                        "finish_pos": [1, 2, 3, 1, 2, 3], "source": "x"})
    return res, races


def test_elo_uses_only_prior_information():
    res, races = _toy()
    h = compute_elo(res, races)
    first = h[h.race_id == "r1"]
    assert (first.elo_pre == 1500).all()  # nobody has history before race 1
    second = h[h.race_id == "r2"].set_index("horse_key")
    assert second.elo_pre["a"] > second.elo_pre["b"] > second.elo_pre["c"]
    # zero-sum within a race
    assert (h[h.race_id == "r1"].elo_post - h[h.race_id == "r1"].elo_pre).sum() == pytest.approx(0, abs=1e-9)


def test_ecuador_shrinkage_stops_tiny_samples_topping_the_table():
    stats = pd.DataFrame({
        "horse_key": ["a", "b", "c"], "name": list("abc"), "season": [2024] * 3,
        "starts": [1, 30, 30], "wins": [1, 15, 10], "seconds": [0, 5, 8], "thirds": [0, 5, 5],
        "earnings": [100, 5000, 4000], "classics_won": [0, 1, 0],
    })
    overall, _ = ecuador_rankings(stats)
    assert overall.set_index("horse_key")["rank"]["b"] == 1  # 1-for-1 doesn't beat 15-for-30


@pytest.mark.skipif(not DB_PATH.exists(), reason="run `python run.py` first")
def test_database_integrity():
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = lambda s: con.sql(s).fetchone()[0]
    assert q("SELECT COUNT(*) FROM (SELECT race_id, horse_key FROM race_entries GROUP BY 1,2 HAVING COUNT(*)>1)") == 0
    assert q("SELECT COUNT(*) FROM race_results r LEFT JOIN races ra USING (race_id) WHERE ra.race_id IS NULL") == 0
    assert q("SELECT COUNT(*) FROM race_results r LEFT JOIN race_entries e USING (race_id, horse_key) WHERE e.race_id IS NULL") == 0
    # keys never collide across countries
    assert q("SELECT COUNT(*) FROM horses WHERE horse_key NOT LIKE source || '%' AND NOT (source='ecuador' AND horse_key LIKE 'ecu:%')") == 0
    # pre-race Elo for a horse's first start must be the starting rating
    assert q("SELECT COUNT(*) FROM rating_history WHERE n_prior_starts=0 AND elo_pre<>1500") == 0
