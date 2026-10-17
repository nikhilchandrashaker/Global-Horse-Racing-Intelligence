"""Global Horse Racing Intelligence — Streamlit dashboard.

Run:  streamlit run app.py      (after `python run.py` has built data/hri.duckdb)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from hri import DB_PATH, REPORT_DIR  # noqa: E402

st.set_page_config(page_title="Global Horse Racing Intelligence", page_icon="🏇", layout="wide")

SOURCE_LABEL = {"jra": "Japan (JRA)", "hkjc": "Hong Kong (HKJC)"}
HYPOTHESIS_NOTE = (
    "These are **hypotheses to investigate**, not verdicts. With a few weeks of race data per source, "
    "most horses have only 1–5 starts, so luck can easily produce any pattern shown here."
)


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def q(sql: str, **params) -> pd.DataFrame:
    return get_con().execute(sql, list(params.values())).df() if params else get_con().sql(sql).df()


if not DB_PATH.exists():
    st.error("Database not found. Run `python run.py` in the project folder first.")
    st.stop()

PAGES = ["Overview", "Horse rankings", "Horse profile", "Race cards", "Pattern finder",
         "International observatory", "Market lab", "Downloads"]
page = st.sidebar.radio("Module", PAGES)
st.sidebar.caption("Ratings are comparable only within a source. JRA and HKJC pools never meet.")


# ------------------------------------------------------------------------ Overview
def page_overview() -> None:
    st.title("🏇 Global Horse Racing Intelligence")
    st.caption("Japan · Hong Kong · Ecuador — one shared data layer, source-specific IDs, nothing silently merged.")
    c = st.columns(5)
    counts = {t: q(f"SELECT COUNT(*) n FROM {t}").n[0] for t in
              ["horses", "races", "race_entries", "race_results", "barrier_trials"]}
    for col, (name, n) in zip(c, counts.items()):
        col.metric(name.replace("_", " ").title(), f"{n:,}")
    st.subheader("Coverage by source")
    cov = q("""SELECT source, MIN(race_date) AS first_day, MAX(race_date) AS last_day, COUNT(*) AS races,
                      SUM(market_complete::INT) AS market_complete_races
               FROM races GROUP BY source ORDER BY source""")
    st.dataframe(cov, hide_index=True, width="stretch")
    st.info(
        "**Read before trusting any number.** JRA covers Jan–Mar 2024; HKJC covers Mar–Apr 2025 and its results "
        "file is a *partial* sample (~70% of racecard runners). Ecuador is three seasonal top-50 leaderboards, not "
        "full careers. The sources don't overlap in time, so cross-country views are descriptive only."
    )
    st.subheader("Data audit")
    st.dataframe(q("SELECT * FROM data_audit"), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Horse rankings
def page_rankings() -> None:
    st.title("Horse rankings")
    tab_race, tab_ecu = st.tabs(["Race-level ratings (JRA / HKJC)", "Ecuador career summaries"])
    with tab_race:
        c1, c2, c3 = st.columns(3)
        src = c1.selectbox("Source", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get)
        min_starts = c2.slider("Minimum observed starts", 1, 6, 2)
        sort_by = c3.selectbox("Sort by", ["overall_rating", "consistency", "recent_form", "win_rate"])
        df = q(f"""SELECT name, starts_observed AS starts, wins, places, ROUND(overall_rating,1) AS overall_rating,
                          ROUND(consistency,2) AS consistency, ROUND(recent_form,2) AS recent_form,
                          ROUND(win_rate,2) AS win_rate, official_rating_last, confidence, sire
                   FROM horse_ratings WHERE source='{src}' AND starts_observed >= {int(min_starts)}
                   ORDER BY {sort_by} DESC NULLS LAST LIMIT 200""")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(
            "**overall_rating** = Elo (1500 = average) shrunk toward 1500 so a horse needs evidence, not just a lucky "
            "start. **consistency** needs ≥3 starts. **recent_form** is an exponentially weighted within-field "
            "score (0.5 = average). Starts count observed runs in this file only, not career starts."
        )
        if src == "hkjc":
            r = q("SELECT corr(overall_rating, official_rating_last) c, COUNT(*) n FROM horse_ratings "
                  "WHERE source='hkjc' AND official_rating_last IS NOT NULL")
            st.warning(
                f"Correlation between this Elo and HKJC's official rating is only {r.c[0]:.2f} (n={r.n[0]}). "
                "Five weeks of results can't recover long-run ability; treat the official rating as the stronger "
                "ability signal for Hong Kong."
            )
    with tab_ecu:
        season = st.selectbox("View", ["All listed seasons combined", "2024", "2025", "2026 (partial)"])
        if season.startswith("All"):
            df = q("""SELECT rank, name, seasons_listed, starts, wins, seconds, thirds, earnings, classics_won,
                             ROUND(win_rate_shrunk,3) AS win_rate_shrunk, ROUND(podium_rate_shrunk,3) AS podium_rate_shrunk,
                             ROUND(earnings_per_start,0) AS earnings_per_start, ROUND(score,2) AS score
                      FROM ecuador_rankings ORDER BY rank""")
        else:
            df = q(f"""SELECT rank, name, starts, wins, seconds, thirds, earnings, classics_won,
                              ROUND(win_rate_shrunk,3) AS win_rate_shrunk, ROUND(earnings_per_start,0) AS earnings_per_start,
                              ROUND(score,2) AS score
                       FROM ecuador_rankings_by_season WHERE season = {int(season[:4])} ORDER BY rank""")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(
            "Only horses that made a seasonal top-50 appear, so win rates are inflated by selection. Rates are "
            "Bayesian-shrunk toward the pool average (10 pseudo-starts). Earnings currency isn't stated in the source."
        )


# ----------------------------------------------------------------- Horse profile
def page_profile() -> None:
    st.title("Horse profile")
    term = st.text_input("Search horse name", "")
    if len(term) < 2:
        st.caption("Type at least two letters.")
        return
    hits = q(f"""SELECT horse_key, name, source FROM horses WHERE name ILIKE '%{term.replace("'", "''")}%'
                 ORDER BY name LIMIT 50""")
    if hits.empty:
        st.warning("No horse matches.")
        return
    pick = st.selectbox("Match", hits.index, format_func=lambda i: f"{hits.name[i]} — {hits.source[i]}")
    key, name, src = hits.horse_key[pick], hits.name[pick], hits.source[pick]
    st.subheader(name)
    if src == "ecuador":
        st.dataframe(q(f"""SELECT season, starts, wins, seconds, thirds, earnings, classics_won, win_pct, podium_pct
                           FROM horse_career_stats WHERE horse_key='{key}' ORDER BY season"""),
                     hide_index=True, width="stretch")
        return
    rt = q(f"SELECT * FROM horse_ratings WHERE horse_key='{key}'")
    if not rt.empty:
        r = rt.iloc[0]
        c = st.columns(5)
        c[0].metric("Overall (Elo, shrunk)", f"{r.overall_rating:.0f}")
        c[1].metric("Consistency", "n/a" if pd.isna(r.consistency) else f"{r.consistency:.2f}")
        c[2].metric("Recent form", f"{r.recent_form:.2f}")
        c[3].metric("Observed starts", int(r.starts_observed))
        c[4].metric("Confidence", str(r.confidence))
        if src == "jra" and pd.notna(r.sire):
            st.caption(f"Sire: {r.sire} · Dam: {r.dam}")
    hist = q(f"""SELECT ra.race_date, ra.course, ra.surface, ra.distance_m, ra.race_class, rr.finish_pos, rr.status,
                        rr.final_odds, h.elo_pre, h.perf_score
                 FROM race_results rr JOIN races ra USING (race_id)
                 LEFT JOIN rating_history h ON h.race_id = rr.race_id AND h.horse_key = rr.horse_key
                 WHERE rr.horse_key = '{key}' ORDER BY ra.race_date""")
    st.dataframe(hist, hide_index=True, width="stretch")
    if hist.elo_pre.notna().sum() >= 2:
        fig = px.line(hist, x="race_date", y="elo_pre", markers=True, title="Pre-race Elo by start")
        st.plotly_chart(fig, width="stretch")
    if src == "hkjc":
        bt = q(f"""SELECT trial_date, course, distance_m, draw, result, finish_time_s, comment
                   FROM barrier_trials WHERE horse_key='{key}' ORDER BY trial_date""")
        if not bt.empty:
            st.markdown("**Barrier trials**")
            st.dataframe(bt, hide_index=True, width="stretch")


# -------------------------------------------------------------------- Race cards
def page_racecards() -> None:
    st.title("Race cards")
    src = st.selectbox("Source", list(SOURCE_LABEL), index=1, format_func=SOURCE_LABEL.get)
    days = q(f"SELECT DISTINCT race_date FROM races WHERE source='{src}' ORDER BY race_date")
    day = st.selectbox("Meeting date", days.race_date, format_func=lambda d: str(d)[:10])
    races = q(f"""SELECT race_id, course, race_no, surface, distance_m, race_class, going, observed_runners,
                         card_runners, market_complete FROM races WHERE source='{src}' AND race_date = '{str(day)[:10]}'
                  ORDER BY race_no""")
    pick = st.selectbox("Race", races.index, format_func=lambda i: f"{races.course[i]} Race {int(races.race_no[i])} — "
                        f"{races.distance_m[i]:.0f}m {races.surface[i]}")
    rid = races.race_id[pick]
    st.dataframe(races.loc[[pick]].drop(columns=["race_id"]), hide_index=True, width="stretch")
    card = q(f"""SELECT COALESCE(h.name, h.source_horse_id) AS horse, e.draw, e.jockey, e.trainer,
                        e.pre_race_rating AS official_rating, ROUND(rh.elo_pre,0) AS elo_pre, e.carried_wt,
                        e.carried_wt_unit AS wt_unit, e.gear, rr.final_odds, rr.finish_pos, rr.status, e.has_result
                 FROM race_entries e JOIN horses h USING (horse_key)
                 LEFT JOIN race_results rr USING (race_id, horse_key)
                 LEFT JOIN rating_history rh USING (race_id, horse_key)
                 WHERE e.race_id = '{rid}' ORDER BY COALESCE(rr.finish_pos, 99), e.draw""")
    st.dataframe(card, hide_index=True, width="stretch")
    if src == "hkjc" and card.has_result.eq(False).any():
        st.caption("Rows with no result are racecard runners missing from the partial results file.")
    st.caption("Win-probability predictions aren't produced yet. See *Market lab* for why the baseline comes first.")


# ---------------------------------------------------------------- Pattern finder
def page_patterns() -> None:
    st.title("Hidden talent & pattern finder")
    st.warning(HYPOTHESIS_NOTE)
    src = st.selectbox("Source", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get)
    t1, t2, t3, t4 = st.tabs(["Consistent placers", "Improving", "Beating their odds", "Sire performance (JRA)"])
    with t1:
        st.dataframe(q(f"""SELECT name, starts_observed AS starts, places, ROUND(place_rate,2) AS place_rate,
                                  ROUND(consistency,2) AS consistency, wins
                           FROM horse_ratings WHERE source='{src}' AND starts_observed >= 3
                           ORDER BY place_rate DESC, consistency DESC NULLS LAST LIMIT 40"""),
                     hide_index=True, width="stretch")
    with t2:
        st.caption("Recent form minus career average within-field score, ≥3 starts. Regression to the mean is the default explanation.")
        st.dataframe(q(f"""SELECT name, starts_observed AS starts, ROUND(mean_perf,2) AS career_perf,
                                  ROUND(recent_form,2) AS recent_form, ROUND(recent_form - mean_perf,2) AS lift
                           FROM horse_ratings WHERE source='{src}' AND starts_observed >= 3
                           ORDER BY lift DESC LIMIT 40"""), hide_index=True, width="stretch")
    with t3:
        st.caption("Actual wins versus the wins the market implied (sum of normalised implied probabilities), complete-market races only.")
        st.dataframe(q(f"""SELECT COALESCE(h.name, h.source_horse_id) AS horse, COUNT(*) AS starts, SUM(o.won::INT) AS wins,
                                  ROUND(SUM(o.implied_prob_norm),2) AS expected_wins,
                                  ROUND(SUM(o.won::INT) - SUM(o.implied_prob_norm),2) AS wins_minus_expected
                           FROM odds_final o JOIN horses h USING (horse_key)
                           WHERE o.market_complete AND o.source='{src}'
                           GROUP BY 1 HAVING COUNT(*) >= 3 ORDER BY wins_minus_expected DESC LIMIT 40"""),
                     hide_index=True, width="stretch")
    with t4:
        if src != "jra":
            st.info("Pedigree fields exist only in the JRA file.")
        else:
            surface = st.selectbox("Surface", ["ALL", "TURF", "DIRT"])
            flt = "" if surface == "ALL" else f"AND ra.surface = '{surface}'"
            st.dataframe(q(f"""SELECT h.sire, COUNT(*) AS runners, SUM(o.won::INT) AS wins,
                                      ROUND(SUM(o.won::INT)::DOUBLE / COUNT(*), 3) AS win_rate,
                                      ROUND(SUM(o.implied_prob_norm), 1) AS expected_wins,
                                      ROUND(SUM(o.won::INT) / SUM(o.implied_prob_norm), 2) AS actual_over_expected
                               FROM odds_final o JOIN horses h USING (horse_key) JOIN races ra USING (race_id)
                               WHERE o.market_complete AND o.source='jra' AND h.sire IS NOT NULL {flt}
                               GROUP BY 1 HAVING COUNT(*) >= 40 ORDER BY actual_over_expected DESC LIMIT 40"""),
                         hide_index=True, width="stretch")
            st.caption("Actual/expected near 1.0 means the market already priced the sire's runners fairly. "
                       "Anything far from 1.0 on ~40 runners is within noise.")


# -------------------------------------------------------- International observatory
def page_observatory() -> None:
    st.title("International racing observatory")
    tab_race, tab_ecu = st.tabs(["Race-level (JRA vs HKJC)", "Career summaries (Ecuador)"])
    with tab_race:
        c1, c2, c3 = st.columns(3)
        surfaces = c1.multiselect("Surface", ["TURF", "DIRT", "AWT"], default=["TURF", "DIRT", "AWT"])
        lo, hi = c2.slider("Distance (m)", 1000, 3600, (1000, 3600), step=200)
        min_field = c3.slider("Minimum observed field size", 2, 16, 2)
        sf = ",".join(f"'{s}'" for s in surfaces) or "''"
        df = q(f"""SELECT source, COUNT(*) AS races, ROUND(AVG(observed_runners),1) AS avg_observed_field,
                          ROUND(AVG(distance_m),0) AS avg_distance_m,
                          ROUND(AVG(CASE WHEN market_complete THEN observed_runners END),1) AS avg_field_complete_markets,
                          SUM(market_complete::INT) AS market_complete_races
                   FROM races WHERE surface IN ({sf}) AND distance_m BETWEEN {lo} AND {hi}
                         AND observed_runners >= {min_field} GROUP BY source ORDER BY source""")
        st.dataframe(df, hide_index=True, width="stretch")
        fav = q(f"""SELECT o.source, COUNT(DISTINCT o.race_id) AS races,
                           ROUND(AVG(CASE WHEN o.fav_rank = 1 THEN o.won::INT END), 3) AS favourite_win_rate
                    FROM odds_final o JOIN races ra USING (race_id)
                    WHERE o.market_complete AND ra.surface IN ({sf}) AND ra.distance_m BETWEEN {lo} AND {hi}
                    GROUP BY o.source""")
        st.dataframe(fav, hide_index=True, width="stretch")
        dist = q(f"""SELECT source, distance_m, COUNT(*) AS races FROM races
                     WHERE surface IN ({sf}) AND distance_m BETWEEN {lo} AND {hi} GROUP BY 1,2""")
        if not dist.empty:
            st.plotly_chart(px.bar(dist, x="distance_m", y="races", color="source", barmode="group",
                                   title="Race distances"), width="stretch")
        st.caption("JRA and HKJC cover different months and different years, and the HKJC results are a partial sample. "
                   "Differences here describe the files, not the countries.")
    with tab_ecu:
        st.dataframe(q("""SELECT season, COUNT(*) AS horses, SUM(starts) AS starts, SUM(wins) AS wins,
                                 ROUND(SUM(wins)::DOUBLE / SUM(starts), 3) AS win_rate,
                                 ROUND(SUM(earnings) / SUM(starts), 0) AS earnings_per_start
                          FROM horse_career_stats GROUP BY season ORDER BY season"""),
                     hide_index=True, width="stretch")
        st.caption("Earnings aren't compared with Japan or Hong Kong: currency is unstated and prize structures differ. "
                   "These are top-50 leaderboards, so rates are selection-biased upward.")


# --------------------------------------------------------------------- Market lab
def page_market() -> None:
    st.title("Odds & market efficiency lab")
    st.caption("Final odds only (no timestamps). Races whose odds cover the whole field only. Chronological holdouts.")
    summ = q("SELECT * FROM market_summary")
    st.dataframe(summ.round(4), hide_index=True, width="stretch")
    cal = q("SELECT * FROM market_calibration")
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=0.8, y1=0.8, line=dict(dash="dash", color="gray"))
    for src, g in cal.groupby("source"):
        fig.add_trace(go.Scatter(
            x=g.mean_implied_prob, y=g.actual_win_rate, mode="markers+lines", name=SOURCE_LABEL.get(src, src),
            error_y=dict(type="data", symmetric=False, array=g.ci_high - g.actual_win_rate,
                         arrayminus=g.actual_win_rate - g.ci_low)))
    fig.update_layout(title="Calibration: implied vs actual win rate (95% Wilson intervals)",
                      xaxis_title="Market-implied probability (normalised)", yaxis_title="Actual win rate")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Flat-stake returns")
    ret = q("SELECT source, strategy, bets, ROUND(roi,3) AS roi, ROUND(ci_low,3) AS ci_low, ROUND(ci_high,3) AS ci_high FROM market_returns")
    st.dataframe(ret, hide_index=True, width="stretch")
    st.caption("95% intervals bootstrap whole races. An interval that spans zero (as every Hong Kong row does) means "
               "no profitability claim is supportable. JRA's heavy losses on long shots show the favourite–longshot bias, "
               "not an exploitable edge: the bet pays out after takeout.")
    st.subheader("Chronological baselines")
    ev = q("SELECT * FROM baseline_eval").round(4)
    st.dataframe(ev, hide_index=True, width="stretch")
    st.caption("Lower log loss is better. `logloss_vs_market` is the model minus the market with a bootstrap interval: "
               "positive means worse than the market. Temperatures were fitted on earlier days only. "
               "T=1e+06 means the fit found no signal and collapsed to uniform, which is the honest answer on 23 training races.")


# --------------------------------------------------------------------- Downloads
def page_downloads() -> None:
    st.title("Research summaries")
    files = sorted(REPORT_DIR.glob("*.csv")) if REPORT_DIR.exists() else []
    if not files:
        st.info("Run `python run.py` to generate CSV reports.")
    for f in files:
        st.download_button(f"Download {f.name}", f.read_bytes(), file_name=f.name, mime="text/csv", key=f.name)


{"Overview": page_overview, "Horse rankings": page_rankings, "Horse profile": page_profile,
 "Race cards": page_racecards, "Pattern finder": page_patterns, "International observatory": page_observatory,
 "Market lab": page_market, "Downloads": page_downloads}[page]()
