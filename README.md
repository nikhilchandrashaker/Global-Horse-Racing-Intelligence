# Global Horse Racing Intelligence

A local, reproducible prototype that turns three racing datasets (Japan, Hong Kong, Ecuador) into one shared
data layer, a rating engine, an odds/market lab and a Streamlit dashboard.

**Status:** steps 1–3 of the build plan are done (audit, shared data layer, ratings and profiles), plus the
market-efficiency baseline from step 5. The ML race predictor is deliberately *not* built yet. See
[What the data can and can't support](#what-the-data-can-and-cant-support) for why.

## Quick start

```bash
pip install -r requirements.txt
python run.py                 # builds data/hri.duckdb + reports/*.csv (about 10 seconds)
streamlit run app.py          # opens the dashboard
pytest -q tests               # parser, Elo and database-integrity tests (run after the build)
```

Tested on Python 3.13 with pandas 3.0, DuckDB 1.5, Streamlit 1.66.

## What's in the dashboard

| Page | What it does |
|---|---|
| Overview | Row counts, coverage by source, the full data audit |
| Horse rankings | Overall rating, consistency and recent form for JRA/HKJC; shrunk leaderboards for Ecuador |
| Horse profile | Search a horse: ratings, every observed start, Elo trajectory, pedigree (JRA), barrier trials (HKJC) |
| Race cards | Pick a meeting and race: draw, jockey, official rating, pre-race Elo, final odds, result |
| Pattern finder | Consistent placers, improving horses, horses beating their odds, sire actual-vs-expected. All labelled as hypotheses |
| International observatory | JRA vs HKJC race-level comparison with surface/distance/field filters; Ecuador kept separate |
| Market lab | Calibration, flat-stake returns with bootstrap intervals, chronological model-vs-market baselines |
| Downloads | CSV research summaries |

## The data

| Source | File(s) | Rows | Coverage |
|---|---|---|---|
| Japan (JRA) | `jra_race_results_2024.csv` | 12,101 starts, 873 races, 6,683 horses | 2024-01-06 → 2024-03-31 |
| Hong Kong results | `race_record_2024.csv` | 1,021 starts, 115 races (2 overseas rows dropped) | 2025-03-02 → 2025-04-09 |
| Hong Kong racecards | 11 × `hkjc_racecard_*.csv` | 1,278 entries | 2025-03-02 → 2025-04-09 |
| Hong Kong barrier trials | `barrier_trail_2024.csv` | 1,000 trials, 684 horses | 2025-03-04 → 2025-04-15 |
| Ecuador (Hipódromo MSD) | `analisis_total_rendimiento_…csv` | 750 rows → **150 after de-duplication** | seasons 2024, 2025, 2026 |

Things the audit found that shaped the design:

1. **The HKJC results file is a partial sample.** Only about 70% of racecard runners appear in it, 13 of 115
   races have no winner recorded, and the median sum of implied probabilities is 0.98 where a full field would be
   about 1.2. Only 41 of 115 HKJC races have odds covering the whole field.
2. **`race_record_2024.csv` is dated 2025**, not 2024. JRA (early 2024) and HKJC (early 2025) don't overlap in time.
3. **The Ecuador file is three seasonal top-50 leaderboards**, each repeated 4–6 times: 600 of 750 rows are
   exact duplicates. It is not full career data, so rates are selection-biased upward. Earnings currency isn't
   stated; `not_finish` is zero everywhere and is ignored; 2026 is a partial season.
4. **JRA horses have almost no history in the window**: 6,683 horses over 12,101 starts, about 1.8 each.
5. **Odds are final odds only**, with no timestamps, so there is no market-movement analysis.
6. **No identifiers link countries.** Keys are `jra:<id>`, `hkjc:<id>`, `ecu:<id>`; names are never used to match
   horses across countries.
7. The raw files list runners in finishing order. Any code that takes "the first row" of a race silently picks
   the winner. The code avoids this (and a test-time bug of exactly this kind was caught and fixed).

## How the pieces fit

```
raw CSVs → hri.pipeline → DuckDB tables → hri.ratings → hri.market → app.py (Streamlit)
                              │                │             │
                              │                │             └─ market_calibration, market_returns,
                              │                │                market_summary, baseline_eval
                              │                └─ rating_history (pre-race Elo), horse_ratings,
                              │                   ecuador_rankings, ecuador_rankings_by_season
                              └─ horses, races, race_entries, race_results, barrier_trials,
                                 horse_career_stats, predictions (empty), data_audit, view odds_final
```

Pre-race facts (`race_entries`) are kept apart from post-race facts (`race_results`). `rating_history` stores the
Elo *before* each race, so it is safe to use as a model feature. `races.market_complete` marks races whose
odds cover the whole field and a winner is recorded. Everything market-related uses only those.

## Ratings (Module A)

- **Race-level (JRA, HKJC):** a chronological multi-runner Elo (start 1500, K = 32) over every pair of finishers.
  - *overall_rating*: Elo shrunk toward 1500 by `n / (n + 3)`, so a horse needs evidence rather than a lucky
    start, and a horse with many starts isn't mistaken for a better one.
  - *consistency*: 1 − (std of within-field performance / 0.5). Only with ≥ 3 starts.
  - *recent_form*: exponentially weighted within-field performance, shrunk to 0.5.
  - Ratings are comparable only **within** a source.
- **Ecuador:** win and podium rates shrunk toward the pooled rate (10 pseudo-starts) combined with log earnings
  per start (weights 0.5 / 0.3 / 0.2), per season and combined.

## What the data can and can't support

These are measured results from the files in `data/raw`, not hopes.

**Market calibration (JRA, 834 complete races).** The market is well calibrated across probability bins.
Favourites win 35.7% of the time against 34.1% implied. The one clear departure is the usual
favourite–longshot bias: horses priced below 2% win about 0.5% (implied 0.8%).

**Returns.** Median overround is 1.26 (implied takeout about 21%). Backing every JRA runner at flat stakes
returns −40% (95% interval −46% to −34%); backing the favourite returns −18% (−26% to −10%); odds of 30+ lose 61%.
Nothing in JRA is profitable. In Hong Kong (41 complete races) every strategy's interval spans zero. For example,
"odds 30+" shows +42% with an interval of −100% to +333%. That is noise, not a finding.

**Model vs market (chronological holdout, fitted on earlier days only):**

| Source | Model | Test races | Test log loss | vs market (95% interval) |
|---|---|---|---|---|
| JRA | Market (final odds) | 289 | 1.866 | – |
| JRA | Elo softmax | 289 | 2.478 | +0.61 (+0.51, +0.71) |
| JRA | Uniform | 289 | 2.587 | +0.72 (+0.62, +0.83) |
| HKJC | Market (final odds) | 18 | 1.609 | – |
| HKJC | Elo / official rating | 18 | 2.269 | collapsed to uniform: no signal found in 23 training races |

Elo beats a coin flip in Japan but is far behind the market. In Hong Kong the training window is too small for
anything to beat uniform, and the 18-race test set can't separate models. So:

- **No predictive model is claimed.** A gradient-boosted or ranking model on ~870 JRA races with ~1.8 starts per
  horse would mostly learn from odds, draw and pedigree. Without the market it would not come close to it.
- Elo built from five weeks of HKJC results has a 0.09 correlation with the official rating. Use the official
  rating as the Hong Kong ability signal.
- The `predictions` table exists (empty) so the dashboard schema is stable when models are added.

## Limitations to keep in mind

- HKJC results are a partial sample; its market-complete subset is 41 races.
- JRA `horse_weight` in the raw file is the weight carried (kg), not body weight; it is stored as `carried_wt`.
- Margins and finish times are parsed to lengths and seconds. Short margins use fixed approximations
  (head 0.2, neck 0.3, short head 0.1, nose 0.05).
- Cross-country views are descriptive: different months, different years, different data completeness.
- Earnings are never compared across countries.

## Next steps (in order of payoff)

1. **More data beats more modelling.** The JRA README says this file is a ~3-month sample of a larger paid
   2024–2026 set. A full season would give horses real histories and a meaningful holdout. Fuller HKJC results
   (all runners, multiple seasons) are the other priority.
2. Re-run `python run.py` on the larger files; nothing in the pipeline is hard-coded to these row counts.
3. Add the predictor (Module B): conditional-logit baseline → LightGBM → ranking model, always evaluated against
   the market baseline on later race days, with features built only from `rating_history` (pre-race).
4. Obtain the Ecuador race-level records so it can join the race-level modules.
5. Optional: a separate identity-matching step for horses that race in more than one jurisdiction.

## Layout

```
run.py                 one-command build
app.py                 Streamlit dashboard
src/hri/parse.py       margin / time / ID parsers
src/hri/pipeline.py    load, clean, standardise → DuckDB
src/hri/ratings.py     Elo, shrunk summaries, Ecuador ranking
src/hri/market.py      calibration, returns, chronological baselines, CSV export
tests/test_hri.py      parsers, Elo leakage, shrinkage, database integrity
data/raw/              the original CSVs (untouched)
reports/               generated CSV summaries
```
