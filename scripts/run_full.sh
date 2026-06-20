#!/usr/bin/env bash
# FULL real reproduction of the entire project from scratch.
# Run from the repo root with the virtualenv ACTIVATED (see docs/REPRODUCE.md).
#
# Wall-clock: ~4-6 hours, dominated by the regular-season officials pull (~2-3 hrs of polite,
# rate-limited nba_api calls) and the full Bayesian fits. Everything is cached under data/raw,
# so if it's interrupted you can just re-run this script and it resumes where it left off.
#
# Env knobs:
#   SEASON_START / SEASON_END   starting calendar years (default 2017..2023 = 2017-18..2023-24)
#   QUICK=--quick               fast, rough sampling (~30 min total instead of hours) for a first pass
#   REFBALL_MCMC_CORES=1        force sequential chains if parallel sampling errors on your machine
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
SS="${SEASON_START:-2017}"; SE="${SEASON_END:-2023}"; QUICK="${QUICK:-}"

echo "== 1/12 playoff games + officials (~15-20 min) =="
python -m refball.data.pull --season-start "$SS" --season-end "$SE"

echo "== 2/12 NBA Last Two Minute graded calls (MIT mirror) =="
python -m refball.data.l2m

echo "== 3/12 closing-odds archive (download) =="
python -c "from refball.data.odds import download_odds_archive as d; print('odds ->', d())"

echo "== 4/12 build the modeling table (with the closing line) =="
python -m refball.features.build_table --odds data/external/nba_odds_archive.json

echo "== 5/12 headline foul models (no-odds is the primary basis) =="
python -m refball.models.fit_stage1 --no-odds $QUICK
python -m refball.models.fit_stage2 --no-odds $QUICK
python -m refball.models.mediation

echo "== 6/12 robustness (LOO / placebo / LOSO / sensitivity / power) =="
python -m refball.models.robustness "${QUICK:---full}"

echo "== 7/12 EDA figures =="
python -m refball.features.eda

echo "== 8/12 L2M crew clutch-error model =="
python -m refball.models.l2m_model $QUICK

echo "== 9/12 within-series identification =="
python -m refball.models.within_series $QUICK

echo "== 10/12 regular-season pull (~2-3 HOURS) + power model =="
python -m refball.data.pull --season-type "Regular Season" --season-start "$SS" --season-end "$SE"
python -m refball.models.regular_season $QUICK

echo "== 11/12 play-by-play foul-type + game-state cleaning =="
python -m refball.data.pbp
python -m refball.models.pbp_clean $QUICK

echo "== 12/12 devil's advocate (try to prove a swing) =="
python -m refball.models.devils_advocate

echo
echo "FULL REPRODUCTION DONE. Launch the site:  streamlit run app/Home.py"
