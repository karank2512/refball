#!/usr/bin/env bash
# Fast OFFLINE demo on synthetic data (~2 min). Proves the core pipeline runs end to end.
# Run from the repo root with the virtualenv ACTIVATED (see docs/REPRODUCE.md).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

echo "== 1/6 synthetic data =="            ; python -m refball.data.synthetic
echo "== 2/6 build modeling table =="      ; python -m refball.features.build_table
echo "== 3/6 EDA figures =="               ; python -m refball.features.eda
echo "== 4/6 Stage 1 + 2 (quick) =="       ; python -m refball.models.fit_stage1 --quick
                                            python -m refball.models.fit_stage2 --quick
echo "== 5/6 mediation =="                 ; python -m refball.models.mediation
echo "== 6/6 robustness (quick) =="        ; python -m refball.models.robustness --quick

echo
echo "DEMO DONE. Launch the site:  streamlit run app/Home.py"
echo "(The L2M / regular-season / play-by-play / devil's-advocate streams need the real data —"
echo " see scripts/run_full.sh.)"
