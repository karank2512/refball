# Reproduce Ref Ball from scratch

Two paths: a **2-minute offline demo** (synthetic data, proves the core pipeline runs) and a
**full real reproduction** (every finding in the project, ~4–6 hours, mostly a polite NBA data
pull). Everything is cached, so the full run resumes if interrupted.

## 0. Prerequisites

- **Python 3.11+**, **git**, and an internet connection.
- ~5 GB free disk (raw caches: officials, play-by-play, L2M, odds).
- Recommended: [`uv`](https://docs.astral.sh/uv/) (fast). Plain `pip` works too.

## 1. Clone + environment

```bash
git clone https://github.com/karank2512/refball.git
cd refball

# --- with uv (recommended) ---
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .                # makes `python -m refball.*` work without PYTHONPATH

# --- or with plain pip ---
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt && pip install -e .
```

Sanity check:

```bash
python -c "import refball, pymc, arviz, streamlit, nba_api; print('env OK')"
pytest -q          # 17 smoke tests, ~1-2 min
```

## 2. Quick demo (offline, ~2 min)

Synthetic data with planted structure — confirms the core foul models, mediation, robustness,
and the website all run before you commit to the long real pull.

```bash
bash scripts/run_demo.sh
streamlit run app/Home.py          # open http://localhost:8501
```

> The app shows a **Demo mode** banner while synthetic data is loaded. The L2M, regular-season,
> play-by-play, and devil's-advocate streams need real data (next step).

## 3. Full real reproduction (~4–6 hours)

One command runs the entire project in order:

```bash
bash scripts/run_full.sh
```

Or step through it yourself (same order the script uses):

```bash
# 1. Playoff games + officials (~15-20 min, rate-limited)
python -m refball.data.pull --season-start 2017 --season-end 2023

# 2. NBA Last Two Minute graded calls (MIT-licensed mirror; instant)
python -m refball.data.l2m

# 3. Closing-line odds archive (instant download)
python -c "from refball.data.odds import download_odds_archive as d; print(d())"

# 4. Build the modeling table, joined to the closing line
python -m refball.features.build_table --odds data/external/nba_odds_archive.json

# 5. Headline foul models (no-odds is the primary 585-game basis)
python -m refball.models.fit_stage1 --no-odds
python -m refball.models.fit_stage2 --no-odds
python -m refball.models.mediation

# 6. Robustness: LOO, permutation placebo, leave-one-season-out, sensitivity, power
python -m refball.models.robustness --full

# 7. EDA figures
python -m refball.features.eda

# 8. Direct evidence — L2M crew clutch-error model
python -m refball.models.l2m_model

# 9. Identification — within-series fixed effects
python -m refball.models.within_series

# 10. POWER — regular-season pull (~2-3 HOURS) + the high-power lean model
python -m refball.data.pull --season-type "Regular Season" --season-start 2017 --season-end 2023
python -m refball.models.regular_season

# 11. Play-by-play foul-type + game-state cleaning
python -m refball.data.pbp
python -m refball.models.pbp_clean

# 12. Devil's advocate — try to PROVE a swing, watch it collapse
python -m refball.models.devils_advocate

# Launch the site
streamlit run app/Home.py
```

### Faster first pass

Add `QUICK=--quick` to do rough, short-chain sampling everywhere (~30 min of compute instead of
hours; diagnostics will read "CHECK" — fine for a sanity pass, not for publishing):

```bash
QUICK=--quick bash scripts/run_full.sh
```

### What you'll have when it finishes

The website (Home + Data Coverage, Referee Effects, Game Explorer, Diagnostics, Methodology,
Clutch Calls/L2M, Evidence Streams) showing the production numbers, plus result tables under
`data/processed/` and traces under `models/`.

## 4. Troubleshooting

- **Parallel-sampling error** (`EOFError` / multiprocessing): force sequential chains with
  `export REFBALL_MCMC_CORES=1` (also `REFBALL_MCMC_QUICK_CORES=1` if you used `--quick`). Rare on
  a normal desktop; common in sandboxes.
- **nba_api timeouts / `KeyError: 'resultSet'`**: you're being rate-limited — don't run two pulls
  at once, and bump the politeness delay: `export REFBALL_REQUEST_MIN_SLEEP_S=1.0`, then re-run
  (the cache resumes). Note `playbyplayv3` is used because `playbyplayv2` is broken in current
  nba_api.
- **Newest playoffs missing officials**: `BoxScoreSummaryV2` flags gaps for games on/after
  2025-04-10; pulling through `--season-end 2023` avoids it (those games drop out gracefully).
- **Odds coverage is partial**: the public archive covers seasons ~2017–2021, so ~332 of 585
  playoff games match. For full coverage, drop a tidy CSV at any path and use
  `build_table --odds your.csv` (see the README "Odds CSV format").
- **Want it faster on a strong machine**: the default sampler uses 4 parallel chains; leave
  `REFBALL_MCMC_CORES` unset to use them.
