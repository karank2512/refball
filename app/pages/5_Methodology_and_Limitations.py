"""Methodology and responsible-interpretation / limitations page."""

from __future__ import annotations

import streamlit as st

from _shared import DISCLAIMER, page_header

st.set_page_config(page_title="Methodology & Limitations", page_icon="📐", layout="wide")
page_header(
    "Methodology & Limitations", "How the estimates are built, and what they can and cannot say."
)

st.warning(DISCLAIMER)

st.markdown(
    r"""
## The modeling spine

```text
Referee crew  →  foul environment / foul margin  →  free throws + game flow  →  point differential
```

We estimate this in three linked Bayesian stages (PyMC v5, non-centered parameterization).

### Stage 1A — Total foul volume
`total_fouls ~ NegativeBinomial(mu, alpha)` with a log link, a `log(possessions)` **offset**,
hierarchical home/away team effects, season effects, optional standardized market controls
(spread, total), and a **multi-membership** referee effect.

### Stage 1B — Directional foul lean
`home_pf` and `away_pf` each `~ NegativeBinomial`, sharing a crew **volume** effect and a crew
**lean** effect that enters `+` on the home mean and `−` on the away mean:

$$\log\mu_{home} = \alpha_h + \log(\text{poss}) + \text{crew}_{vol} + \text{crew}_{lean} + \dots$$
$$\log\mu_{away} = \alpha_a + \log(\text{poss}) + \text{crew}_{vol} - \text{crew}_{lean} + \dots$$

A **positive lean** ⇒ games with that official are associated with **more home-team fouls**
relative to away. Because `foul_diff_home = home_pf − away_pf`, positive lean means the home
team is *whistled more* — not *favored*.

### Stage 2 — Point differential
`point_diff_home ~ StudentT(nu, mu, sigma)` (robust to blowouts):

$$\mu = \gamma_0 + \gamma_{line} z_{spread} + \gamma_{total} z_{total} + \gamma_{foul} \,\text{foul\_diff} + \gamma_{fta}\,\text{fta\_diff} + \gamma_{ftm}\,\text{ftm\_diff} + \text{season}$$

`gamma_foul_diff` is the residual foul-margin association after the line and free throws are
accounted for. This stage is **associational** — fouls are themselves driven by game state,
aggression, strategy, and intentional fouling.

### Mediation
For each official we compose the posterior foul-margin association (in countable units) with
`gamma_foul_diff`, propagating the **full posterior**:

```text
indirect_points_r  =  foul_margin_r  ×  gamma_foul_diff
```

The result is a wide-interval **screening composite**, not a causal effect.
"""
)

st.markdown(
    """
## Why partial pooling

A referee who worked only a handful of playoff games should **not** shoot to the top of the
rankings on a tiny sample. The hierarchical model **shrinks** noisy estimates toward zero
unless the data repeatedly support an effect. The half-normal priors on the referee sigmas
are deliberately tight because playoff referee samples are small.

## Why multi-membership

A game has three officials. The model does **not** credit the whole game to a single "crew
chief", and does **not** duplicate the game three times as if officials were independent.
Each game contributes a row of `1/3` weights to a `[games × referees]` matrix, and the crew
effect is the **average** of the assigned officials' effects.

## Why this is not proof of bias

- **Assignment is not random.** Playoff crews are chosen by the league; better/veteran crews
  may be sent to bigger, more physical, more competitive games. That is textbook **selection
  bias** / endogeneity.
- **Lines help, but don't solve it.** The betting market encodes team strength and context,
  which helps control for *who is playing*, but it does not make referee assignment random.
- **Associational, not causal.** Even a clean foul-margin association can reflect team style,
  pace, or coaching, not officiating.
- **Outliers ≠ misconduct.** With dozens of officials and wide intervals, some will land at
  the extremes by chance. A screening signal is a prompt for scrutiny, not a verdict.

## What would count as (better) evidence

- **Many more games per official**, narrowing the intervals.
- **Possession-level / call-level data** (play-by-play fouls, shot locations, foul types) so
  the "whistle" is measured directly rather than via box-score totals.
- **A natural experiment** approximating random assignment.
- **Pre-registered** hypotheses and out-of-sample replication across future postseasons.
- **Referee variance that survives** the permutation placebo and leave-one-season-out checks
  by a wide, stable margin.

## Robustness checks we run

PSIS-LOO model comparison (no-ref vs ref-volume vs ref-volume+lean), a within-season
permutation **placebo**, **leave-one-season-out** stability, and **with/without-odds**
sensitivity. See the Diagnostics page.

## Data & reproducibility

Playoff box scores and officials come from `nba_api`; odds are loaded through a swappable
adapter that **never** joins on nba_api game ids. Everything is cached and reproducible from
the CLI. A synthetic generator with planted structure powers demo mode and the test suite.
"""
)

st.info(
    "The goal is not to 'prove Scott Foster is rigging games.' It is to build the best public, "
    "reproducible framework for asking: *how much do playoff referee crews appear to shape the "
    "whistle, and how much does that whistle show up on the scoreboard* — with honest uncertainty."
)
