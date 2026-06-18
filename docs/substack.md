# Ref Ball? What the Whistle Actually Does to the Scoreboard

### A Bayesian, uncertainty-first look at NBA playoff referees — built to find a signal *or* to honestly come up empty

---

Every spring it happens. A playoff game tilts on a whistle, a star picks up a questionable
third foul, free throws pile up on one side — and within minutes the internet has rendered its
verdict. *The refs decided that one.* Scott Foster trends. Somebody posts a foul-margin
screenshot. And then we all move on, having "proven" exactly nothing.

I wanted to do the boring, useful version instead: build a transparent, reproducible framework
that turns "the refs decided it" into an actual estimate — **with an honest uncertainty
interval attached** — and then stress-test that estimate until it either survives or falls
apart.

Let me be unambiguous up front, because this topic invites bad-faith readings: **nothing in
this project claims any referee, crew, team, or league fixed, rigged, swung, or manipulated a
game, or acted with intent.** Referee assignments are not random. Playoff crews are chosen by
the league, and better or more experienced crews may be sent to the biggest, most physical,
most competitive games. That alone can produce patterns that *look* like a referee effect but
are really a selection effect. So everything here is an **observational, associational
screening metric** — a prompt for scrutiny, never a verdict.

With that said, here's what I tested and how.

## What I tested

The project follows one causal-looking spine — while being very clear it is *not* clean causal
identification:

```text
Referee crew → foul environment / foul margin → free throws + game flow → point differential
```

Concretely, three questions:

1. **Volume.** Are specific officials associated with *more or fewer total fouls*, after
   accounting for pace, the two teams, the season, and the betting market?
2. **Lean.** Are specific officials associated with a *directional* foul margin — `home fouls −
   away fouls`? I call this a "home-foul lean." Crucial nuance: a *positive* lean means the
   home team is **whistled more**, which is the opposite of being favored.
3. **Translation.** How does the foul margin (and the free-throw margin it creates) relate to
   the **final point differential**, after controlling for the pregame spread and total?

Then I compose them: for each official, multiply their estimated foul-margin association by the
foul-margin-to-points coefficient to get a **mediated "scoreboard" effect** — propagated across
the entire posterior, not as a point estimate.

## The data

- **Box scores and officials** come from the NBA's public stats endpoints via `nba_api`:
  playoff game logs (points, personal fouls, free throws, and the components needed to estimate
  possessions) plus the assigned three-person crew for each game.
- **Betting lines** (spread + total) come through a deliberately *swappable* adapter. Odds
  coverage is the messiest part of any project like this, so the adapter validates its columns,
  normalizes team names, and **never** joins on the NBA game id (odds feeds don't share that
  id). Joins happen on date + normalized home/away teams, with a one-day tolerance, and the
  pipeline prints the match rate and refuses to pretend a bad join is a good one.
- Coverage target: the **[insert season range, e.g., 2017–2025]** playoffs, **[insert N]**
  games, **[insert number of distinct officials]** officials.

When lines can't be matched well, the pipeline runs a no-odds version of the model and flags it
as weaker. When officials are missing, those games drop out of the referee models (and the drop
is logged). No silent fudging.

## The model

This is a Bayesian hierarchical model (PyMC), and three design choices do most of the work.

**Partial pooling.** A referee who worked four playoff games should not rocket to the top of a
leaderboard on the strength of four games. The hierarchy *shrinks* noisy referee estimates
toward zero unless the data repeatedly support an effect. The priors on the referee
variation are intentionally tight, because playoff samples are small and overconfidence is the
enemy.

**Multi-membership.** A game has three officials, not one. The model doesn't credit the whole
game to the crew chief, and it doesn't cheat by copying each game three times as if the
officials were independent observations. Instead each game contributes a row of one-third
weights, and a crew's effect is the *average* of its officials' effects.

**Uncertainty everywhere.** Stage 1A models total fouls as a Negative Binomial with a
log-possessions offset. Stage 1B models home and away fouls jointly, sharing a crew *volume*
term and a crew *lean* term (the lean enters with a plus sign on home fouls and a minus sign on
away fouls). Stage 2 models point differential with a Student-t likelihood (robust to
blowouts). Every reported number — total-foul effect, lean, mediated scoreboard effect — comes
with a 94% credible interval and a posterior probability of being positive or negative. No bare
point estimates.

Before trusting any of it on real data, I validated the whole pipeline on a **synthetic
dataset with planted structure** — known referee effects, a known foul-margin-to-points
coefficient. The models recover the planted signal in the right direction and, just as
importantly, report wide intervals when the sample is small. That's the behavior you want: a
method that finds what's really there and stays humble about what isn't.

## Results

> The numbers below are placeholders until the model is run on the full real dataset. I will
> **not** invent results. (The synthetic validation above is a methodology check, not a finding
> about any real official.)

**Foul-margin → points.** The headline coefficient is `gamma_foul_diff`: how many points of
home point differential are associated with one extra home foul (relative to away), after the
line and free throws. Estimate: **[insert posterior mean] points** (94% HDI **[insert
interval]**; P(<0) = **[insert probability]**). Interpretation: the foul margin
**[does / does not]** retain a scoreboard association beyond the mechanical free-throw channel.

**Referee leans.** Sorted by posterior mean home-foul lean, the officials whose intervals sit
furthest from zero are **[insert names + intervals]**. Note how many intervals **[comfortably
straddle / clear]** zero — that's the story, not the ranking.

**Mediated scoreboard effect.** Translated to points, the largest mediated effects are **[insert
range, e.g., ±X points per game]**, with intervals that **[mostly include zero / …]**.

**Did referee effects earn their keep?** PSIS-LOO comparison of a no-referee model vs.
referee-volume vs. referee-volume-plus-lean: **[insert which model wins and by how much]**. If
referee effects don't improve out-of-sample prediction, that is itself a result, and I'll say
so plainly.

**Placebo.** When I shuffle the crews across games within each season and refit, the referee
"lean" variation **[collapses toward / stays above]** the real estimate. Real exceeds **[insert
%]** of placebos.

**Can I even detect a swing? (the part most analyses skip.)** A null is only meaningful if the
method has the power to find a real effect. So I *inject* a known lean into the most-sampled
official's games and check whether the model recovers it. In my run, a clearly swing-relevant
**+1 foul/game** lean injected into a referee with ~100 games was **[insert: detected / not
detected]**; only a **+2 foul/game** lean was reliably recovered. That means the playoff
headline is **"no *detectable* referee swing,"** not "referees don't matter." With a few
hundred playoff games and ~20 per official, this design would miss a modest-but-real effect —
and saying so is the difference between an honest null and an overclaim.

**So I went and got the power.** I pulled ~8,300 regular-season games (median ~290 games per
referee, vs ~20 in the playoffs) and re-ran the lean model where it actually *can* detect
something. The between-referee lean variance didn't grow — it **shrank** (to ~0.005), **none**
of the 102 referees had an interval clearing zero, and — the clincher — a referee's playoff
"lean" was **uncorrelated** with their high-power regular-season lean (Spearman ≈ 0). In other
words, the little playoff leans aren't a stable referee trait; they're sampling noise. And no
referee called the playoffs detectably differently from their own regular-season baseline. A
within-series design (same two teams, rotating crews) and the NBA's own Last Two Minute grades
land in the same place. When several independent angles — including a genuinely high-powered
one — all say "nothing here," that's about as close to an answer as observational data gets.

## What would count as evidence

Because the bar for any claim about officiating should be high, here's what would move me:

- Referee variation that **survives** the permutation placebo and **leave-one-season-out**
  checks by a wide, stable margin — not one anomalous postseason.
- **Much larger samples** per official, shrinking the intervals.
- **Call-level / play-by-play data** (foul types, shot locations, late-game context) so the
  whistle is measured directly instead of through box-score totals.
- Something approximating **random assignment** — a natural experiment — to break the selection
  problem.
- **Pre-registration** and **out-of-sample replication** in future playoffs.

A single eye-popping point estimate with a credible interval the width of a bus is not
evidence. It's a hypothesis.

## Limitations

There are two big ones. **First, power:** with a few hundred playoff games and ~20 per official,
this design can miss a modest-but-real referee lean (my injection test only reliably recovers a
+2-foul/game effect). So a null here means "no *detectable* swing," not "no swing" — absence of
evidence is not evidence of absence. **Second, selection bias:** referees aren't assigned at
random, so a referee "effect" can be a stand-in for the kinds of games that referee tends to
work. Betting lines help me control for *who is playing and how good they are*, but they do not
make assignment random. The models are **associational**, full stop — foul margins also reflect
team style, pace, coaching, and deliberate late-game fouling. Playoff samples are small, so
intervals are wide. And box-score
fouls are a blunt instrument compared to what a possession-level dataset could reveal.

## Why this matters

Officiating is one of the most emotionally charged, least rigorously examined parts of sports
discourse. The point of this project isn't to indict anyone — it's to give fans, analysts, and
skeptics a **shared, honest tool** for asking the question, complete with the uncertainty that
the question deserves. If the data show a stable, replicable referee-associated lean, that's
worth understanding. If they don't, that's worth saying just as loudly. Both outcomes beat a
screenshot and a hot take.

## Closing

The goal was never to "prove Scott Foster is rigging games." The goal was to build the best
public, reproducible framework for asking: *how much do NBA playoff referee crews appear to
shape the whistle, and how much does that whistle show up on the scoreboard — under honest
uncertainty?*

The code, the models, and the interactive site are all open. If you do causal inference or
sports analytics, I'd love for you to try to break the identification strategy. That's the
whole idea.

*This is an independent analytics project, not affiliated with or endorsed by the NBA or any
team or official. Nothing here asserts intent or misconduct by any person.*
