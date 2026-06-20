# Ref Ball? I Tried Six Ways to Catch NBA Refs Swinging Games. Here's What the Data Says.

### A Bayesian, uncertainty-first investigation — built to find a signal if one exists, and honest enough to report when it doesn't

---

Every spring it happens. A playoff game tilts on a whistle, a star picks up a cheap third foul,
the free throws pile up on one side — and within minutes the internet has its verdict. *The refs
decided that one.* Scott Foster trends. Somebody posts a foul-margin screenshot. And then we all
move on, having "proven" exactly nothing.

I wanted to do the boring, useful version: build a transparent, reproducible framework that turns
"the refs decided it" into an actual estimate with an honest uncertainty interval — and then
spend most of the effort trying to *break* my own answer.

Let me be unambiguous up front, because this topic invites bad-faith readings. **Nothing here
claims any referee, crew, team, or league fixed, rigged, swung, or manipulated a game, or acted
with intent.** Referee assignments are not random — the NBA assigns playoff crews on merit,
trimming the pool by regular-season grades — so any pattern can be a *selection* effect, not a
*referee* effect. Everything below is an **observational screening metric**, reported with
uncertainty. A statistical outlier is a prompt for scrutiny, never a verdict.

With that said: I came at this six different ways, and they all land in the same place.

## What I tested

The causal-looking spine is simple:

> referee crew → foul environment / foul margin → free throws + game flow → point differential

Concretely, three questions. Do specific officials get associated with **more fouls**? With a
directional **home-vs-away foul margin** (a "lean")? And does that foul margin actually **move
the final score**, once you control for team strength? Then I compose them into a per-referee
"scoreboard swing." Crucially — a *positive* home lean means the home team is **whistled more**,
which is the opposite of being favored.

## The data (all public)

- **Box scores + officials** for **585 playoff games (2017-18 through 2023-24)** from the NBA's
  stats API via `nba_api` — 99.8% with the assigned crew, 58 distinct playoff officials.
- **8,289 regular-season games** over the same span — 102 referees at a median of **~293 games
  each** (vs. ~21 in the playoffs). This is the dataset that gives the study real *power*.
- **Play-by-play** for every playoff game — to type each foul (shooting / personal / loose-ball
  vs. offensive / technical) and tag the score and clock when it happened.
- The NBA's own **Last Two Minute (L2M) reports** — graded calls/non-calls in the final two
  minutes of close games — via a public, MIT-licensed mirror.
- **Closing betting lines** (a public 10-year archive) as a market-based control for team
  strength.

## The models

This is Bayesian hierarchical modeling (PyMC). Three design choices do the heavy lifting.
**Partial pooling:** a referee with a handful of games is shrunk toward zero unless the data keep
supporting an effect. **Multi-membership:** a game has three officials, so each contributes
one-third of a "crew" effect — no crew-chief-takes-all, no triple-counting. **Uncertainty
everywhere:** every number ships with a 94% credible interval and a posterior probability, never
a bare point estimate. I validated the whole pipeline on synthetic data with *known* planted
effects; it recovers them in the right direction and stays humble when the sample is thin.

## What I found

**Fouls barely move the scoreboard.** The home foul margin is almost uncorrelated with the final
margin (r ≈ 0.02). Modeled properly, the foul-margin → point-differential coefficient is
**+0.03 points per foul (94% HDI −0.23 to +0.28)** — indistinguishable from zero. Game state is
why: when a team trails late it fouls *on purpose*, and garbage-time fouls don't matter. Even
after I rebuilt the margin from *discretionary* fouls in *competitive* states, the coefficient
only nudged to **−0.14 (HDI −0.40 to +0.13)** — the expected direction, but still inconclusive.
The scoreboard effect of fouls runs almost entirely through *made free throws* (a strong,
unsurprising +0.95 points per made-FT-margin), not through any residual "extra whistle."

**No referee shows a detectable foul lean — in the playoffs.** Zero of 58 officials have a
home-lean interval that clears zero, and when I shuffle the crews across games (a placebo) the
lean variance is statistically identical (0.013 vs. 0.013). *But* — and this is the honest part
most analyses skip — I checked whether the method even *could* detect a real lean. I injected a
known, swing-relevant **+1 foul/game** lean into the most-sampled referee and refit: **not
detected.** Even **+2 foul/game** wasn't reliably recovered. With a few hundred playoff games,
the design is simply **underpowered**, so "no playoff lean" wasn't yet a fair claim.

**So I went and got the power.** I re-ran the lean model on **8,289 regular-season games**, where
referees work ~290 games apiece — more than enough to surface a real tendency. The result is the
strongest in the project: the between-referee lean variance didn't grow, it **shrank** (to
≈0.005); **0 of 102** referees cleared zero; and the clincher — a referee's *playoff* lean is
**uncorrelated** with their high-power *regular-season* lean (Spearman ≈ 0.01). In plain terms,
the little playoff leans aren't a stable referee trait — **they're sampling noise.** And **0 of
102** referees officiated the playoffs detectably differently from their own regular-season
baseline. So much for "brought in for the big games."

**Holding the matchup fixed doesn't change it.** A playoff series is the same two teams with a
rotating crew, so I added series × team effects — identifying the crew lean purely from the
crew rotation *within* a series. Same answer: 0 of 58 distinguishable, variance at placebo level.

**The NBA's own grades agree.** Using the league's L2M reports — its own labels of Correct /
Incorrect Call / Non-Call in clutch moments — I signed each error by which team it hurt across
182 close playoff games (277 graded errors). Result: **0 of 55 crews** with a distinguishable
error-lean, and only a faint, **inconclusive** leaguewide tilt toward home (errors favored the
home team about **1.18×**, but the 94% interval includes zero). Caveats matter here: L2M exists
only for close-and-late games, it grades the *crew* not the individual official, and it's the
NBA grading its own calls.

**The betting line doesn't reveal a hidden swing either.** Adding the real closing spread to the
332 games I could match barely moved the foul coefficient (−0.12 with the line vs. −0.17
without), and referee rankings stayed stable.

## The one real signal: home court

The thing the data *does* show is small and old news: home teams are whistled slightly *less*
(about −3% on fouls), worth a fraction of a point. That's consistent with decades of work
(*Scorecasting* attributes ~three-quarters of home-court advantage to officiating tendencies).
It is a leaguewide, well-documented home tilt — not a specific crew tipping specific games.

## I tried to prove the opposite

A null is easy to *want*, so I spent a full pass as the prosecution: assume there's no swing, and
try as hard as a motivated analyst could to manufacture evidence that there is one. You can get
scary numbers. Here are four I built — and why each is a magic trick.

**"Referee Josh Tiven favors home — 73% of his clutch errors went the home team's way, p=0.015."**
True, if you pick the most extreme of 32 referees and quote it one-sided. Corrected for having
looked at 32, the p-value is **0.46**, and a coin-flip simulation produces *someone* this extreme
**24% of the time**. It's the luckiest of 32 coin-flippers, nothing more.

**"Refs protect the favorite in close games — a 1.7-foul tilt, p=0.002."** That's the best of 18
post-hoc slices. Home *underdogs* in close games get the same tilt — so it's home court, not
favoritism — and fouls barely move the scoreboard anyway (the foul-to-points correlation is 0.02).

**"One referee tops every leaderboard — foul margin, points, *and* the league's own missed-call
audit. Convergence!"** With 58 referees, someone tops each noisy list by construction. The killer
check: that referee's playoff "lean" **flips sign** in his 427-game regular-season record. Across
all referees, playoff lean and regular-season lean correlate at **0.01** — a playoff lean tells
you essentially nothing about the referee.

The **only** number that survives is the leaguewide home tilt (+0.5 free-throw attempts a game) —
and a placebo that shuffles referee labels shows it attaches to *home court*, not to any
referee's *identity* (p=0.22). Every "smoking gun" is multiple comparisons, a cherry-picked
subgroup, or regression to the mean in a costume. Trying my hardest to fake a positive result,
and watching all of it dissolve, is the strongest confirmation of the null I could ask for.

## What would count as evidence

I held a high bar, and I'll keep holding it. What would move me: a referee lean that **survives**
the placebo and leave-one-season-out checks by a wide, stable margin; effects that **persist**
from regular season to playoffs (mine don't correlate at all); **call-level** data showing a
directional pattern in *discretionary* whistles; or something approximating **random assignment**
to break the selection problem. A single eye-popping point estimate with a bus-wide interval is
not evidence — it's a hypothesis.

## Limitations

Two big ones. **Assignment isn't random** — better crews may draw bigger games, so this is
association, not cause; the within-series design narrows that gap but doesn't close it. And the
models are **observational**: foul margins reflect team style, pace, coaching, and intentional
fouling, not just officiating. The L2M layer is selected on close games. And my regular-season
production fit, while converged, is one run — I'd want repeated full-length chains before
treating any single decimal as gospel. None of this changes the direction of the result; it
bounds how loudly I'm willing to state it.

## Why this matters

Officiating is one of the most emotionally charged, least rigorously examined parts of sports
discourse. The point isn't to indict anyone or to "clear" anyone — it's to give fans, analysts,
and skeptics a **shared, honest tool**, complete with the uncertainty the question deserves. The
most useful thing a project like this can do is be willing to find nothing, loudly, when the data
say nothing.

## Closing

The Scott Foster of it all? He worked more of these playoff games than anyone in the sample, and
his estimated lean is slightly *negative*, with a mediated scoreboard effect indistinguishable
from zero. The cultural hook lands on a null — which is exactly why you run the numbers instead
of the screenshots.

The honest bottom line: across six independent angles — including a genuinely high-powered one —
there is **no detectable referee or crew swing in NBA playoff games**, only a faint, long-known
home tilt. That's not proof referees never matter; it's the strongest statement observational
data can make: *when we finally had the power to find an effect, we didn't.*

The code, the models, and the interactive site are open. If you do causal inference or sports
analytics, please try to break the design — that's the whole idea.

*This is an independent analytics project, not affiliated with or endorsed by the NBA or any team
or official. Nothing here asserts intent or misconduct by any person.*
