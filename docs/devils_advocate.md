# The Prosecution's Case: Can We *Manufacture* a Referee Swing?

> A red-team exercise. Having spent the project as the *defense* (trying to break the null), I
> flipped sides: assume there's no swing, and try as hard as a motivated analyst possibly could
> to build evidence that there **is** one — using only the real data. Then show exactly which of
> those arguments are legitimate and which are the classic ways you fool yourself.
>
> Reproduce every number here with: `python -m refball.models.devils_advocate`

**Spoiler:** you *can* produce scary-looking numbers. Every one of them dies under a standard
correction. The only thing that survives is the small, well-documented, **crew-invariant** home
tilt — which is home-court advantage, not a referee swing, and says nothing about intent.

---

## Exhibit A — "Referee Josh Tiven favors the home team" (extreme-ref mining)

**The claim:** In the NBA's own Last Two Minute graded calls, **19 of Tiven's 26** clutch
officiating errors went against the road team — **73% in the home team's favor**. A fair-coin
referee does that with probability **0.0145**. He's the single most home-favoring official in the
league's own audit.

**How it was built:** I attributed each game's L2M errors to its three crew members, tallied
favor-home vs. favor-away per referee, kept the 32 refs with ≥10 graded errors, ran a one-sided
binomial on each, and **quoted the single most extreme of 32**.

**Why it collapses:** It's pure max-of-32 selection. Bonferroni-corrected **p = 0.46**;
Benjamini-Hochberg survivors at q=0.05: **zero**. A null simulation (every crew a fair coin) shows
*some* referee looks at least this extreme **24% of the time** by chance. And every *shrunk*
model agrees the most extreme referee's posterior P(home lean) caps at ~0.61 — **0/58** playoff,
**0/55** L2M, **0/58** within-series, **0/102** regular-season refs clear zero.

---

## Exhibit B — "Refs protect the favorite in close games" (forking-paths subsetting)

**The claim:** Among home favorites in close games, the home side is whistled for **1.71 fewer
fouls** than its opponent (t=−3.2, **p=0.0024**, n=58) — a 2-foul tilt toward the team the market
already favored, in exactly the games decided at the whistle.

**How it was built:** I sliced the data **18 ways** (round, favorite/underdog, close/blowout, and
intersections) and crowned the single best-looking cell — a triple-intersected subgroup chosen
*after* seeing the results.

**Why it collapses:** One of 18 tests → Bonferroni **p ≈ 0.04**, right at the edge with no margin.
Worse, it's **mislabeled**: home *underdogs* in close games get the *same* tilt, so it's
home-court advantage, not "favorite protection." And there's **no swing mechanism** — leaguewide,
the foul margin barely moves the scoreboard at all (slope **0.054**, p=0.69, r=0.017). A foul-count
gap that doesn't translate to points cannot "swing" anything.

---

## Exhibit C — "Officiating systematically favors home" (the one real number, over-framed)

**The claim:** Across 8,289 regular-season games, home teams shoot **+0.52 more free-throw
attempts per game** — a **5.4-sigma** effect (**p = 8.3 × 10⁻⁸**). Not chance.

**This part is real** — pre-specified, full-sample, no subsetting. **But** here's the decisive
test: does any of it attach to *referee identity* beyond the home tilt every crew shares? A
**label-permutation placebo** — shuffling which referee is tagged to each game — says **no**: the
real between-referee spread in home-FTA tilt is statistically indistinguishable from randomly
relabeled crews (**p = 0.22**). The home tilt is a single shared constant, not 90 distinct biased
referees. So the only surviving claim is **a small, leaguewide, crew-invariant home-court
officiating tilt** — the long-documented home-advantage phenomenon — **not** a referee or crew
swinging specific games, and nothing about intent.

---

## Exhibit D — "One referee tops every leaderboard" (selection + regression to the mean)

**The claim:** Tiven is #1 of 58 on playoff foul margin, #1 on mediated points, *and* #1 of 55 on
L2M error-lean. "Convergence you can't hand-wave away."

**Why it collapses — the killer check:** Take the referee who tops the *playoff* leaderboard and
look at their **high-power regular-season** record. Tiven's playoff lean (+0.0057, 66 games)
**flips sign to −0.0013** over 427 regular-season games — it regresses *past* zero. Across all
referees, the correlation between a ref's playoff lean and their regular-season lean is **Spearman
0.012 (p=0.93)** — a playoff "lean" carries essentially **no** information about the referee's
actual tendency. The leaderboard-topper is just the most extreme draw of noise, by definition.

---

## Exhibit E — "Maybe it's not all refs, just one bad apple" (the specific bad-actor test)

The fairest version of the prosecution: forget the *average* referee — is there *one* official
who swings games, even if most don't? This is the Scott Foster hypothesis, and it deserves a
direct test, because a population-level null doesn't automatically rule out a lone outlier.

Two checks, on the high-power regular-season data (102 referees, ~293 games each):

1. **Per-referee, at power.** With ~293 games apiece, a real bad actor would surface. None does:
   **0 of 102** referees have a 94% interval excluding zero, and the single most extreme lean in
   the league is **≈ 0.04 fouls per game** — and it belongs to Brett Nansel, not anyone with a
   reputation. The "prime suspects" all have intervals straddling zero.

2. **Within-referee split-half replication (model-free, no shrinkage).** A genuine bad actor's
   lean should reappear in a fresh batch of their own games. Splitting each referee's games in
   half, the correlation between a referee's home-foul lean on one half and the other is
   **Pearson ≈ 0.19 (p ≈ 0.10) — small and not statistically significant** (and it drops toward
   zero with noisier splits). Even taken at face value, this *raw* measure conflates which teams
   and venues a referee happens to draw; the team-and-season-controlled Bayesian model removes
   those confounds and lands at the 0/102 above.

So a specific bad actor would need to be **both** real-sized (distinguishable from zero at power)
**and** reproducible (lean replicates across their games). No individual referee satisfies
either condition decisively, let alone both.

## Verdict

| Exhibit | Manufactured headline | Defeated by | Survives? |
|---|---|---|---|
| A — extreme ref | Tiven 73% home, p=0.015 | Bonferroni 0.46; 24% by chance | ❌ |
| B — best subgroup | −1.7 foul tilt, p=0.002 | multiplicity; same for underdogs; fouls≠points | ❌ |
| C — home tilt | +0.52 FTA/game, p=8e-8 | real, but identity-placebo p=0.22 → crew-invariant | ✅ *as home-court only* |
| D — one ref, 3 metrics | #1 everywhere | regression to mean (sign flips); Spearman 0.01 | ❌ |
| E — one bad apple | a specific guilty referee | 0/102 at power; max ≈0.04 fouls/game; split-half n.s. | ❌ |

Trying as hard as I could to prove a swing — and watching every case dissolve under standard
corrections — is the **strongest possible confirmation of the null**. The only thing that
survives is the boring, decades-old, leaguewide home-court tilt. There is no referee- or
crew-specific swing in this data, and nothing here speaks to intent.

*This remains an observational project. "No detectable swing" is not "referees never matter" — it
is the strongest statement the public data supports, made honestly from both directions.*
