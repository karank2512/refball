# LinkedIn post

---

Every NBA playoff run, the same argument flares up: *"the refs decided that one."* Scott
Foster's name trends, a foul-margin screenshot goes viral, and almost none of it gets tested.

So I built a framework to ask the question properly and then spent
most of my effort trying to *break* my own answer.

🏀 The spine: refs → fouls → point differential, estimated with Bayesian partial pooling (PyMC).
A three-person crew is modeled with multi-membership effects; every estimate ships with a
credible interval, not a point estimate.

📉 First pass: across **585 playoff games**, no referee shows a detectable home-foul lean. But an
injection-recovery test showed the playoffs are *underpowered* — so "no effect" wasn't yet a fair
claim.

🔬 So I went and got the power: **8,289 regular-season games (~290 per referee)**. The model
*could* now detect a lean and still found none. The lean variance *shrank* toward zero, **0 of
102** referees cleared zero, and a referee's playoff "lean" was **uncorrelated** with their
high-power regular-season lean (Spearman ≈ 0.01). The playoff leans are sampling noise.

Five more angles agree — within-series identification, the NBA's own Last Two Minute clutch-call
grades (0/55 crews), play-by-play game-state cleaning, and a closing-line control. The only real
signal is a faint, well-documented *home* tilt.

This isn't proof refs are clean, and it's **not** a claim of bias or intent — assignment isn't
random. It's the strongest observational answer the public data supports: **no detectable
referee swing, even with the power to find one.**

Code + interactive site are open. If you do causal inference or sports analytics, try to break it.

#NBA #SportsAnalytics #Bayesian #DataScience #PyMC
