# LinkedIn post

---

Every NBA playoff run, the same argument flares up: *"The refs decided that one."* Scott
Foster's name trends, box scores get screenshotted, and very little of it gets tested
rigorously.

So I built a small, honest framework to ask the question properly: **how much are referee
crews associated with the playoff whistle — and how much does that whistle actually show up
on the scoreboard?**

It's a Bayesian, observational project (PyMC + ArviZ), and it's deliberately un-sensational:

🏀 **Refs → fouls → point differential.** Three linked models estimate (1) total foul volume,
(2) a directional home-vs-away foul "lean," and (3) how foul and free-throw margins relate to
the final point differential — controlling for the betting line, pace, teams, and season.

📊 **Uncertainty everywhere.** Partial pooling means a referee with a few playoff games gets
shrunk toward zero, not crowned an outlier. A three-person crew is modeled with
multi-membership effects (no crew-chief-takes-all). Every ranking ships with a 94% credible
interval and a posterior probability — never a bare point estimate.

🧪 **Built to be wrong gracefully.** Permutation placebos (shuffle the crews), leave-one-
season-out checks, PSIS-LOO model comparison, and with/without-odds sensitivity. If referee
effects don't survive, the project says so.

The part I care about most is the framing. Referee assignments aren't random — better crews
may get bigger games — so this is **association, not proof, and certainly not a claim of
intent or misconduct.** A statistical outlier is a prompt for scrutiny, not a verdict.

Code is reproducible end-to-end, with an interactive Streamlit site (referee caterpillar
plots, a crew-level game explorer, full diagnostics, and a limitations page).

If you work in sports analytics or causal inference, I'd genuinely love your critique of the
identification strategy.

#NBA #SportsAnalytics #Bayesian #DataScience #PyMC #BasketballAnalytics
