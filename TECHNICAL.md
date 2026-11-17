# Cochinae — frame-level possession from automatic tracking

**Team:** Cochinae (Sebastian Dulong Salazar, Santiago Fernández del Castillo, Ulises Quevedo)  
**Challenge:** More than a Hack 2027, Phase 1 — Ball Possession (FC Barcelona vs Athletic Club)  
**Code:** https://github.com/dulongski/cochinae-mtah-2027 · `./reproduce.sh` regenerates `Cochinae.csv` byte for byte.

---

## 1. Approach in one paragraph

We read possession from the geometry of the 22 players and the ball. We learn that reading on
**external data only**: 20 public SkillCorner matches whose provider "phases of play" say which
team has the ball. Two gradient-boosted classifiers, one for the team in possession and one for
dead ball, see the same 121 features on SkillCorner and on BAR–ATH. A Viterbi decoder with
switching costs and a minimum stoppage length turns their per-frame probabilities into whole
possessions. Every decoder setting was chosen by cross-validation on SkillCorner. **Nothing is
trained, fitted or tuned on the BAR–ATH ground truth or video**, as the organisers' 3 November
clarification requires. The public 5 minutes are scored only afterwards, as a check.

## 2. Submitted pipeline (`run_external()` in `cochinae/model.py`)

| stage | what it does | data used |
|---|---|---|
| `load.py` | Parses the tactical ATD into arrays: metres, video-frame index. | BAR–ATH tracking (input) |
| **features** (`feats.py`) | **99 team-shape features** (per team: centroid, spread, depth quantiles, width, forward velocity, speeds, plus differences between the teams; each also averaged over ±0.5 s and ±2 s), **14 ball-context features** (per-team distance to the ball, oriented ball position and velocity, visibility) and **8 dead-ball cues** (player mean and maximum speed, share of time the ball travels or is still, ball visibility, ball out of the pitch, players near the ball, ball distance to the nearest line). Oriented so that home attacks +x in both halves. | input only |
| **classifiers** (`skillcorner.py`) | Gradient-boosted P(away) on live frames and P(dead) on all frames (scikit-learn `HistGradientBoosting`, fixed seed, no early stopping). | **20 SkillCorner matches only** (591k rows; see §3) |
| **decode** | Viterbi over H / A / D per half. Switching team costs 30 nats; entering or leaving D costs 30; D lasts at least 25 frames (1 s). | settings from §4 |
| **known stoppage** | Graded frames 110,808–113,741 precede the second-half kickoff and are all dead ball (FAQ.md), so they are set to D. | challenge documentation |

## 3. External training data

**SkillCorner open data** (github.com/SkillCorner/opendata, commit 4340d27, MIT licence; credit
SkillCorner and PySport) has 20 matches of 10 Hz broadcast tracking. Players are extrapolated
off-screen, and the ball is present where detected.

- Each match is resampled to 25 Hz and moved to challenge coordinates (metres from a corner).
  Then the **identical feature code** runs on it.
- **Labels:** inside a phase of play, the team in possession. Inside a period with no phase, dead
  ball. This makes the dead-ball labels weak, because phases can stop short of a stoppage.
- Every frame is added a second time with the teams swapped and the label flipped, so the
  classifiers cannot learn which side is "home". One frame is kept every 0.4 s.
- **Held-out check** (trained on 16 matches, tested on 4): the team classifier is right on 93.6%
  of live frames, and the dead-ball classifier reaches AUC 0.985.

## 4. Tuning, on external data only (`tune_skillcorner.py`)

The 20 SkillCorner matches are split into 4 folds. For each fold, both classifiers are fitted on
15 matches. The other 5 are decoded *exactly* as BAR–ATH is, at full 25 Hz resolution, and scored
with the official `spellF1` against SkillCorner's labels. The grid has 60 settings:

- switching cost 10–40;
- dead-ball cost 10–30;
- minimum stoppage 1 or 2 s;
- with or without the boundary refinement described in §7.

The best mean over the 20 held-out matches is **switching 30, dead 30, minimum 1 s, no
refinement: held-out spellF1 0.697, macroF1 0.909**. The top ten settings are within 0.003 of each
other, so the choice is not on a knife edge. `reproduce.sh` reruns this search and the model
reads its result.

## 5. Possession rules implemented

- **Live ball always belongs to a team.** A loose ball stays with the team that had it until the
  evidence for the other team outweighs the switching cost (about 1–2 s of clear evidence). So
  deflections and uncontrolled touches do not change possession.
- **Short possessions are sacrificed.** Spells under about 1 s are mostly merged into their
  neighbours, because `spellF1` charges a split possession twice.
- **Stoppages last at least 1 s.** Dead-ball ownership is never assigned, as the rules ask.

## 6. Results

| | spellF1 | macroF1 |
|---|---|---|
| SkillCorner, 20 held-out matches (used for tuning) | 0.697 | 0.909 |
| **BAR–ATH public 5 min (check only, never used for any choice)** | **0.607** | **0.843** |
| official baseline_smooth50, public 5 min | 0.326 | 0.621 |

On the full match the submission reports 211 possessions, mean 15.7 s, and 48% of graded time
dead. The official scorer puts the public-window noise band near ±0.4 spellF1 (17 possessions).

## 7. Development history and what we dropped

Our first version was **self-supervised**. A team-shape classifier learned from frames where the
ball was clearly at one team's feet. Then came ball proximity, a dead-ball logistic regression
fitted on the public 5 minutes, the same decoder, a self-training round, and a refinement that
moved restarts to the players' slowest moment. It scored 0.542 / 0.867 on the public window.

The dead-ball model and several settings of that version were fitted or chosen on the public
ground truth. The original documentation allowed that; **the 3 November clarification does
not**, so that version is not used for the submission. Its code remains in the repository
(`run()`), with its ablation in the paper.

Tried and dropped, with numbers in [paper/PAPER.md](paper/PAPER.md):

| idea | why it was dropped |
|---|---|
| Smart Tagging team tags | put the right team in possession on only 70% of live frames |
| Last-controller rule | brittle to ball noise |
| Snapping team switches to ball proximity | made timing worse |
| Ball-track cleaning | no gain |
| Broadcast tracking as a second ball source | aligned, but adds a ball on only 8.5% of missing frames |
| Two off-the-shelf video ball detectors (YOLOv8s COCO, FootAndBall) | the ball is 3–5 px; under 50% of detections land within 2 m of it |

## 8. Limitations

- **Domain shift.** SkillCorner tracking comes from broadcast video with extrapolated players;
  the challenge tracking comes from a tactical camera. Held-out accuracy transfers across
  SkillCorner matches, but calibration on BAR–ATH is unverified by design.
- **Weak dead-ball labels.** Dead ball is the complement of the provider's phases.
- **Short possessions.** Spells under about 1 s are mostly missed.

## 9. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cochinae.txt
MTAH_DATA_DIR=/path/to/bundle ./reproduce.sh      # ~50 min on an Apple M1
```

The script:

1. deletes all cached results;
2. parses the data;
3. downloads SkillCorner at the pinned commit (about 1.9 GB, kept if present);
4. builds its features and fits the classifiers;
5. reruns the cross-validated tuning;
6. runs `run_external()`;
7. validates the file and prints its SHA-256.

Everything is deterministic: fixed seeds, no random validation splits, and content-hashed
caches. From-scratch runs produce the same file.

**Code map:** `cochinae/load.py`, `feats.py` (features), `skillcorner.py` and
`skillcorner_download.sh` (external data and classifiers), `tune_skillcorner.py` (tuning),
`model.py` (`run_external` = submission; `run` = development version), `refine.py`,
`evalu.py`. Experiments: `align.py`, `broadcast.py`, `video_ball.py`.
