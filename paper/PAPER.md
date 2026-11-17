# Who Has the Ball? Frame-Level Possession from Uncorrected Tracking with Team Shape, External Supervision and Duration-Constrained Decoding

**Sebastian Dulong Salazar, Santiago Fernández del Castillo, Ulises Quevedo**  
Team Cochinae · More than a Hack 2027 (Metrica Sports × FC Barcelona × Mobile World Capital)

---

## Abstract

We label every frame of a professional match (FC Barcelona vs Athletic Club, 199,319 frames) as
*home possession*, *away possession* or *dead ball*. The only inputs are automatic, uncorrected
tracking and its metadata. The ball is missing on 41% of graded frames, and the ranking metric
(spellF1) rewards recovering whole possessions rather than per-frame accuracy.

We combine three signals:

1. A **team-shape classifier**, trained self-supervised on frames where the ball is
   unambiguously at one team's feet, which reads possession from the 22 players alone.
2. **Ball-proximity evidence** where the ball is visible.
3. A **physical dead-ball model**, an 8-feature logistic regression on player and ball motion.

A Viterbi decoder with switching costs and a minimum stoppage length turns these into whole
possessions. That was our development version. **The submitted version** computes the same
features on **20 public SkillCorner matches with provider possession phases**. Supervised team
and dead-ball classifiers are trained there, and every decoder setting is chosen by
cross-validation there. Nothing is fitted or tuned on the challenge's ground truth, as the
organisers' 3 November clarification requires.

The submitted pipeline scores **spellF1 0.697 / macroF1 0.909 on 20 held-out SkillCorner
matches**. On the BAR–ATH public 5-minute window, scored only as a check, it reaches **0.607 /
0.843**; the published baseline gets 0.326 / 0.621. No stage uses hand labels, and the output is
reproducible byte for byte. We also report what did not work: tactical tagging as a team prior,
a last-controller rule, ball-track cleaning, broadcast tracking as a ball source, and two
off-the-shelf video ball detectors.

---

## 1. Introduction

Possession is the first segmentation any analyst wants and the first thing automatic systems get
wrong, especially around restarts, contested balls and deflections. The challenge asks for one
label per frame. It scores those labels as *possessions*: a predicted possession counts only if it
overlaps a true one with temporal IoU ≥ 0.3…0.7 (spellF1, averaged over thresholds). Two
consequences shape everything below:

- **Flicker is fatal.** A per-frame classifier that is 90% accurate but switches labels every few
  frames invents hundreds of possessions and scores near zero.
- **Timing matters at the scale of a second.** A possession recovered perfectly but one second
  late scores about 0.65.

So we treat the problem as **segmentation with duration constraints**, not classification.
Because the ball, the most informative signal, is missing exactly in the hardest moments, we
need evidence that survives its absence.

## 2. Data and task

| item | details |
|---|---|
| Video | tactical camera, 199,319 frames at 25 fps; graded frames 11,185–90,651 and 110,808–190,197 (158,857) |
| Tracking (ATD) | 46 anonymous track slots with team membership; position and speed per slot; ball position. Automatic, with gaps and identity switches. |
| Ball | visible on 59.3% of graded frames |
| Other | Smart Tagging (638 automatic phase intervals), per-frame homography, broadcast feed with its own ATD (not synchronised) |
| Ground truth | public: first 5 minutes (7,500 frames, 17 possessions); private: full match |

Labels in the ground truth come mechanically from *corrected* tracking: `D` where the ball is
marked dead, otherwise the team the tracking names as owner.

**An observation that changed the design.** The ball is missing in long blocks, not scattered
frames: the median gap is 10.7 s and the 75th percentile 21 s. 93% of the missing frames end up
decoded as dead ball. Missing ball is mostly a symptom of the ball being *out of play*. In live
play the ball is usually visible, so the hard part is boundaries, not blind stretches.

## 3. Method

All geometry is in metres, oriented so that the home team attacks towards +x in both halves. We
detect each team's defending side from its mean x-position over the half.

### 3.1 Team shape (self-supervised)

For every frame we compute 33 descriptors of the two teams. Per team: centroid, spread, depth
quantiles, width, forward velocity and speed. Then differences between the teams: who is higher,
more stretched, faster forward. Each descriptor is also averaged over ±0.5 s and ±2 s, giving 99
features in total.

**Labels without labels.** A frame gets a proxy label when the ball is visible, one player is
within 1.5 m of it and the nearest opponent is at least 1 m further away (35k frames). A
gradient-boosted classifier (scikit-learn `HistGradientBoosting`) learns P(away | shape) from
these frames. Predictions are **out-of-fold over 20,000-frame time blocks**, so no frame is
scored by a model that saw it.

Shape alone puts 87.5% of live public frames on the correct team. Nearest-player-to-ball gets
77.8%.

**Lag.** Teams reorganise *after* the ball changes hands, so shape evidence runs late. We estimate
the delay without ground truth, by cross-correlating shape evidence with ball-proximity evidence
over the whole match. The correlation peaks with a 30–40 frame shift. We shift shape evidence
30 frames earlier.

### 3.2 Ball proximity

Where the ball is visible and someone is within 3 m of it, the home logit gains
1.2 × (d_away − d_home), with the distance difference in metres and clipped to ±3 m.

### 3.3 Dead ball

The dead-ball model is a logistic regression (L2, C = 0.05, class-balanced) on 8 physical cues:

- mean player speed (±2 s);
- maximum player speed (±3 s);
- share of frames in which the ball travels faster than 3 m/s (±3 s);
- share of frames in which the ball is still (±3 s);
- ball visibility (±3 s);
- ball outside the pitch (±4 s);
- number of players within 5 m of the ball (±1 s);
- distance from the ball to the nearest line (±1 s).

This is the **only** component fit on the public ground truth. Inside the public window its
predictions are 4-fold out-of-fold, so the public score is not in-sample. Feature selection used
the same folds: these 8 physical cues reach out-of-fold AUC 0.94, against 0.81 for a first set
built on Smart Tagging and smoothed speeds.

### 3.4 Duration-constrained decoding

We decode each half with Viterbi over the states H, A, D:

- **Emissions:** log P(alive) + log P(team | evidence) for H and A, and log P(dead) for D.
- **Transitions:** switching team costs 30 nats; entering or leaving D costs 20 nats.
- **Minimum stoppage length:** D is expanded into a chain of 50 sub-states, so a stoppage lasts at
  least 2 s. This constraint is necessary, not cosmetic. Without it, a one-frame D costs two
  dead-ball transitions, which is cheaper than one team switch, and the decoder learns to change
  team through a fake stoppage.

### 3.5 Self-training

A second classifier sees the shape features plus 14 ball features: per-team distances and their
difference, at several smoothings, plus the oriented ball position and velocity and the ball's
visibility. Its labels are the decoded H/A frames, excluding ±1 s around every decoded change.
Predictions are out-of-fold as before. Its evidence lags by only about 5 frames (same
cross-correlation test), and the match is decoded again with it.

### 3.6 Boundary refinement

Averaged over the public restarts, mean player speed bottoms out exactly at the restart frame
(about 0.9 m/s) and then climbs (Fig. 2, left). The dead-ball probability fades over 2–3 s
afterwards, so the decoder places restarts late. We therefore move each decoded restart to the
players' slowest moment within the 6 s before it (and up to 2 s after). Each decoded stoppage
moves to the best fast-to-slow changepoint of player speed. This step cut the median restart
error from 75 to 34 frames.

![Signal profiles around true boundaries](figs/profiles.png)
*Figure 2. Medians around true boundaries in the public window. Left, centre: mean player speed
(solid) and P(dead) (dashed) around restarts and stoppages. Right: distance from each team to the
ball, and the shape model's P(new team), around team switches. Shape evidence crosses 0.5 about
1 s after the switch, which is the lag we correct.*

### 3.7 External supervision (SkillCorner open data) — the submitted pipeline

The self-supervised stages never see a real possession label from this match. SkillCorner's
open data (20 matches, MIT licence) provides broadcast tracking at 10 Hz plus *phases of play*:
intervals with the team in possession. We resample each match to 25 Hz, move it to the same
coordinates as the challenge tracking, and run **the exact feature code used on BAR–ATH**:
99 shape, 14 ball-context and 8 dead-ball features. Frames inside a period that no phase covers
are labelled dead ball, so the dead-ball labels are weak. Every frame is added a second time with
the teams swapped and the label flipped, so the classifiers cannot learn which side is "home".
That gives 591k training rows, one every 0.4 s.

Two gradient-boosted classifiers are trained on all 20 matches: P(away) on live frames and
P(dead) on all frames. On 4 held-out matches (trained on the other 16) the team classifier is
right on **93.6%** of live frames and the dead-ball classifier reaches **AUC 0.985**.

During development we first blended this evidence with ours. The table shows each blend weight
on the public window. **It is a development record, not how settings were chosen**: under the
3 November clarification the public ground truth may not be used for tuning.

| SkillCorner weight (team / dead) | spellF1 | macroF1 |
|---|---|---|
| 0 / 0 (self-supervised only) | 0.542 | 0.867 |
| 0.5 / 0.5 | 0.587 | 0.891 |
| 1 / 0 | 0.621 | 0.887 |
| 1 / 0.5 | 0.643 | 0.908 |
| 1 / 1 | 0.621 | 0.894 |

**Submitted configuration.** The submission uses the SkillCorner classifiers alone, for both
team and dead ball. The dead-ball logistic regression fitted on the public window is dropped. We
chose the decoder settings by 4-fold cross-validation over the 20 SkillCorner matches. For each
fold, both classifiers are fitted on 15 matches, and the other 5 are decoded at 25 Hz exactly
as BAR–ATH is and scored with the official spellF1 against SkillCorner's labels. The grid has 60
settings: switching cost 10–40, dead-ball cost 10–30, minimum stoppage 1–2 s, and with or
without §3.6's refinement. The best is switching 30, dead 30, minimum 1 s, no refinement, with
**held-out spellF1 0.697 / macroF1 0.909**. The top ten settings lie within 0.003.

### 3.8 A known stoppage

The FAQ states that graded frames 110,808–113,741 precede the second-half kickoff and are all
dead ball. We set them to D after decoding. 1,746 of those frames had been decoded as live. The
first half is graded from its kickoff, so the public window is unaffected.

## 4. Results

### 4.1 Ablation (public window)

| configuration | spellF1 | macroF1 | possessions in window (truth 17) | full match: possessions / mean duration |
|---|---|---|---|---|
| published baseline (nearest player + ±2 s vote) | 0.326 | 0.621 | 26 | — |
| ball proximity only | 0.389 | 0.798 | 18 | 333 / 9.6 s |
| team shape only | 0.364 | 0.788 | 16 | 253 / 12.8 s |
| shape + proximity + dead + Viterbi | 0.508 | 0.823 | 20 | 293 / 11.1 s |
| + lag correction | 0.467 | 0.814 | 19 | 302 / 10.8 s |
| + self-training | 0.503 | 0.833 | 14 | 272 / 12.1 s |
| + boundary refinement | 0.542 | 0.867 | 14 | 269 / 13.1 s |
| + SkillCorner classifiers, blended (development) | 0.621 | 0.905 | 12 | 234 / 14.6 s |
| **submitted: SkillCorner only, settings tuned on SkillCorner** | **0.607** | **0.843** | 12 | 211 / 15.7 s |

![Ablation](figs/ablation.png)
*Figure 1. Ablation on the public window.*

The official scorer's bootstrap puts the public noise band near ±0.4 spellF1, so individual
steps in this table are not statistically separable. Two numbers are informative despite that:

- The jump from either signal alone (0.36–0.39) to their combination under the decoder (0.51).
- The macroF1 gain from refinement (0.833 → 0.867), which comes from restart timing.
- The external pipeline reaches 0.607 spellF1 with no BAR–ATH label used anywhere, against 0.542
  for the development version that did use the public window. Its lower macroF1 (0.843) comes
  from dead-ball timing: SkillCorner's dead-ball labels are weak (§6).

The lag correction lowers the public score at that intermediate step: 0.508 to 0.467, which is
one possession out of 17. In the final pipeline it does not. With every other stage in place,
shape lag 0 gives 0.537, lag 15 gives 0.542, lag 30 (ours) 0.542 and lag 45 0.463. The value 30
comes from the label-free cross-correlation, not from this table, and it sits in the stable
region of §4.2.

### 4.2 Robustness without labels

This analysis was run on the self-supervised model (before §3.7). We could not use 17
possessions to choose settings. Instead we perturbed one parameter at a time
and scored each perturbed run against the reference run on the **full match** with the official
metric:

| perturbation | agreement (spellF1) |
|---|---|
| switching costs 20–40 (both decodes) | 0.89–0.99 |
| dead-ball entry cost 10–30; minimum stoppage 1–4 s | 0.92–0.94 |
| dead-ball regularisation C 0.005–1.0 (200× range) | 0.88–0.92 |
| restart window 4–8 s; self-training lag 0–10 frames | 0.96–0.99 |
| shape lag 0–45 frames; proximity weight 0.6–2.0 | 0.83–0.90 |

For the two most sensitive parameters we ran a 3×3 grid and chose the **medoid**: the run that
agrees best with the other eight. It is the configuration we use (weight 1.2, lag 30), with a
mean agreement of 0.875. On the full match the model reports 269 possessions with a mean of
13.1 s and 45% of graded time dead. The public window's own mean possession is 14.2 s.

## 5. What did not work

| idea | result | why we dropped it |
|---|---|---|
| Smart Tagging team tags as possession prior | 70% of live public frames on the right team (shape alone: 87.5%) | its intervals are phases, not possessions |
| Smart Tagging set pieces in the dead-ball model | no gain over the physical cues | redundant |
| Ball-track cleaning (drop jumps over 40 m/s, bridge gaps up to 0.4 s) | public spellF1 0.542 → 0.512 | no measurable gain; off |
| "Last controller keeps it" rule (radius r, k frames of control) | best public spellF1 0.59 vs 0.60 for the decoder, with the same dead-ball mask | brittle to ball noise, no evidence when the ball is missing |
| Snap team switches to the ball-proximity changepoint | median switch error 16 → 26 frames | in uncorrected tracking the true switch is not where the new team first reaches the ball |
| Broadcast tracking as a second ball source | aligned to the tactical clock (offset ≈ 4,015 / 26,921 frames per half; 95% of coarse matches within 1 m), but it adds a ball on only 8.5% of missing frames | both cameras lose the ball at the same moments |
| YOLOv8s (COCO "sports ball") on the tactical video | at confidence ≥ 0.6, detects on 22% of frames and only 50% of those are within 2 m of the tracked ball | the ball is 3–5 px in a full-pitch shot; heads, boots and the penalty spot look alike |
| FootAndBall (Komorowski et al., 2020) | detects on 29% of frames and under 5% of those are within 2 m | trained on half-pitch cameras, where the ball is about twice the size |

<img src="figs/detectors.png" alt="Detectors" width="420">
*Figure 3. Off-the-shelf ball detectors on 2,000 frames where the tracking has the ball:
coverage against agreement as the confidence threshold rises. Neither is usable as-is.*

The video path is also capped in value. Only about 8.5k graded frames (about 5%) are both live
and missing the ball, so even a perfect detector would move spellF1 by an amount comparable to
the evaluation noise.

## 6. Limitations and future work

- **Short possessions.** Spells under about 2 s (a touch immediately lost) are mostly merged into
  their neighbours: 9% of our spells are under 2 s, against 18% in the public truth. This is a
  deliberate trade, because spellF1 charges a split possession twice, but it is where most of
  the remaining recall is.
- **Dead-ball labels.** Our own dead-ball model is fit on four stoppages, and SkillCorner's
  dead-ball labels are the complement of its phases, which can stop short of a stoppage.
  Averaging the two limits the damage of either, but this remains the weakest label we use.
- **Domain shift.** SkillCorner tracking comes from broadcast video with players extrapolated
  off-screen; the challenge tracking comes from a tactical camera. Both are resampled to the
  same features, and the held-out accuracy transfers, but the calibration need not.
- **Detector on this camera.** The natural next step is to fine-tune a small detector on *this*
  camera, with labels generated automatically by projecting the 113k tracked ball positions into
  the video through the homography. No hand labelling is involved, and it targets exactly the
  3–5 px domain where off-the-shelf models fail.
- **Zone-dependent switching.** Possession changes are more frequent in midfield duels than in a
  team's own third, so the switching cost could depend on where the ball is.

## 7. Compliance with the rules clarification

The organisers' clarification of 3 November 2026 forbids training, fitting, calibrating or
tuning on the ground truth or the video. In the development version (§3.1–3.6), the dead-ball
model was fitted on the public 5 minutes, and some settings were read off that window, which
the original documentation allowed. **The submitted file comes from `run_external()` only**:
- classifiers trained on SkillCorner;
- decoder settings from cross-validation on SkillCorner;
- the documented pre-kickoff stoppage.

The public window is scored after the fact, as an automatic check, and drives no choice.
Every table above marked "public window" is a record of development, not a selection
criterion for the submission.

## 8. Reproducibility and data handling

`./reproduce.sh` clears every cached result, parses the data, downloads SkillCorner's open data
at a pinned commit (about 1.9 GB, kept if present), builds and fits its classifiers, reruns the
cross-validated tuning, runs `run_external()` and validates the file. From-scratch runs produce
the same CSV byte for byte (about 50 minutes on an Apple M1).
The classifiers use fixed seeds and no random validation splits, and caches are keyed by content
hashes. `paper/collect.py` and `paper/make_figs.py` regenerate every number and figure in this
document.

All processing runs locally. The repository contains no match data and no frame-level
derivatives: the figures are medians over events and calibration shares.

**Code:** https://github.com/dulongski/cochinae-mtah-2027

## References

- Metrica Sports, *More than a Hack 2027 — Phase 1: Ball Possession*, challenge repository and
  evaluation code, 2026.
- J. Komorowski, G. Kurzejamski, G. Sarwas. *FootAndBall: Integrated player and ball detector.*
  VISAPP 2020.
- G. Jocher et al. *Ultralytics YOLOv8*, 2023.
- SkillCorner and PySport, *SkillCorner open data*, github.com/SkillCorner/opendata
  (commit 4340d27), MIT licence.
- G. D. Forney. *The Viterbi algorithm.* Proceedings of the IEEE, 1973.
- F. Pedregosa et al. *Scikit-learn: Machine learning in Python.* JMLR, 2011.
