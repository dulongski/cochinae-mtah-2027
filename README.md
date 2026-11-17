# Cochinae — More than a Hack 2027 · Ball Possession

Team **Cochinae**: Sebastian Dulong Salazar · Santiago Fernández del Castillo · Ulises Quevedo

We label every frame of FC Barcelona vs Athletic Club as **home possession** (`H`), **away
possession** (`A`) or **dead ball** (`D`), using only the automatic tracking. We learn possession
on 20 other matches (SkillCorner open data) and decode it into whole possessions with a few
football rules. **Nothing is trained, fitted or tuned on this match's ground truth or video.**

| | spellF1 | macroF1 |
|---|---|---|
| SkillCorner, 20 held-out matches (used for tuning) | 0.697 | 0.909 |
| **This match, public 5 min (check only)** | **0.607** | **0.843** |
| Official baseline, public 5 min | 0.326 | 0.621 |

## Documents

- **Technical summary, 2 pages (the submitted document):** [paper/Cochinae_two_pager.pdf](paper/Cochinae_two_pager.pdf)
- Technical document, long form: [TECHNICAL.md](TECHNICAL.md) · [PDF](TECHNICAL.pdf)
- Full paper (method, development history, ablations, negative results): [paper/PAPER.md](paper/PAPER.md) · [PDF](paper/PAPER.pdf)

## Reproduce

Python 3.13, tested on macOS arm64. The match data is not in this repository: get the bundle
from the organisers and point `MTAH_DATA_DIR` at it. It must contain `tactical/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MTAH_DATA_DIR=/path/to/bundle ./reproduce.sh     # ~50 min -> evaluation/results/Cochinae.csv
```

`reproduce.sh`:
1. starts from an empty cache;
2. downloads SkillCorner's open data at a pinned commit;
3. trains and tunes on it;
4. labels the match;
5. validates the file.

It is deterministic and regenerates the submitted file exactly. It prints the SHA-256, which
starts with `80820af0`.

## Layout

```
cochinae/                  our code
  load.py                  parse the challenge tracking into arrays
  feats.py                 team-shape, ball and dead-ball features
  skillcorner.py           external data: load, features, labels, the two classifiers
  skillcorner_download.sh  download SkillCorner open data (pinned commit)
  tune_skillcorner.py      decoder settings by cross-validation on SkillCorner
  model.py                 run_external(): the submitted pipeline (run(): development version)
  refine.py, evalu.py      boundary refinement, in-process scorer
  experiments/             not used for the submission: broadcast alignment, video ball
                           detectors, live progress bars
paper/                     two-pager, paper, figures and the scripts that regenerate them
reproduce.sh               the one command that regenerates the submission
TECHNICAL.md / .pdf        technical document
requirements.txt           pinned dependencies

games.py, evaluation/      from the organisers' repository, unchanged: match constants and the
                           official scorer / validator
```

## Data and credits

- Challenge data © the organisers. It is not redistributed here.
- [SkillCorner open data](https://github.com/SkillCorner/opendata), MIT licence. Credit to
  SkillCorner and PySport.
- `games.py` and `evaluation/` come from the
  [challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027), MIT licence
  (see LICENSE). Our code is under the same licence.
