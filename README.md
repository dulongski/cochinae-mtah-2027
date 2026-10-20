# Cochinae — More than a Hack 2027, Phase 1 (Ball Possession)

Our entry to the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
possession challenge: label every frame of FC Barcelona vs Athletic Club as `H`, `A` or `D` from
automatic tracking.

**How it works, what we tried and dropped, and its limits: [TECHNICAL.md](TECHNICAL.md).**

In short: a self-supervised team-shape classifier (who has the ball, read from the 22 players)
plus ball-proximity evidence plus a physical dead-ball model, decoded as whole possessions with
a Viterbi pass. Then one round of self-training and a restart/stoppage boundary refinement.

| public 5-min window | macroF1 | spellF1 |
|---|---|---|
| **Cochinae** | **0.867** | **0.542** |
| baseline_smooth50 (published floor) | 0.621 | 0.326 |

## Reproduce

Python 3.13 (tested on macOS arm64). The match data is not in this repository. Get the bundle
from the organisers and point `MTAH_DATA_DIR` at it (it must contain `tactical/`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cochinae.txt
MTAH_DATA_DIR=/path/to/bundle ./reproduce.sh     # ~3 min -> evaluation/results/Cochinae.csv
```

The run is deterministic and starts from an empty cache, so it regenerates the submitted file
exactly; the script prints its SHA-256.

## Layout

```
cochinae/            our code (load, feats, model, refine, evalu; align/broadcast = experiment)
reproduce.sh         the one command that regenerates the submission
TECHNICAL.md         the technical document
requirements-cochinae.txt
games.py evaluation/ starter/   the organisers' repository, unchanged (README_CHALLENGE.md)
```

The organisers' code and docs are MIT-licensed (see LICENSE). Our code is under the same licence.
