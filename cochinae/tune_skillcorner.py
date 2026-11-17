"""Tune the decoder on external data only: cross-validation over the 20 SkillCorner matches.

    python -m cochinae.tune_skillcorner predict     # 4 folds: fit on 15 matches, predict the other 5
    python -m cochinae.tune_skillcorner grid        # decode them under each setting, score

No BAR-ATH data is used here, and in particular not its ground truth (organisers' clarification
of 3 Nov 2026: parameters must be tuned on external data). Each held-out match is decoded exactly
as BAR-ATH is (same Viterbi, same refinement) and scored with the official spellF1 against the
SkillCorner phase labels. The setting with the best mean over the 20 matches is written to
$CACHE/sc_tuned.json, which model.py reads.
"""

import glob
import itertools
import json
import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae import model as M, skillcorner as S  # noqa: E402
from cochinae.evalu import score  # noqa: E402
from cochinae.load import CACHE  # noqa: E402
from cochinae.refine import refine_dead  # noqa: E402

warnings.filterwarnings("ignore")
FOLDS = 4
GRID = {"sw_cost": [10.0, 15.0, 20.0, 30.0, 40.0],
        "dead_cost": [10.0, 20.0, 30.0],
        "dead_min": [25, 50],
        "refine": [False, True]}


def mids():
    return sorted(int(os.path.basename(p)) for p in glob.glob(os.path.join(S.SC_DIR, "[0-9]*")))


def predict():
    ids = mids()
    for k in range(FOLDS):
        hold = ids[k::FOLDS]
        team, dead = S.fit(holdout=hold)
        for mid in hold:
            X, lab, half, sp = S.match_full(mid)
            np.savez(os.path.join(CACHE, "sc_cv_%d.npz" % mid),
                     p_away=np.clip(team.predict_proba(X)[:, 1], 0.02, 0.98),
                     p_dead=np.clip(dead.predict_proba(X)[:, 1], 0.01, 0.99),
                     lab=lab, half=half, sp=sp)
            print("match %d done (fold %d)" % (mid, k), flush=True)


def blocks_of(half):
    return [(h, int(np.flatnonzero(half == h)[0]), int(np.flatnonzero(half == h)[-1]))
            for h in (1, 2) if (half == h).any()]


def decode_match(mid, setting):
    z = np.load(os.path.join(CACHE, "sc_cv_%d.npz" % mid))
    sw, dc, dm, ref = setting
    params = dict(M.PARAMS, dead_cost=dc, dead_min=dm)
    blocks = blocks_of(z["half"])
    p = z["p_away"]
    labels = M.decode(-np.log(p / (1 - p)), z["p_dead"], params, sw, blocks)
    if ref:
        labels = refine_dead(labels, z["sp"], params["refine_before"], params["refine_after"],
                             "min", blocks)
    return score(labels, z["lab"])[:2]


def grid():
    ids = mids()
    settings = list(itertools.product(*GRID.values()))
    jobs = [(mid, s) for s in settings for mid in ids]
    res = Parallel(n_jobs=6)(delayed(decode_match)(mid, s) for mid, s in jobs)
    table = {}
    for (mid, s), r in zip(jobs, res):
        table.setdefault(s, []).append(r)
    rows = sorted(((np.mean([r[0] for r in v]), np.mean([r[1] for r in v]), s)
                   for s, v in table.items()), reverse=True)
    for spell, macro, s in rows[:10]:
        print("sw %4.0f dead %3.0f min %2d refine %-5s  spellF1 %.3f  macroF1 %.3f" % (s + (spell, macro)))
    best = rows[0][2]
    out = {"sw_cost": best[0], "dead_cost": best[1], "dead_min": best[2], "refine": best[3],
           "cv_spellF1": rows[0][0], "cv_macroF1": rows[0][1],
           "all": [[list(s), sp, ma] for sp, ma, s in rows]}
    json.dump(out, open(os.path.join(CACHE, "sc_tuned.json"), "w"), indent=1)
    print("best", {k: out[k] for k in ("sw_cost", "dead_cost", "dead_min", "refine",
                                        "cv_spellF1", "cv_macroF1")})


if __name__ == "__main__":
    {"predict": predict, "grid": grid}[sys.argv[1]]()
