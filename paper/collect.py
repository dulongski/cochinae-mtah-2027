"""Collect the paper's numbers: ablation ladder and aggregate figure data.

    MTAH_DATA_DIR=/path/to/bundle python paper/collect.py

Writes cache/ablation.json and cache/fig_data.json. Everything written is aggregate (scores,
medians over events, calibration shares); no frame-level data leaves the cache.
"""

import glob
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae import feats as F, model as M  # noqa: E402
from cochinae.evalu import gt_labels, score  # noqa: E402
from cochinae.load import CACHE, load_cache  # noqa: E402

warnings.filterwarnings("ignore")
A, B = 11185, 18685          # public ground-truth window


def full_stats(lab):
    spells = []
    for _, s, e in F.half_video_blocks(F.GAME):
        v = lab[s:e + 1]
        cut = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(v)]
        spells += [(b - a) / 25 for a, b in zip(cut[:-1], cut[1:]) if v[a] != "D"]
    graded = F.halves_mask(len(lab)) > 0
    return len(spells), float(np.mean(spells)), float((lab[graded] == "D").mean())


def ablation(gt):
    off = dict(self_iters=0, shape_lag=0, self_lag=0, refine_before=0, refine_after=0)
    ladder = [("ball proximity only", dict(off, shape_w=0.0)),
              ("team shape only", dict(off, prox_scale=0.0)),
              ("shape + proximity + dead + Viterbi", off),
              ("+ lag correction", dict(off, shape_lag=30)),
              ("+ self-training", dict(off, shape_lag=30, self_iters=1, self_lag=5)),
              ("+ boundary refinement (final)", {})]
    rows = []
    for name, over in ladder:
        lab, _ = M.run(dict(M.PARAMS, **over), gt=gt)
        rows.append((name,) + score(lab, gt) + full_stats(lab))
        print("%-38s spell %.3f macro %.3f" % (name, rows[-1][1], rows[-1][2]), flush=True)
    return rows


def profiles(gt):
    xy, sp, ball, team, _ = load_cache()
    bf = F.ball_feats(xy, team, ball)
    _, aux = M.run(gt=gt)
    spd = F.rolling_mean(np.nanmean(sp, 1), 6)
    g = gt[A:B]
    offs = np.arange(-150, 151, 5)
    out = {"offs": offs.tolist()}
    for kind, cond in (("restart", lambda a, b: a == "D" and b != "D"),
                       ("stoppage", lambda a, b: a != "D" and b == "D")):
        ks = [k for k in np.flatnonzero(g[1:] != g[:-1]) + 1 if cond(g[k - 1], g[k])]
        out[kind] = {"n": len(ks),
                     "speed": np.nanmedian([[spd[A + k + o] for o in offs] for k in ks], 0).tolist(),
                     "p_dead": np.nanmedian([[aux["p_dead"][A + k + o] for o in offs]
                                             for k in ks], 0).tolist()}
    shape = np.load(sorted(glob.glob(os.path.join(CACHE, "shape_*.npy")))[0])
    ks = [k for k in np.flatnonzero(g[1:] != g[:-1]) + 1 if g[k - 1] != "D" and g[k] != "D"]

    def med(fn):
        return np.nanmedian([[fn(k, A + k + o) for o in offs] for k in ks], 0).tolist()
    out["switch"] = {
        "n": len(ks),
        "p_new": med(lambda k, f: 1 - shape[f] if g[k] == "H" else shape[f]),
        "d_new": med(lambda k, f: min((bf["dH"] if g[k] == "H" else bf["dA"])[f], 15)),
        "d_old": med(lambda k, f: min((bf["dA"] if g[k] == "H" else bf["dH"])[f], 15)),
    }
    out["calib"] = {}
    for name in ("yolo", "fab"):
        path = os.path.join(CACHE, "video_ball_%s_calib.npz" % name)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        f, c = z["frame"], z["cand"][:, 0, :]
        d = np.linalg.norm(c[:, :2] - ball[f, :2], axis=1)
        ths = [0.05, 0.15, 0.25, 0.4, 0.6, 0.8]
        out["calib"][name] = {
            "th": ths, "share": [float((c[:, 2] >= t).mean()) for t in ths],
            "within2": [float((d[c[:, 2] >= t] < 2).mean()) if (c[:, 2] >= t).any() else None
                        for t in ths]}
    return out


if __name__ == "__main__":
    gt = gt_labels()
    json.dump(ablation(gt), open(os.path.join(CACHE, "ablation.json"), "w"))
    json.dump(profiles(gt), open(os.path.join(CACHE, "fig_data.json"), "w"))
    print("wrote", CACHE)
