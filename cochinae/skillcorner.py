"""External supervision from SkillCorner open data (MIT; credit SkillCorner and PySport).

    ./cochinae/skillcorner_download.sh             # 20 matches, pinned commit
    python -m cochinae.skillcorner build           # features + labels -> $CACHE/sc_train.npz
    python -m cochinae.skillcorner fit             # two classifiers   -> $CACHE/sc_models.pkl

Each SkillCorner match has 10 Hz broadcast tracking (players extrapolated off-screen, ball only
where detected) and "phases of play": intervals with the team in possession. Inside a period,
frames not covered by any phase are taken as dead ball, so the labels are weak (provider phases
can stop short of a stoppage).

Matches are resampled to 25 Hz and moved to the same coordinates as the challenge tracking
(metres from a corner), so the exact feature code used on BAR-ATH runs on them unchanged:
team shape (feats.team_shape + context), ball context and the dead-ball cues. Every frame is
also added with the teams swapped and the label flipped, so the classifiers cannot learn which
side is "home".

Two HistGradientBoosting classifiers are fitted on all 20 matches:
  team -- P(away | shape, ball) on live frames
  dead -- P(dead | shape, ball, dead cues) on all frames
"""

import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae import feats as F  # noqa: E402
from cochinae.load import CACHE  # noqa: E402
from games import DATA_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SC_DIR = os.environ.get("MTAH_SKILLCORNER") or os.path.join(DATA_DIR, "skillcorner")
SC_FPS = 10
UP = 2.5            # 10 Hz -> 25 Hz
STRIDE = 10         # keep every 10th 25 Hz frame for training (0.4 s)


def load_match(mid):
    folder = os.path.join(SC_DIR, str(mid))
    meta = json.load(open(os.path.join(folder, "%s_match.json" % mid)))
    L, W = meta["pitch_length"], meta["pitch_width"]
    home = meta["home_team"]["id"]
    slots = {}
    for p in meta["players"]:
        slots[p["id"]] = None
    ids = sorted(slots, key=lambda i: (next(p["team_id"] for p in meta["players"] if p["id"] == i) != home, i))
    slot = {pid: k for k, pid in enumerate(ids)}
    team = np.array([0 if next(p["team_id"] for p in meta["players"] if p["id"] == i) == home else 1
                     for i in ids], np.int8)

    periods = [(p["start_frame"], p["end_frame"]) for p in meta["match_periods"]]
    f0, f1 = periods[0][0], periods[-1][1]
    n = f1 - f0 + 1
    xy = np.full((n, len(ids), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    with open(os.path.join(folder, "%s_tracking_extrapolated.jsonl" % mid)) as fh:
        for line in fh:
            r = json.loads(line)
            f = r["frame"] - f0
            if f < 0 or f >= n:
                continue
            for pd_ in r["player_data"]:
                k = slot.get(pd_["player_id"])
                if k is not None and pd_["x"] is not None:
                    xy[f, k] = (pd_["x"] + L / 2, pd_["y"] + W / 2)
            b = r["ball_data"]
            if b["x"] is not None and b.get("is_detected"):
                ball[f] = (b["x"] + L / 2, b["y"] + W / 2)

    lab = np.full(n, "X", dtype="<U1")
    period_of = np.zeros(n, np.int8)
    for k, (a, b) in enumerate(periods, 1):
        lab[a - f0:b - f0 + 1] = "D"
        period_of[a - f0:b - f0 + 1] = k
    ph = pd.read_csv(os.path.join(folder, "%s_phases_of_play.csv" % mid))
    for r in ph.itertuples(index=False):
        a, b = periods[int(r.period) - 1]
        s, e = max(int(r.frame_start), a), min(int(r.frame_end), b)
        if e > s:
            lab[s - f0:e - f0] = "H" if r.team_in_possession_id == home else "A"
    return xy, ball, team, lab, period_of


def upsample(x):
    """Linear 10 -> 25 Hz along axis 0; NaN where either neighbour is NaN."""
    n = len(x)
    t = np.arange(int((n - 1) * UP) + 1) / UP
    i = np.floor(t).astype(int)
    j = np.minimum(i + 1, n - 1)
    w = (t - i).reshape((-1,) + (1,) * (x.ndim - 1))
    return (x[i] * (1 - w) + x[j] * w).astype(np.float32)


def nearest(x):
    n = len(x)
    t = np.arange(int((n - 1) * UP) + 1) / UP
    return x[np.round(t).astype(int).clip(0, n - 1)]


def match_full(mid):
    """Every 25 Hz frame of one match, teams as given: (X, labels, half, player speed)."""
    from cochinae import model as M
    xy10, ball10, team, lab10, per10 = load_match(mid)
    xy, ball = upsample(xy10), upsample(ball10)
    lab, half = nearest(lab10), nearest(per10)
    sp = F.rolling_mean(np.linalg.norm(np.gradient(xy, axis=0), axis=2) * F.FPS, 6).astype(np.float32)
    sign = F.attack_sign(xy, team, half)
    S, names = F.team_shape(xy, sp, team, sign)
    X, _ = F.with_context(S, names)
    bf = F.ball_feats(xy, team, np.c_[ball, np.full(len(ball), np.nan)])
    X = np.concatenate([X, M.ball_context(bf, sign), M.dead_features(xy, sp, bf, len(xy))], 1)
    return X.astype(np.float32), np.where(half > 0, lab, "X"), half, sp


def match_features(mid):
    from cochinae import model as M
    xy10, ball10, team, lab10, per10 = load_match(mid)
    xy, ball = upsample(xy10), upsample(ball10)
    lab, half = nearest(lab10), nearest(per10)
    sp = F.rolling_mean(np.linalg.norm(np.gradient(xy, axis=0), axis=2) * F.FPS, 6).astype(np.float32)
    out = []
    for flip in (False, True):
        tm = (1 - team) if flip else team
        sign = F.attack_sign(xy, tm, half)
        S, names = F.team_shape(xy, sp, tm, sign)
        X, _ = F.with_context(S, names)
        bf = F.ball_feats(xy, tm, np.c_[ball, np.full(len(ball), np.nan)])
        XB = np.concatenate([X, M.ball_context(bf, sign), M.dead_features(xy, sp, bf, len(xy))], 1)
        y = lab.copy()
        if flip:
            y = np.where(lab == "H", "A", np.where(lab == "A", "H", lab))
        keep = np.flatnonzero((half > 0) & (lab != "X"))[::STRIDE]
        out.append((XB[keep].astype(np.float32), y[keep]))
    return out


def build():
    mids = sorted(int(os.path.basename(p)) for p in glob.glob(os.path.join(SC_DIR, "[0-9]*")))
    Xs, ys, gs = [], [], []
    for mid in mids:
        for X, y in match_features(mid):
            Xs.append(X)
            ys.append(y)
            gs.append(np.full(len(y), mid))
        print("match", mid, "rows", sum(len(y) for y in ys), flush=True)
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    np.savez(os.path.join(CACHE, "sc_train.npz"), X=X, y=y, g=g)
    print("built", X.shape, {c: int((y == c).sum()) for c in "HAD"})


def classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def fit(holdout=()):
    z = np.load(os.path.join(CACHE, "sc_train.npz"))
    X, y, g = z["X"], z["y"], z["g"]
    tr = ~np.isin(g, list(holdout))
    live = tr & (y != "D")
    team = classifier().fit(X[live], (y[live] == "A").astype(int))
    dead = classifier().fit(X[tr], (y[tr] == "D").astype(int))
    return team, dead


def fit_all():
    team, dead = fit()
    with open(os.path.join(CACHE, "sc_models.pkl"), "wb") as fh:
        pickle.dump({"team": team, "dead": dead}, fh)
    print("saved", os.path.join(CACHE, "sc_models.pkl"))


if __name__ == "__main__":
    {"build": build, "fit": fit_all}[sys.argv[1]]()
