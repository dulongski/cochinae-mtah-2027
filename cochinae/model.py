"""Cochinae possession model.

    python -m cochinae.model                # writes evaluation/results/Cochinae.csv

Pipeline
  1. shape   -- a gradient-boosted classifier reads possession from the 22 players' geometry.
                It is trained self-supervised: on frames where the ball is visible and one team is
                clearly closer to it, that team is the label. Predictions are out-of-fold (time
                blocks), so no frame is scored by a model that saw it.
  2. touch   -- where the ball is visible, the nearer team gets evidence, scaled by how close.
  3. dead    -- probability the ball is out of play, from player speeds, ball motion and restarts.
  4. decode  -- Viterbi over {H, A, D} with switching costs, so possessions come out as whole
                intervals instead of flickering frames.
  5. self     -- one round of self-training: a classifier with shape + ball features learns from
                the confidently decoded frames, then the match is decoded again.
  6. refine   -- dead-ball boundaries move to the player-speed signature of a restart/stoppage.
"""

import hashlib
import os
import sys

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae import feats as F  # noqa: E402
from cochinae.load import CACHE, load_cache  # noqa: E402
from cochinae.refine import refine_dead  # noqa: E402
from games import half_video_blocks, GAME  # noqa: E402

PUBLIC = (11185, 18685)  # public ground-truth window, [start, stop)

PARAMS = {
    "block": 20000,        # time-block size for out-of-fold shape predictions
    "proxy_r": 1.5,        # proxy label: nearest player within this many metres...
    "proxy_gap": 1.0,      # ...and the other team at least this much further away
    "prox_scale": 1.2,     # logit per metre of distance advantage
    "prox_clip": 3.0,
    "prox_r": 3.0,         # ball further than this from everyone gives no touch evidence
    "shape_w": 1.0,
    "sw_cost": 30.0,       # nats to switch H <-> A
    "dead_cost": 20.0,     # nats to enter or leave D
    "dead_min": 50,        # a stoppage lasts at least 2 s
    "self_iters": 1,
    "shape_lag": 30,       # frames; from lag_scan(), not from the ground truth
    "self_lag": 5,
    "refine_before": 150,  # frames searched before a decoded D boundary
    "refine_after": 50,
    "sw_cost_self": 30.0,
}


def shape_probs(xy, sp, team, sign, y, labelled, half, block, cache_name="shape_oof.npy"):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        return np.load(path)
    S, names = F.team_shape(xy, sp, team, sign)
    X, _ = F.with_context(S, names)
    p = np.full(len(X), np.nan)
    blk = np.arange(len(X)) // block
    for k in np.unique(blk[half > 0]):
        tr = np.flatnonzero(labelled & (blk != k))[::3]
        te = (blk == k) & (half > 0)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    np.save(path, p)
    return p


def dead_features(xy, sp, bf, n):
    """Physical cues of a stoppage, each a rolling mean over a second or a few.

    Players slow down, the ball stops travelling, it sits near a line with few players around it
    (throw-ins, corners, goal kicks) or has just left the pitch. The Smart Tagging was tried here
    and dropped: its set-piece intervals add nothing once these are in.
    """
    R = F.rolling_mean
    vis = bf["vis"]
    allspd = np.nanmean(sp, 1)
    maxspd = np.nanmax(sp, 1)
    moving = (vis & (bf["speed"] > 3) & ~bf["out"]).astype(float)
    still = (vis & (bf["speed"] < 0.5)).astype(float)
    d = np.linalg.norm(xy - bf["b"][:, None, :], axis=2)
    near5 = np.where(vis, (d < 5).sum(1), np.nan)
    bx, by = bf["b"][:, 0], bf["b"][:, 1]
    edge = np.where(vis, np.minimum.reduce([bx, F.LENGTH - bx, by, F.WIDTH - by]), np.nan)
    cols = [R(allspd, 50), R(maxspd, 75), R(moving, 75), R(still, 75),
            R(vis.astype(float), 75), R(bf["out"].astype(float), 100), R(near5, 25),
            R(edge, 25)]
    X = np.stack(cols, 1)
    med = np.nanmedian(X, 0)
    return np.where(np.isfinite(X), X, med)


def viterbi(ll, sw_cost, dead_cost, dead_min):
    """ll: (T, 3) log-likelihoods for H, A, D. Returns state codes 0/1/2.

    D is a chain of `dead_min` sub-states, so a stoppage lasts at least that many frames. Without
    it a one-frame D is a cheap way to change team (two dead_costs instead of one sw_cost), and the
    decoder takes it.
    """
    T, K = len(ll), dead_min
    S = K + 2                                 # 0 H, 1 A, 2.. D1..DK
    NEG = -1e18
    score = np.full(S, NEG)
    score[:2] = ll[0, :2]
    score[2:] = ll[0, 2]                      # a half may open on a dead ball
    back = np.zeros((T, S), np.int32)
    for t in range(1, T):
        new = np.empty(S)
        bk = np.empty(S, np.int32)
        # into H / A: stay, switch from the other team, or leave the last D sub-state
        for k in (0, 1):
            c = (score[k], score[1 - k] - sw_cost, score[S - 1] - dead_cost)
            j = int(np.argmax(c))
            new[k], bk[k] = c[j], (k, 1 - k, S - 1)[j]
        # into D1: from H or A
        j = 0 if score[0] >= score[1] else 1
        new[2], bk[2] = score[j] - dead_cost, j
        # chain D1 -> ... -> DK, DK loops
        new[3:S] = score[2:S - 1]
        bk[3:S] = np.arange(2, S - 1)
        if score[S - 1] > new[S - 1]:
            new[S - 1], bk[S - 1] = score[S - 1], S - 1
        new[:2] += ll[t, :2]
        new[2:] += ll[t, 2]
        score, back[t] = new, bk
    path = np.zeros(T, np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.minimum(path, 2)


def run(params=PARAMS, gt=None, verbose=True, drop_ball=None):
    """drop_ball: optional boolean mask of frames on which to hide the ball (robustness tests)."""
    xy, sp, ball, team, tags = load_cache()
    if drop_ball is not None:
        ball = ball.copy()
        ball[drop_ball] = np.nan
    n = len(xy)
    half = F.halves_mask(n)
    sign = F.attack_sign(xy, team, half)
    bf = F.ball_feats(xy, team, ball)

    # 1. shape
    dmin = np.minimum(bf["dH"], bf["dA"])
    proxy = (bf["dA"] < bf["dH"]).astype(int)          # 1 = away
    labelled = (half > 0) & bf["vis"] & (dmin < params["proxy_r"]) & \
        (np.abs(bf["dH"] - bf["dA"]) > params["proxy_gap"])
    p_away = shape_probs(xy, sp, team, sign, proxy, labelled, half, params["block"])
    p_away = np.clip(np.nan_to_num(p_away, nan=0.5), 0.02, 0.98)
    # Teams reorganise after the ball changes hands, so shape evidence runs late. The lag is
    # measured without ground truth: where shape evidence best correlates with ball proximity.
    p_away = shift(p_away, params["shape_lag"])
    logit_home = -np.log(p_away / (1 - p_away)) * params["shape_w"]

    # 2. touch
    adv = np.clip(bf["dA"] - bf["dH"], -params["prox_clip"], params["prox_clip"])
    near = bf["vis"] & (dmin < params["prox_r"])
    logit_home = logit_home + np.where(near, params["prox_scale"] * np.nan_to_num(adv), 0.0)

    # 3. dead -- fitted on the public window; inside it, out-of-fold so its score stays honest
    D = dead_features(xy, sp, bf, n)
    a, b = PUBLIC
    mu, sd = D[half > 0].mean(0), D[half > 0].std(0) + 1e-9
    Z = (D - mu) / sd
    y = (gt[a:b] == "D").astype(int)

    def fit(rows):
        return LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000).fit(
            Z[a:b][rows], y[rows])
    p_dead = fit(np.ones(b - a, bool)).predict_proba(Z)[:, 1]
    fold = np.arange(b - a) * 4 // (b - a)
    for k in range(4):
        p_dead[a:b][fold == k] = fit(fold != k).predict_proba(Z[a:b][fold == k])[:, 1]
    p_dead = np.clip(p_dead, 0.01, 0.99)

    # 4. decode
    labels = decode(logit_home, p_dead, params, params["sw_cost"])

    # 5. self-training: relearn possession from the decoded labels with ball features added
    for it in range(params["self_iters"]):
        p_away = self_train(xy, ball, team, sign, half, bf, labels, params, it)
        p_away = shift(np.clip(p_away, 0.02, 0.98), params["self_lag"])
        logit_home = -np.log(p_away / (1 - p_away))
        labels = decode(logit_home, p_dead, params, params["sw_cost_self"])

    # 6. boundary refinement: restarts at the players' slowest moment, stoppages where they slow
    labels = refine_dead(labels, sp, params["refine_before"], params["refine_after"], "min")
    return labels, {"p_dead": p_dead, "logit_home": logit_home}


def decode(logit_home, p_dead, params, sw_cost):
    lp_home = -np.logaddexp(0, -logit_home)
    lp_away = -np.logaddexp(0, logit_home)
    ll = np.stack([np.log(1 - p_dead) + lp_home, np.log(1 - p_dead) + lp_away, np.log(p_dead)], 1)
    labels = np.full(len(ll), "X", dtype="<U1")
    for _, s, e in half_video_blocks(GAME):
        path = viterbi(ll[s:e + 1], sw_cost, params["dead_cost"], params["dead_min"])
        labels[s:e + 1] = np.array(["H", "A", "D"])[path]
    return labels


def ball_context(bf, sign):
    """Ball features for the self-trained classifier, oriented like the shape features."""
    R = F.rolling_mean
    dH = np.where(bf["vis"], np.minimum(bf["dH"], 15), np.nan)
    dA = np.where(bf["vis"], np.minimum(bf["dA"], 15), np.nan)
    bx = (bf["b"][:, 0] - F.LENGTH / 2) * sign
    bvx = (np.gradient(R(bf["b"], 2), axis=0) * F.FPS)[:, 0] * sign
    vis = bf["vis"].astype(float)
    return np.stack([dH, dA, dH - dA, R(dH, 12), R(dA, 12), R(dH - dA, 12), R(dH - dA, 25),
                     R(dH - dA, 50), bx, R(bx, 25), bvx, R(bvx, 12), vis, R(vis, 25)],
                    1).astype(np.float32)


def self_train(xy, ball, team, sign, half, bf, labels, params, it):
    """Out-of-fold P(away) from a classifier trained on confidently decoded frames.

    Confident = decoded H or A, more than a second away from any decoded change. Uses the shape
    features plus ball_context. Cached per iteration and per decoding (by a hash of the labels).
    """
    key = hashlib.sha1(labels.tobytes()).hexdigest()[:10]
    path = os.path.join(CACHE, "self_%d_%s.npy" % (it, key))
    if os.path.exists(path):
        return np.load(path)
    n = len(labels)
    px = os.path.join(CACHE, "shapeX.npy")
    if not os.path.exists(px):
        S, names = F.team_shape(xy, sp_cache(), team, sign)
        X, _ = F.with_context(S, names)
        np.save(px, X.astype(np.float32))
    X = np.concatenate([np.load(px), ball_context(bf, sign)], 1)
    y = np.where(labels == "A", 1, np.where(labels == "H", 0, -1))
    near = np.zeros(n, bool)
    for c in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        near[max(0, c - 25):c + 25] = True
    ok = (y >= 0) & ~near & (half > 0)
    blk = np.arange(n) // params["block"]
    p = np.full(n, 0.5)
    for k in np.unique(blk[half > 0]):
        tr = np.flatnonzero(ok & (blk != k))[::3]
        te = (blk == k) & (half > 0)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    np.save(path, p)
    return p


def shift(x, lag):
    """y[t] = x[t + lag], edge-padded."""
    if lag == 0:
        return x
    idx = np.clip(np.arange(len(x)) + lag, 0, len(x) - 1)
    return x[idx]


def sp_cache():
    return load_cache()[1]


if __name__ == "__main__":
    from cochinae.evalu import gt_labels, score
    gt = gt_labels()
    labels, aux = run(gt=gt)
    graded = F.halves_mask(len(labels)) > 0
    s = score(labels, gt)
    ch = np.sum(labels[1:][graded[1:]] != labels[:-1][graded[1:]])
    print("public spellF1 %.4f macroF1 %.4f  pred poss in window %d" % s)
    print("full match: D share %.3f, label changes %d" % ((labels[graded] == "D").mean(), ch))
    out = os.path.join(ROOT, "evaluation", "results", "Cochinae.csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    print("wrote", out)
