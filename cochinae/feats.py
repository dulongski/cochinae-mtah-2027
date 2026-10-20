"""Per-frame features from the cached tracking.

Two families:
  * ball features   -- where the ball is, how fast it moves, how close each team is to it.
  * shape features  -- the geometry of the 22 players alone, usable when the ball is missing.

Everything is expressed in a "home attacks +x" frame so the same feature means the same thing in
both halves: per half, the side each team defends is read off the team centroids.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

LENGTH, WIDTH = GAMES[GAME]["pitch"]
FPS = GAMES[GAME]["fps"]


def halves_mask(n):
    """(half_id array: 0 outside, 1/2 inside the graded halves)."""
    h = np.zeros(n, np.int8)
    for half, a, b in half_video_blocks(GAME):
        h[a:b + 1] = half
    return h


def rolling_mean(x, w):
    """Centred NaN-aware moving average over +/- w frames, along axis 0."""
    if w <= 0:
        return x
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    pad = [(1, 0)] + [(0, 0)] * (x.ndim - 1)
    cs = np.pad(np.cumsum(v, axis=0), pad)
    cn = np.pad(np.cumsum(ok, axis=0), pad)
    n = len(x)
    i = np.arange(n)
    lo, hi = np.maximum(0, i - w), np.minimum(n, i + w + 1)
    num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def attack_sign(xy, team, half):
    """+1 if home attacks towards +x in that frame's half, else -1 (0 outside halves)."""
    sign = np.zeros(len(xy), np.int8)
    for h in (1, 2):
        m = half == h
        if not m.any():
            continue
        hx = np.nanmean(xy[m][:, team == 0, 0])
        ax = np.nanmean(xy[m][:, team == 1, 0])
        # A team's centroid sits on the half it defends, on average over a half.
        sign[m] = 1 if hx < ax else -1
    return sign


def team_shape(xy, sp, team, sign):
    """Shape features, home-attacks-+x orientation. Returns (F, K) float32 and names."""
    x = (xy[..., 0] - LENGTH / 2) * sign[:, None]          # centred, oriented
    y = xy[..., 1] - WIDTH / 2
    vx = np.gradient(rolling_mean(x, 3), axis=0) * FPS      # oriented x velocity per slot
    cols, names = [], []
    for t, tag in ((0, "h"), (1, "a")):
        m = team == t
        tx, ty, tv, ts = x[:, m], y[:, m], vx[:, m], sp[:, m]
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(tx).sum(1)
            feats = {
                "n": cnt,
                "cx": np.nanmean(tx, 1), "cy": np.nanmean(ty, 1),
                "sx": np.nanstd(tx, 1), "sy": np.nanstd(ty, 1),
                "maxx": np.nanmax(tx, 1), "minx": np.nanmin(tx, 1),
                "q80x": np.nanpercentile(tx, 80, axis=1), "q20x": np.nanpercentile(tx, 20, axis=1),
                "vx": np.nanmean(tv, 1), "spd": np.nanmean(ts, 1), "spdmax": np.nanmax(ts, 1),
                "wid": np.nanmax(ty, 1) - np.nanmin(ty, 1),
            }
        for k, v in feats.items():
            cols.append(v)
            names.append(tag + "_" + k)
    F = np.stack(cols, 1).astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}
    # Relational features: who is pushed up, who is stretched, who runs forward.
    rel = {
        "d_cx": F[:, idx["h_cx"]] + F[:, idx["a_cx"]],      # both oriented: + means home is higher
        "d_sx": F[:, idx["h_sx"]] - F[:, idx["a_sx"]],
        "d_sy": F[:, idx["h_sy"]] - F[:, idx["a_sy"]],
        "d_wid": F[:, idx["h_wid"]] - F[:, idx["a_wid"]],
        "d_vx": F[:, idx["h_vx"]] - F[:, idx["a_vx"]],
        "d_spd": F[:, idx["h_spd"]] - F[:, idx["a_spd"]],
        "all_spd": np.nanmean(sp, 1),
    }
    F = np.concatenate([F, np.stack(list(rel.values()), 1).astype(np.float32)], 1)
    names += list(rel)
    return F, names


def with_context(F, names, windows=(12, 50)):
    """Append rolling means of every feature over a few windows (seconds of context)."""
    out, nm = [F], list(names)
    for w in windows:
        out.append(rolling_mean(F, w).astype(np.float32))
        nm += ["{}_m{}".format(n, w) for n in names]
    return np.concatenate(out, 1), nm


def ball_feats(xy, team, ball):
    """Ball position/speed, nearest distance per team (inf where unknown)."""
    b = ball[:, :2].astype(np.float64)
    vis = np.isfinite(b[:, 0])
    d = np.linalg.norm(xy - b[:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        dH = np.nanmin(np.where(team == 0, d, np.inf), 1)
        dA = np.nanmin(np.where(team == 1, d, np.inf), 1)
    dH[~vis], dA[~vis] = np.inf, np.inf
    vel = np.gradient(rolling_mean(b, 2), axis=0) * FPS
    speed = np.linalg.norm(vel, axis=1)
    out = (b[:, 0] < -0.5) | (b[:, 0] > LENGTH + 0.5) | (b[:, 1] < -0.5) | (b[:, 1] > WIDTH + 0.5)
    return {"vis": vis, "dH": dH, "dA": dA, "speed": speed, "out": out & vis, "b": b}
