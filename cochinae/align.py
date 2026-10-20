"""Map the broadcast tracking onto the tactical timeline, and borrow its ball.

The broadcast feed is not synchronised, but it is near-live: after a coarse search the offset
(tactical frame - broadcast frame) sits at one value per half and drifts by a few frames. So:

  1. coarse  -- every 10 s of broadcast, find the tactical frame whose players best match its
                players (median distance from each broadcast player to the nearest tactical
                player of the same team). Keep the good matches.
  2. fine    -- per broadcast frame, search +/- SEARCH frames around the interpolated coarse
                offset, at frame resolution, then median-filter the offsets.
  3. borrow  -- wherever the matched broadcast frame has a ball and a valid homography, that
                ball is a candidate position for the tactical frame.

    python -m cochinae.align           # writes $MTAH_CACHE/ball_broadcast.npy
"""

import os
import sys
import warnings

import numpy as np
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae import feats as F  # noqa: E402
from cochinae.load import CACHE, load_cache  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

COARSE_STEP = 250
SEARCH = 40
MIN_PLAYERS = 6
GOOD = 1.0       # metres, median player match distance for a usable frame


def match_cost(P_by_team, T_by_team):
    """Median over broadcast players of the distance to the nearest same-team tactical player.

    P_by_team: [(p, 2)] per team. T_by_team: [(C, n, 2)] per team. Returns (C,)."""
    ds = []
    for P, T in zip(P_by_team, T_by_team):
        if len(P):
            d = np.linalg.norm(T[:, None, :, :] - P[None, :, None, :], axis=3)
            ds.append(np.nanmin(d, axis=2))
    return np.nanmedian(np.concatenate(ds, 1), 1) if ds else np.full(len(T_by_team[0]), np.inf)


def players(xy_f, team):
    out = []
    for t in (0, 1):
        P = xy_f[team == t]
        out.append(P[np.isfinite(P[:, 0])])
    return out


def build():
    xy, _, ball, team, _ = load_cache()
    z = np.load(os.path.join(CACHE, "atd_broadcast.npz"))
    bxy, bball, bteam, bvalid = z["xy"], z["ball"], z["team"], z["valid"]
    n, nb = len(xy), len(bxy)
    half = F.halves_mask(n) > 0
    nplay = np.isfinite(bxy[..., 0]).sum(1)
    usable = bvalid & (nplay >= MIN_PLAYERS)

    # 1. coarse
    cand = np.flatnonzero(half)[::5]
    Tc = [xy[cand][:, team == t] for t in (0, 1)]
    us, offs = [], []
    for u in range(0, nb, COARSE_STEP):
        if not usable[u]:
            continue
        c = match_cost(players(bxy[u], bteam), Tc)
        j = int(np.nanargmin(c))
        if c[j] < GOOD:
            us.append(u)
            offs.append(cand[j] - u)
    us, offs = np.array(us), np.array(offs)
    # a coarse match far from its neighbours is a replay or a false match
    keep = np.abs(offs - median_filter(offs, 5, mode="nearest")) < 100
    base = np.interp(np.arange(nb), us[keep], offs[keep]).round().astype(int)

    # 2. fine
    off = np.full(nb, np.nan)
    cost = np.full(nb, np.inf)
    shifts = np.arange(-SEARCH, SEARCH + 1)
    for u in np.flatnonzero(usable):
        t = u + base[u] + shifts
        ok = (t >= 0) & (t < n)
        if not ok.any():
            continue
        t = t[ok]
        c = match_cost(players(bxy[u], bteam), [xy[t][:, team == k] for k in (0, 1)])
        j = int(np.nanargmin(c))
        if c[j] < GOOD and 0 < j < len(t) - 1:      # an edge minimum is not a minimum
            off[u], cost[u] = t[j] - u, c[j]
    good = np.isfinite(off)
    smooth = np.full(nb, np.nan)
    smooth[good] = median_filter(off[good], 25, mode="nearest")
    agree = good & (np.abs(off - smooth) <= 3)

    # 3. borrow
    bb = np.full((n, 2), np.nan, np.float32)
    src = np.flatnonzero(agree & np.isfinite(bball[:, 0]))
    tgt = (src + smooth[src]).astype(int)
    inside = (tgt >= 0) & (tgt < n)
    bb[tgt[inside]] = bball[src[inside], :2]
    np.save(os.path.join(CACHE, "ball_broadcast.npy"), bb)

    tv = np.isfinite(ball[:, 0]) & half
    both = tv & np.isfinite(bb[:, 0])
    gap = np.linalg.norm(ball[both, :2] - bb[both], axis=1)
    filled = (~tv) & half & np.isfinite(bb[:, 0])
    print("broadcast frames aligned %.3f (usable %.3f)" % (agree.mean(), usable.mean()))
    print("ball agreement where both see it: median %.2f m, <2 m %.3f (n=%d)"
          % (np.median(gap), (gap < 2).mean(), both.sum()))
    print("tactical missing-ball graded frames filled: %.3f"
          % (filled.sum() / max(1, ((~tv) & half).sum())))


if __name__ == "__main__":
    build()
