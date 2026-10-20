"""Boundary refinement after decoding.

The decoder works on features averaged over seconds, so its boundaries are blurred by about that
much. Here each boundary is moved, within a small window, to where a sharp per-frame signal says
it is:

  * H <-> A   -- the changepoint of the unsmoothed ball-proximity evidence: the split that best
                 puts "old team nearer" frames before it and "new team nearer" frames after it.
  * D -> live -- the restart: onset of the ball travelling again.
  * live -> D -- the stoppage: the ball leaving the pitch, or the last frame it is seen moving.
"""

import numpy as np

from cochinae import feats as F


def proximity_evidence(bf, r=3.0, clip=3.0):
    """Per-frame evidence for home (+) / away (-), zero where the ball is not near anyone."""
    dmin = np.minimum(bf["dH"], bf["dA"])
    e = np.clip(np.nan_to_num(bf["dA"] - bf["dH"]), -clip, clip)
    return np.where(bf["vis"] & (dmin < r), e, 0.0)


def runs(labels, s, e):
    """[(start, stop, label)] inside [s, e]."""
    v = labels[s:e + 1]
    cut = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(v)]
    return [(s + a, s + b, v[a]) for a, b in zip(cut[:-1], cut[1:])]


def refine_switches(labels, ev, window=75):
    out = labels.copy()
    for _, s, e in F.half_video_blocks(F.GAME):
        rr = runs(labels, s, e)
        for (a0, b0, l0), (a1, b1, l1) in zip(rr[:-1], rr[1:]):
            if {l0, l1} != {"H", "A"}:
                continue
            t = b0                                   # first frame of the new label
            lo, hi = max(a0 + 1, t - window), min(b1 - 1, t + window)
            if hi <= lo:
                continue
            sgn = 1.0 if l0 == "H" else -1.0        # + = evidence for the old team
            seg = ev[lo:hi] * sgn
            # score(split at k) = old evidence before k - old evidence after k
            cs = np.r_[0, np.cumsum(seg)]
            score = cs - (cs[-1] - cs)
            best = np.flatnonzero(score >= score.max() - 1e-9)
            k = best[np.argmin(np.abs(best + lo - t))]
            new_t = lo + int(k)
            if new_t < t:
                out[new_t:t] = l1
            elif new_t > t:
                out[t:new_t] = l0
    return out


def _best_split(seg, before_high):
    """Index k maximising the mean difference between seg[:k] and seg[k:] (signed)."""
    n = len(seg)
    if n < 4:
        return None
    cs = np.r_[0, np.cumsum(seg)]
    k = np.arange(2, n - 1)
    m1 = cs[k] / k
    m2 = (cs[-1] - cs[k]) / (n - k)
    gain = (m1 - m2) if before_high else (m2 - m1)
    # weight by balance so tiny edge segments do not win on noise
    gain = gain * np.sqrt(k * (n - k) / n)
    return int(k[np.argmax(gain)])


def refine_dead(labels, sp, before=150, after=25, mode="split"):
    """Move D boundaries to the player-speed changepoint.

    restart (D -> live): players are slowest at the restart and accelerate after it.
    stoppage (live -> D): players decelerate just after it.
    """
    spd = F.rolling_mean(np.nanmean(sp, 1), 6)
    spd = np.where(np.isfinite(spd), spd, np.nanmedian(spd))
    out = labels.copy()
    for _, s, e in F.half_video_blocks(F.GAME):
        rr = runs(labels, s, e)
        for (a0, b0, l0), (a1, b1, l1) in zip(rr[:-1], rr[1:]):
            t = b0
            if l0 == "D" and l1 != "D":                 # restart
                lo, hi = max(a0 + 1, t - before), min(b1 - 1, t + after)
                if hi - lo < 4:
                    continue
                if mode == "min":
                    k = int(np.argmin(spd[lo:hi]))
                else:
                    k = _best_split(spd[lo:hi], before_high=False)
                new_t = lo + k
                if new_t < t:
                    out[new_t:t] = l1
                elif new_t > t:
                    out[t:new_t] = "D"
            elif l0 != "D" and l1 == "D":               # stoppage
                lo, hi = max(a0 + 1, t - before), min(b1 - 1, t + after)
                if hi - lo < 4:
                    continue
                k = _best_split(spd[lo:hi], before_high=True)
                new_t = lo + k
                if new_t < t:
                    out[new_t:t] = "D"
                elif new_t > t:
                    out[t:new_t] = l0
    return out
