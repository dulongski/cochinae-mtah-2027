"""Cache the broadcast ATD (its own timeline) as arrays, like load.py does for tactical."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cochinae.load import CACHE, parse_atd, team_of_slots  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402


def build(game=GAME):
    meta, raw, hom = require(["atd_meta", "atd_raw", "homography"], game, source="broadcast")
    team = team_of_slots(meta)
    n = source_frames("broadcast", game)
    xy, sp, ball = parse_atd(raw, len(team), n, GAMES[game]["pitch"])
    valid = np.loadtxt(hom, delimiter=",", skiprows=1, usecols=(11,))
    np.savez_compressed(os.path.join(CACHE, "atd_broadcast.npz"), xy=xy, sp=sp, ball=ball,
                        team=team, valid=valid.astype(bool))
    print("broadcast slots", len(team), "frames", n,
          "ball seen %.3f" % np.isfinite(ball[:, 0]).mean(), "valid hom %.3f" % valid.mean())


if __name__ == "__main__":
    build()
