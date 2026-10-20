"""Read the tactical ATD + Smart Tagging once and cache them as compact numpy arrays.

    python -m cochinae.load            # writes $MTAH_CACHE/atd.npz and tags.json

Everything downstream works on the cache: players (F, S, 2) metres, speeds (F, S), ball (F, 2)
metres, team of each slot (S,). Index 0 of every array is video frame 0 (frame_id - 1).
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

CACHE = os.environ.get("MTAH_CACHE") or os.path.join(DATA_DIR, "cache")


def team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def _triple(group):
    a, b, c = group.split(",")[:3]
    return float(a), float(b), float(c)


def parse_atd(raw_path, n_slots, n_frames, pitch):
    length, width = pitch
    xy = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    sp = np.full((n_frames, n_slots), np.nan, np.float32)
    ball = np.full((n_frames, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            for s, g in enumerate(parts[1].split(";")[:n_slots]):
                if g and g[0] != "N":
                    x, y, v = _triple(g)
                    xy[f, s] = (x * length, y * width)
                    sp[f, s] = v
            b = parts[2]
            if b and b[0] != "N":
                vals = [float(t) if t[:1] != "N" else np.nan for t in b.split(",")[:3]]
                ball[f, :len(vals)] = vals
    ball[:, 0] *= length
    ball[:, 1] *= width
    return xy, sp, ball


def parse_tags(path, fps):
    out = []
    for inst in ET.parse(path).iter("instance"):
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        out.append({"code": (inst.findtext("code") or "").upper(),
                    "start": int(round(s * fps)), "end": int(round(e * fps)), "tags": tags})
    return out


def build(game=GAME):
    meta, raw, tagging = require(["atd_meta", "atd_raw", "smart_tagging"], game)
    team = team_of_slots(meta)
    n = source_frames("tactical", game)
    xy, sp, ball = parse_atd(raw, len(team), n, GAMES[game]["pitch"])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "atd.npz"), xy=xy, sp=sp, ball=ball, team=team)
    tags = parse_tags(tagging, GAMES[game]["fps"])
    with open(os.path.join(CACHE, "tags.json"), "w") as fh:
        json.dump(tags, fh)
    # Shapes and coverage only: the data itself never leaves this machine.
    print("slots", len(team), "frames", n, "ball seen %.3f" % np.isfinite(ball[:, 0]).mean(),
          "tags", len(tags))


def load_cache():
    z = np.load(os.path.join(CACHE, "atd.npz"))
    with open(os.path.join(CACHE, "tags.json")) as fh:
        tags = json.load(fh)
    return z["xy"], z["sp"], z["ball"], z["team"], tags


if __name__ == "__main__":
    build()
