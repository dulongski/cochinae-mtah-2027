"""Find the ball in the tactical video where the tracking lost it.

Two steps, because they run in different environments:

    python -m cochinae.video_ball select                 # model env: which frames to look at
    python -m cochinae.video_ball detect                 # vision env (ultralytics): run YOLO

`select` writes $CACHE/video_frames.npz: every graded frame where the ATD has no ball and the
dead-ball model is not sure play is stopped (P(dead) < 0.95), plus a calibration sample of 2,000
frames where the ATD does have the ball. The sample is used to measure how well detections agree
with the tracked ball.

`detect` decodes the video once, front to back, and runs a COCO-pretrained YOLOv8s ("sports
ball", class 32) at full resolution on the selected frames. Each detection is projected to the
pitch with the inverse of that frame's homography. Writes $CACHE/video_ball.npz with up to 3
candidates per frame: pitch x/y in metres, confidence.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from cochinae.load import CACHE  # noqa: E402

LENGTH, WIDTH = GAMES[GAME]["pitch"]
WEIGHTS = os.environ.get("MTAH_YOLO") or os.path.join(os.path.dirname(CACHE), "models",
                                                      "yolov8s.pt")
P_DEAD_MAX = 0.95
N_CALIB = 2000
TOPK = 3
BATCH = 16
IMGSZ = 1920
SPORTS_BALL = 32


def select():
    from cochinae import feats as F, model as M
    from cochinae.evalu import gt_labels
    from cochinae.load import load_cache
    _, aux = M.run(gt=gt_labels(), use_video=False)
    _, _, ball, _, _ = load_cache()
    graded = F.halves_mask(len(ball)) > 0
    seen = np.isfinite(ball[:, 0])
    target = np.flatnonzero(graded & ~seen & (aux["p_dead"] < P_DEAD_MAX))
    rng = np.random.default_rng(0)
    calib = np.sort(rng.choice(np.flatnonzero(graded & seen), N_CALIB, replace=False))
    np.savez(os.path.join(CACHE, "video_frames.npz"), target=target, calib=calib)
    print("target frames", len(target), "calibration frames", len(calib))


def to_pitch(Hm, u, v):
    """Pixel -> pitch metres through the inverse homography (pitch 0-1 -> pixels)."""
    p = np.linalg.solve(Hm, np.array([u, v, 1.0]))
    return p[0] / p[2] * LENGTH, p[1] / p[2] * WIDTH


def detect():
    import cv2
    from ultralytics import YOLO
    video, hom = require(["video", "homography"], GAME)
    sel = np.load(os.path.join(CACHE, "video_frames.npz"))
    frames = np.union1d(sel["target"], sel["calib"])
    want = np.zeros(source_frames("tactical", GAME), bool)
    want[frames] = True

    H = np.loadtxt(hom, delimiter=",", skiprows=1, usecols=range(1, 10)).reshape(-1, 3, 3)
    model = YOLO(WEIGHTS)
    out_f, out_c = [], []

    def flush(batch_idx, batch_img):
        res = model.predict(batch_img, imgsz=IMGSZ, device="mps", classes=[SPORTS_BALL],
                            conf=0.05, verbose=False)
        for f, r in zip(batch_idx, res):
            cand = np.full((TOPK, 3), np.nan, np.float32)
            boxes = r.boxes
            order = np.argsort(-boxes.conf.cpu().numpy())[:TOPK]
            for j, k in enumerate(order):
                x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy()
                u, v = (x1 + x2) / 2, y2 - (y2 - y1) / 2
                if np.isfinite(H[f]).all():
                    cand[j, :2] = to_pitch(H[f], u, v)
                cand[j, 2] = float(boxes.conf[k])
            out_f.append(f)
            out_c.append(cand)

    cap = cv2.VideoCapture(video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_video == len(want), "video has %d frames, expected %d" % (n_video, len(want))
    last = int(frames.max())
    batch_idx, batch_img = [], []
    for f in range(last + 1):
        if not cap.grab():
            raise RuntimeError("video ended early at frame %d" % f)
        if not want[f]:
            continue
        ok, img = cap.retrieve()
        if not ok:
            raise RuntimeError("could not decode frame %d" % f)
        batch_idx.append(f)
        batch_img.append(img)
        if len(batch_img) == BATCH:
            flush(batch_idx, batch_img)
            batch_idx, batch_img = [], []
        if f % 20000 == 0:
            print("frame", f, "detections so far", len(out_f), flush=True)
    if batch_img:
        flush(batch_idx, batch_img)
    cap.release()
    np.savez(os.path.join(CACHE, "video_ball.npz"), frame=np.array(out_f),
             cand=np.array(out_c))
    print("done:", len(out_f), "frames processed")


def calib():
    """How often the best detection lands within 2 m of the tracked ball, by confidence."""
    from cochinae.load import load_cache
    _, _, ball, _, _ = load_cache()
    sel = np.load(os.path.join(CACHE, "video_frames.npz"))
    z = np.load(os.path.join(CACHE, "video_ball.npz"))
    idx = np.isin(z["frame"], sel["calib"])
    f, c = z["frame"][idx], z["cand"][idx, 0, :]
    d = np.linalg.norm(c[:, :2] - ball[f, :2], axis=1)
    print("calibration frames with any detection: %.3f" % np.isfinite(c[:, 2]).mean())
    for lo in (0.05, 0.15, 0.25, 0.4, 0.6):
        m = c[:, 2] >= lo
        print("conf >= %.2f: share %.3f  within 2 m %.3f  median dist %.2f m"
              % (lo, m.mean(), (d[m] < 2).mean() if m.any() else np.nan,
                 np.median(d[m]) if m.any() else np.nan))
    t = np.isin(z["frame"], sel["target"])
    print("target frames with a detection >= 0.25: %.3f" % (z["cand"][t, 0, 2] >= 0.25).mean())


if __name__ == "__main__":
    {"select": select, "detect": detect, "calib": calib}[sys.argv[1]]()
