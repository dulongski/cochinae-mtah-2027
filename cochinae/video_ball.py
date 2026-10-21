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
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from cochinae.load import CACHE  # noqa: E402

LENGTH, WIDTH = GAMES[GAME]["pitch"]
WEIGHTS = os.environ.get("MTAH_YOLO") or os.path.join(os.path.dirname(CACHE), "models",
                                                      "yolov8s.pt")
FAB_DIR = os.environ.get("MTAH_FAB") or os.path.join(os.path.dirname(CACHE), "models", "fab",
                                                     "repo")
P_DEAD_MAX = 0.95
N_CALIB = 2000
TOPK = 3
BATCH = 4          # 16 at 1920 px pushed a 16 GB M1 into swap
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


def yolo_predictor():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)

    def predict(images):
        out = []
        for r in model.predict(images, imgsz=IMGSZ, device="mps", classes=[SPORTS_BALL],
                               conf=0.05, half=True, verbose=False):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            out.append([((x1 + x2) / 2, (y1 + y2) / 2, c) for (x1, y1, x2, y2), c in zip(xyxy, conf)])
        return out
    return predict


def fab_predictor():
    """FootAndBall (Komorowski et al., 2020): a small detector built for long-shot soccer video.

    Code and weights: github.com/jac99/footandball, cloned to $MTAH_FAB. Its detection head
    allocates float64 tensors, which the Apple GPU does not support, so the backbone runs on the
    GPU and the head on the CPU.
    """
    import torch
    sys.path.insert(0, FAB_DIR)
    import network.footandball as footandball
    import data.augmentation as augmentation
    model = footandball.model_factory("fb1", "detect", ball_threshold=0.0, max_ball_detections=TOPK)
    model.load_state_dict(torch.load(os.path.join(FAB_DIR, "models", "model_20201019_1416_final.pth"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    head = model.detect
    model.detect = lambda p, b, ball: head(p.cpu(), b.cpu(), ball.cpu())
    model = model.to("mps")

    def predict(images):
        x = torch.stack([augmentation.numpy2tensor(im) for im in images]).to("mps")
        with torch.no_grad():
            res = model(x)
        out = []
        for r in res:
            keep = r["labels"].cpu().numpy() == augmentation.BALL_LABEL
            boxes, scores = r["boxes"].cpu().numpy()[keep], r["scores"].cpu().numpy()[keep]
            out.append([((x1 + x2) / 2, (y1 + y2) / 2, float(c))
                        for (x1, y1, x2, y2), c in zip(boxes, scores)])
        return out
    return predict


def detect(method="yolo", which="all"):
    """Run a detector over the selected frames. which = all | calib."""
    import cv2
    video, hom = require(["video", "homography"], GAME)
    sel = np.load(os.path.join(CACHE, "video_frames.npz"))
    frames = sel["calib"] if which == "calib" else np.union1d(sel["target"], sel["calib"])
    want = np.zeros(source_frames("tactical", GAME), bool)
    want[frames] = True

    H = np.loadtxt(hom, delimiter=",", skiprows=1, usecols=range(1, 10)).reshape(-1, 3, 3)
    predict = {"yolo": yolo_predictor, "fab": fab_predictor}[method]()
    out_f, out_c = [], []

    def flush(batch_idx, batch_img):
        for f, dets in zip(batch_idx, predict(batch_img)):
            cand = np.full((TOPK, 3), np.nan, np.float32)
            for j, (u, v, c) in enumerate(sorted(dets, key=lambda d: -d[2])[:TOPK]):
                if np.isfinite(H[f]).all():
                    cand[j, :2] = to_pitch(H[f], u, v)
                cand[j, 2] = c
            out_f.append(f)
            out_c.append(cand)

    cap = cv2.VideoCapture(video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_video == len(want), "video has %d frames, expected %d" % (n_video, len(want))
    last = int(frames.max())
    batch_idx, batch_img = [], []
    t0, total = time.time(), int(want.sum())
    for f in range(last + 1):
        if not cap.grab():
            raise RuntimeError("video ended early at frame %d" % f)
        if f % 5000 == 0 and out_f:
            rate = len(out_f) / (time.time() - t0)
            print("video frame %d  processed %d/%d  %.1f fps  eta %.0f min"
                  % (f, len(out_f), total, rate, (total - len(out_f)) / rate / 60), flush=True)
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
    if batch_img:
        flush(batch_idx, batch_img)
    cap.release()
    path = os.path.join(CACHE, "video_ball_%s%s.npz" % (method, "_calib" if which == "calib" else ""))
    np.savez(path, frame=np.array(out_f), cand=np.array(out_c))
    print("done:", len(out_f), "frames ->", path)


def calib(name="video_ball_yolo"):
    """How often the best detection lands within 2 m of the tracked ball, by confidence."""
    from cochinae.load import load_cache
    _, _, ball, _, _ = load_cache()
    sel = np.load(os.path.join(CACHE, "video_frames.npz"))
    z = np.load(os.path.join(CACHE, name + ".npz"))
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
    {"select": select, "detect": detect, "calib": calib}[sys.argv[1]](*sys.argv[2:])
