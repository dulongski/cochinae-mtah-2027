"""Render the paper's figures from aggregate numbers only.

    python paper/make_figs.py $MTAH_DATA_DIR/cache       # needs matplotlib

Reads cache/ablation.json and cache/fig_data.json (written by paper/collect.py): medians over
events and calibration shares, never frame-level data.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CACHE = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(CACHE, "fig_data.json")))
A = json.load(open(os.path.join(CACHE, "ablation.json")))

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, GREY, TEAL = "#1f2a44", "#c0392b", "#9aa3b2", "#1f8a70"
t = np.array(D["offs"]) / 25

# 1. ablation ladder
short = ["proximity\nonly", "shape\nonly", "shape + prox\n+ dead + Viterbi", "+ lag\ncorrection",
         "+ self-\ntraining", "+ boundary\nrefinement"]
names = ["baseline\n(smooth50)"] + short
spell = [0.3256] + [r[1] for r in A]
macro = [0.6207] + [r[2] for r in A]
fig, ax = plt.subplots(figsize=(7.2, 2.8))
x, w = np.arange(len(names)), 0.38
ax.bar(x - w / 2, macro, w, color=GREY, label="macroF1")
bars = ax.bar(x + w / 2, spell, w, color=INK, label="spellF1 (ranking)")
bars[-1].set_color(ACC)
for i, v in enumerate(spell):
    ax.text(i + w / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Ablation on the public 5-minute window (17 possessions; noise band about ±0.4 "
             "spellF1)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ablation.png"), dpi=200)

# 2. median signal profiles around true boundaries
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
for ax, k, title in ((axs[0], "restart", "Restart (D to live)"),
                     (axs[1], "stoppage", "Stoppage (live to D)")):
    ax2 = ax.twinx()
    ax.plot(t, D[k]["speed"], color=INK, lw=1.6)
    ax2.plot(t, D[k]["p_dead"], color=ACC, lw=1.4, ls="--")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_title("%s, n=%d" % (title, D[k]["n"]), fontsize=8.5, loc="left")
    ax.set_xlabel("s from true boundary")
    ax.set_ylabel("mean player speed (m/s)", color=INK, fontsize=7.5)
    ax2.set_ylabel("P(dead)", color=ACC, fontsize=7.5)
    ax2.set_ylim(0, 1)
    ax2.spines["top"].set_visible(False)
ax = axs[2]
ax.plot(t, D["switch"]["d_new"], color=TEAL, lw=1.5, label="new team to ball")
ax.plot(t, D["switch"]["d_old"], color=GREY, lw=1.5, label="old team to ball")
ax3 = ax.twinx()
ax3.plot(t, D["switch"]["p_new"], color=ACC, lw=1.4, ls="--")
ax3.set_ylim(0, 1)
ax3.set_ylabel("P(new team), shape", color=ACC, fontsize=7.5)
ax3.spines["top"].set_visible(False)
ax.axvline(0, color=GREY, lw=0.8)
ax.set_title("Team switch, n=%d" % D["switch"]["n"], fontsize=8.5, loc="left")
ax.set_xlabel("s from true switch")
ax.set_ylabel("nearest distance (m)", fontsize=7.5)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "profiles.png"), dpi=200)

# 3. off-the-shelf ball detectors
fig, ax = plt.subplots(figsize=(3.8, 2.6))
for name, c, label in (("yolo", INK, "YOLOv8s (COCO)"), ("fab", ACC, "FootAndBall")):
    s = D["calib"][name]
    ax.plot(s["share"], [np.nan if v is None else v for v in s["within2"]], "o-", color=c,
            label=label, ms=3.5)
ax.set_xlabel("share of frames with a detection")
ax.set_ylabel("share within 2 m of tracked ball")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Off-the-shelf detectors, 2,000 frames", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "detectors.png"), dpi=200)
print("figures ->", OUT)
