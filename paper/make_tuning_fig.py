"""Figure for the two-pager: SkillCorner cross-validated spellF1 against the switching cost.

    python paper/make_tuning_fig.py $MTAH_DATA_DIR/cache        # reads cache/sc_tuned.json
External data only (tune_skillcorner.py); nothing from the challenge match.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = json.load(open(os.path.join(sys.argv[1], "sc_tuned.json")))["all"]
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 1.9))
style = {(10.0, False): ("#9aa3b2", "-"), (30.0, False): ("#b8322a", "-"),
         (30.0, True): ("#b8322a", "--"), (20.0, False): ("#16213a", "-")}
for (dead, refine), (color, ls) in style.items():
    pts = sorted((s[0], sp) for s, sp, _ in rows if s[1] == dead and s[2] == 25 and s[3] == refine)
    label = "stoppage cost %d%s" % (dead, ", + refinement" if refine else "")
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, marker="o", ms=3,
            lw=1.4, label=label)
ax.set_xlabel("switching cost (nats)")
ax.set_ylabel("spellF1, held-out")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "tuning.png")
fig.savefig(out, dpi=220)
print("->", out)
