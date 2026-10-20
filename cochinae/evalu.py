"""Score a label array against the public 5-minute ground truth, in-process."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def gt_labels():
    import pandas as pd
    return pd.read_csv(GT_PATH)["label"].astype(str).str.strip().str.upper().values


def score(labels, gt=None):
    """(spellF1, macroF1, n_pred_possessions_in_window) on the public window."""
    gt = gt_labels() if gt is None else gt
    g = pm.label_codes(gt)
    p = pm.label_codes(np.asarray(labels))
    runs = pm.valid_runs(g)
    cm = pm.scored_confusion(pm.confusion(g, p))
    filled = pm.fill_abstentions(p, runs)
    sp = pm.spell_scores(g, filled, runs)
    n_pred = len(pm.possession_spells(filled, runs))
    return sp["mean_f1"], pm.macro_f1(cm), n_pred
