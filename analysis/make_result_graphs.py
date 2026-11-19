"""
SentraGrade — Results graphs for the Review 2 slides.

Read-only: this script trains nothing. It reads the checkpoints and OOD-eval
CSVs already produced by the training notebooks (other/02_supervised_cnn_baseline.ipynb
+ other/03_ood_eval_gradcam.ipynb for PyTorch ResNet50, supervised_ood_eval_gradcam_tf.ipynb
for the TF/Keras ResNet50) and re-derives every number and figure on the
Results slides from them:

  g1  ResNet50 (PyTorch) AUROC per held-out class
  g2  ResNet50 (PyTorch) FPR@95 per held-out class
  g3  Mean AUROC by backbone x scoring method (ResNet50 PT/TF + DINOv2)
  g4  Energy-score ROC curves, all 6 folds + hardest-fold Energy-vs-MSP
  g5  Energy-score histograms (known vs OOD), 3 example folds
  g6  Per-epoch training loss / validation accuracy (ResNet50, PyTorch)
  g7  TR-6 sRGB class distribution (unique images after de-duplication)

Requires ~/sentragrade_data/checkpoints/{supervised_cnn,supervised_cnn_tf}/
(produced by the training notebooks) to exist. DINOv2's per-fold AUROC
values are not saved anywhere as a CSV — they're only printed as notebook
output — so they're hardcoded below from that printed output; update them
if SentraGrade_DINOv2_Notebook.ipynb is re-run.

Usage:
    python analysis/make_result_graphs.py
Output:
    analysis/outputs/*.png
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

PROJ = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

DATA_ROOT = Path.home() / "sentragrade_data"
PT_DIR = DATA_ROOT / "checkpoints" / "supervised_cnn"
TF_DIR = DATA_ROOT / "checkpoints" / "supervised_cnn_tf"
for d in (PT_DIR, TF_DIR):
    if not d.exists():
        raise RuntimeError(
            f"{d} not found. Run the training notebooks first (or point "
            "DATA_ROOT at wherever their checkpoints/ folder lives)."
        )

RED, INK, GREY = "#C00000", "#1A1A1A", "#8A8A8A"
COL = {"energy": "#C00000", "msp": "#2F6DB5", "proto": "#E08A00"}
LBL = {"energy": "Energy", "msp": "MSP", "proto": "Prototype"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 11,
        "axes.titleweight": "bold",
    }
)

# ---------------------------------------------------------------------------
# Load the two ResNet50 runs' per-fold OOD-eval CSVs (written by the
# notebooks' Section 7 "Evaluation — AUROC and FPR@95" cells).
# ---------------------------------------------------------------------------
pt_eval = pd.read_csv(PT_DIR / "ood_eval_per_fold.csv")
tf_eval = pd.read_csv(TF_DIR / "ood_eval_per_fold.csv")
classes = list(pt_eval[pt_eval.method == "energy"].sort_values("fold").held_out_class)
short = [c.replace("Indian_Gooseberry", "Gooseberry") for c in classes]

# DINOv2's per-fold AUROC, transcribed from the printed output of
# SentraGrade_DINOv2_Notebook.ipynb Section 9 ("Compute OOD scores for
# every fold") — no CSV is saved by that notebook, only stdout.
DINOV2_AUROC = {
    "energy": [1.0000, 0.9870, 0.9996, 0.9937, 1.0000, 1.0000],
    "msp": [1.0000, 1.0000, 0.9993, 0.9953, 1.0000, 1.0000],
    "proto": [1.0000, 1.0000, 0.9999, 0.9989, 0.9999, 1.0000],
}


# ---------------------------------------------------------------------------
# g1 / g2 — ResNet50 (PyTorch): AUROC and FPR@95 per held-out class
# ---------------------------------------------------------------------------
def grouped_bar(df, metric, title, fname, ylim=None, pct=False):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.26
    x = np.arange(6)
    for k, m in enumerate(["energy", "msp", "proto"]):
        v = df[df.method == m].sort_values("fold")[metric].values * (100 if pct else 1)
        ax.bar(x + (k - 1) * w, v, w, label=LBL[m], color=COL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(fold {i})" for i, s in enumerate(short)])
    ax.set_title(title, loc="left")
    ax.set_ylabel(metric + (" (%)" if pct else ""))
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


grouped_bar(pt_eval, "AUROC", "ResNet50 (PyTorch): AUROC per held-out class", "g1_resnet_pt_auroc.png", ylim=(0.9, 1.005))
grouped_bar(pt_eval, "FPR@95", "ResNet50 (PyTorch): FPR@95 per held-out class (lower is better)", "g2_resnet_pt_fpr95.png", pct=True)

# ---------------------------------------------------------------------------
# g3 — mean AUROC by backbone x scoring method
# ---------------------------------------------------------------------------
means = {
    "ResNet50 (PyTorch)": [pt_eval[pt_eval.method == m].AUROC.mean() for m in ["energy", "msp", "proto"]],
    "ResNet50 (TF)": [tf_eval[tf_eval.method == m].AUROC.mean() for m in ["energy", "msp", "proto"]],
    "DINOv2 ViT-S/14": [np.mean(DINOV2_AUROC[m]) for m in ["energy", "msp", "proto"]],
}
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(3)
w = 0.26
for k, m in enumerate(["energy", "msp", "proto"]):
    vals = [means[b][k] for b in means]
    bars = ax.bar(x + (k - 1) * w, vals, w, label=LBL[m], color=COL[m])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(list(means))
ax.set_ylim(0.95, 1.006)
ax.set_ylabel("Mean AUROC over 6 folds")
ax.set_title("Backbone x scoring method: mean AUROC (LOCO, 6 folds)", loc="left")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
fig.tight_layout()
fig.savefig(OUT / "g3_model_comparison_auroc.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# g4 / g5 — ROC curves and score histograms, recomputed from the cached
# validation/OOD logits inside each fold checkpoint (fold{n}.pt). No
# inference is run here; these are the exact logits the notebooks cached
# during training.
# ---------------------------------------------------------------------------
ckpts = {f: torch.load(PT_DIR / f"fold{f}.pt", map_location="cpu", weights_only=False) for f in range(6)}


def energy_msp_scores(ckpt):
    out = {}
    for split in ["val", "ood"]:
        logits = ckpt[split]["logits"]
        out[split] = {
            "energy": (-torch.logsumexp(logits, 1)).numpy(),
            "msp": (-torch.softmax(logits, 1).max(1).values).numpy(),
        }
    return out


scores = {f: energy_msp_scores(ckpts[f]) for f in ckpts}

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
cmap = plt.get_cmap("tab10")
for f in range(6):
    y = np.r_[np.zeros(len(scores[f]["val"]["energy"])), np.ones(len(scores[f]["ood"]["energy"]))]
    s = np.r_[scores[f]["val"]["energy"], scores[f]["ood"]["energy"]]
    fpr, tpr, _ = roc_curve(y, s)
    axs[0].plot(fpr, tpr, color=cmap(f), label=f"{short[f]} (AUROC {roc_auc_score(y, s):.3f})")
axs[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
axs[0].axhline(0.95, color=GREY, lw=0.8, ls=":")
axs[0].set_xlabel("False positive rate")
axs[0].set_ylabel("True positive rate")
axs[0].set_title("Energy score ROC, all 6 folds", loc="left")
axs[0].legend(frameon=False, fontsize=8.5, loc="lower right")

hardest_fold = 3  # Indian_Gooseberry — lowest energy AUROC in the PyTorch run
y = np.r_[np.zeros(len(scores[hardest_fold]["val"]["energy"])), np.ones(len(scores[hardest_fold]["ood"]["energy"]))]
for m in ["energy", "msp"]:
    s = np.r_[scores[hardest_fold]["val"][m], scores[hardest_fold]["ood"][m]]
    fpr, tpr, _ = roc_curve(y, s)
    axs[1].plot(fpr, tpr, color=COL[m], label=f"{LBL[m]} (AUROC {roc_auc_score(y, s):.3f})")
axs[1].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
axs[1].axhline(0.95, color=GREY, lw=0.8, ls=":")
axs[1].set_xlabel("False positive rate")
axs[1].set_title(f"Hardest fold ({short[hardest_fold]}): Energy vs MSP", loc="left")
axs[1].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "g4_roc_curves.png", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, f in zip(axs, [0, 3, 4]):
    idv, ood = scores[f]["val"]["energy"], scores[f]["ood"]["energy"]
    lo, hi = min(idv.min(), ood.min()), max(idv.max(), ood.max())
    bins = np.linspace(lo, hi, 50)
    ax.hist(idv, bins=bins, alpha=0.75, color="#2F6DB5", label="Known (val)", density=True)
    ax.hist(ood, bins=bins, alpha=0.75, color=RED, label=f"OOD ({short[f]})", density=True)
    thr = np.percentile(idv, 95)
    ax.axvline(thr, color=INK, ls="--", lw=1)
    ax.text(thr, ax.get_ylim()[1] * 0.92, " 95th pct of known", fontsize=8, ha="left", color=INK)
    ax.set_title(f"Fold {f}: {short[f]} held out", loc="left")
    ax.set_xlabel("Energy score (higher = more OOD-like)")
    if f == 0:
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "g5_energy_histograms.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# g6 — per-epoch training curves, parsed from the printed stdout captured
# inside other/02_supervised_cnn_baseline.ipynb (no re-training).
# ---------------------------------------------------------------------------
nb = json.load(open(PROJ / "other" / "02_supervised_cnn_baseline.ipynb"))
log_text = ""
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for o in cell.get("outputs", []):
            if "text" in o and "Early stopping" in "".join(o["text"]):
                log_text += "".join(o["text"])

fold_blocks = re.split(r"Fold (\d)\n=+", log_text)
curves = {}
for i in range(1, len(fold_blocks), 2):
    fold_idx = int(fold_blocks[i])
    rows = re.findall(
        r"epoch\s+(\d+)/25\s+train_loss=([\d.]+) train_acc=([\d.]+)\s+val_loss=([\d.]+) val_acc=([\d.]+)",
        fold_blocks[i + 1],
    )
    curves[fold_idx] = np.array(rows, dtype=float)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for fold_idx, arr in sorted(curves.items()):
    axs[0].plot(arr[:, 0], arr[:, 1], color=cmap(fold_idx), marker="o", ms=3, label=short[fold_idx])
    axs[1].plot(arr[:, 0], arr[:, 4], color=cmap(fold_idx), marker="o", ms=3)
axs[0].set_title("Training loss per epoch (ResNet50, 6 folds)", loc="left")
axs[1].set_title("Validation accuracy per epoch", loc="left")
for ax in axs:
    ax.set_xlabel("Epoch")
axs[0].set_ylabel("Cross-entropy loss")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim(0.99, 1.002)
axs[0].legend(frameon=False, fontsize=8.5, ncol=2)
fig.text(
    0.01,
    0.005,
    "Logged by the notebook to 3 decimals; early stopping (patience 6) ended training at epoch 7 (fold 3: epoch 8).",
    fontsize=8,
    color=GREY,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "g6_training_curves.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# g7 — TR-6 sRGB class distribution (unique images per class, from
# folds/fold_summary.csv's ood_n column — each class's OOD count equals its
# total unique-image count, since it's the full class when held out).
# ---------------------------------------------------------------------------
fold_summary = pd.read_csv(DATA_ROOT / "folds" / "fold_summary.csv")
counts = fold_summary.set_index("held_out_class")["ood_n"]
counts.index = [c.replace("Indian_Gooseberry", "Indian Gooseberry") for c in counts.index]
counts = counts.reindex(["Banana", "Carrot", "Guava", "Indian Gooseberry", "Mango", "Tomato"])

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(counts.index, counts.values, color=RED)
for r, v in zip(bars, counts.values):
    ax.text(r.get_x() + r.get_width() / 2, v + 25, str(v), ha="center", fontsize=10)
ax.set_ylabel("Unique sRGB images")
ax.set_title(f"TR-6 sRGB class distribution ({int(counts.sum()):,} unique images)", loc="left")
plt.setp(ax.get_xticklabels(), rotation=20)
fig.tight_layout()
fig.savefig(OUT / "g7_class_distribution.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary — the operating-point numbers quoted on the Results slides
# ---------------------------------------------------------------------------
print(f"Wrote {sorted(p.name for p in OUT.glob('*.png'))} -> {OUT}")
print("\nEnergy-score operating point (95th percentile of known-class energy):")
for f in range(6):
    idv, ood = scores[f]["val"]["energy"], scores[f]["ood"]["energy"]
    thr = np.percentile(idv, 95)
    print(
        f"  fold {f} {classes[f]:18s} threshold={thr:7.3f}  "
        f"OOD caught={100 * (ood > thr).mean():5.1f}%  "
        f"known median={np.median(idv):6.2f}  OOD median={np.median(ood):6.2f}"
    )
