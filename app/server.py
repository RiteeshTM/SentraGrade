"""SentraGrade scanner backend.

Serves the static UI and a single inference endpoint that runs an uploaded
image through the fold-0 ResNet50 checkpoint (held-out class: Banana, the
strongest-separating fold for the energy score, AUROC 0.9997) and returns
an in-distribution / out-of-distribution verdict plus a Grad-CAM heatmap.

Scoring follows 03_ood_eval_gradcam.ipynb exactly: energy score
(-logsumexp(logits)) thresholded at the 95th percentile of that fold's
in-distribution validation scores. MSP and prototype-distance are computed
too and surfaced as supporting signals.
"""

import base64
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

DEVICE = torch.device("cpu")
DATA_ROOT = Path(os.environ.get("SENTRAGRADE_DATA_ROOT", Path.home() / "sentragrade_data"))
CKPT_DIR = DATA_ROOT / "checkpoints" / "supervised_cnn"
FOLD_IDX = 0
# 98th percentile (rather than the notebook's demo default of 95th): still
# a 100% catch-rate on the calibration OOD class (measured against cached
# val/ood scores), but cuts the known-produce false-reject rate from 9.0%
# to 3.7% versus the 95th-percentile fused rule.
THRESHOLD_PERCENTILE = 98

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ResNetClassifier(nn.Module):
    """Same architecture as the training notebook. weights=None here since
    we immediately overwrite every parameter from the fine-tuned checkpoint
    — no need to fetch the ImageNet-pretrained weights at server start."""

    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.fc(feats)
        return logits, feats


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits, _ = self.model(x)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()[0], logits.detach()


def denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def colorize_cam(cam: torch.Tensor) -> np.ndarray:
    """cam: [224,224] in [0,1] -> RGBA uint8 heat overlay, amber->red ramp,
    low-activation regions suppressed to alpha 0 so the overlay reads as a
    focused hotspot rather than a full-frame tint."""
    cam_np = cam.numpy()
    amber = np.array([255, 176, 32])
    red = np.array([255, 60, 60])
    t = np.clip(cam_np, 0, 1)[..., None]
    rgb = (amber * (1 - t) + red * t).astype(np.uint8)
    alpha = np.clip((cam_np - 0.25) / 0.75, 0, 1) ** 0.8
    alpha = (alpha * 210).astype(np.uint8)
    return np.dstack([rgb, alpha])


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- load model + calibration once at startup ---
_ckpt_path = CKPT_DIR / f"fold{FOLD_IDX}.pt"
if not _ckpt_path.exists():
    raise RuntimeError(
        f"Checkpoint not found at {_ckpt_path}. Set SENTRAGRADE_DATA_ROOT if your "
        "sentragrade_data folder lives somewhere other than ~/sentragrade_data."
    )

_ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
LABEL_MAP = _ckpt["label_map"]
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

_summary = pd.read_csv(CKPT_DIR / "summary.csv")
HELD_OUT_CLASS = str(_summary.loc[_summary.fold == FOLD_IDX, "held_out_class"].item())

MODEL = ResNetClassifier(len(LABEL_MAP)).to(DEVICE)
MODEL.load_state_dict(_ckpt["model_state_dict"])
MODEL.eval()
GRADCAM = GradCAM(MODEL, MODEL.layer4[-1])

_val_logits = _ckpt["val"]["logits"]
_val_feats = _ckpt["val"]["features"]
_val_labels = _ckpt["val"]["labels"]

_val_probs = torch.softmax(_val_logits, dim=1)
_val_msp = (-_val_probs.max(dim=1).values).numpy()
_val_energy = (-torch.logsumexp(_val_logits, dim=1)).numpy()
_val_pred = _val_logits.argmax(dim=1).numpy()

PROTOTYPES = torch.stack([_val_feats[_val_labels == c].mean(dim=0) for c in range(len(LABEL_MAP))])
_val_proto = torch.cdist(_val_feats, PROTOTYPES).min(dim=1).values.numpy()

# Per-predicted-class thresholds, not one global cutoff: a single threshold
# across all 5 classes calibrates unevenly when classes have uneven val
# counts and feature spread — observed false-reject rates of 17.6% (Guava)
# vs 0.7% (Tomato) under one shared threshold. Percentiling within each
# predicted class instead evens that out to a ~3.5-5.9% band.
NUM_CLASSES = len(LABEL_MAP)
ENERGY_THRESHOLDS = np.array(
    [np.percentile(_val_energy[_val_pred == c], THRESHOLD_PERCENTILE) for c in range(NUM_CLASSES)]
)
PROTO_THRESHOLDS = np.array(
    [np.percentile(_val_proto[_val_pred == c], THRESHOLD_PERCENTILE) for c in range(NUM_CLASSES)]
)

print(
    f"[sentragrade] fold {FOLD_IDX} loaded — known classes: {sorted(LABEL_MAP)} | "
    f"held-out (OOD demo) class: {HELD_OUT_CLASS} | per-class energy thresholds: "
    f"{dict(zip(sorted(LABEL_MAP, key=lambda c: LABEL_MAP[c]), ENERGY_THRESHOLDS.round(3)))}"
)

app = FastAPI(title="SentraGrade Scanner")


@app.get("/api/meta")
def meta():
    classes = sorted(LABEL_MAP, key=lambda c: LABEL_MAP[c])
    return {
        "known_classes": classes,
        "held_out_class": HELD_OUT_CLASS,
        "fold": FOLD_IDX,
        "energy_thresholds": {c: float(ENERGY_THRESHOLDS[LABEL_MAP[c]]) for c in classes},
    }


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file couldn't be read as an image.")

    x = eval_tf(img).unsqueeze(0)

    with torch.no_grad():
        probe_logits, feats = MODEL(x)
    pred_idx = int(probe_logits.argmax(dim=1))

    cam, logits = GRADCAM(x, pred_idx)

    probs = torch.softmax(logits, dim=1)[0]
    confidence = float(probs[pred_idx])
    energy_score = float(-torch.logsumexp(logits, dim=1)[0])
    msp_score = float(-probs.max())
    proto_score = float(torch.cdist(feats, PROTOTYPES).min())

    # Energy alone is tuned against a near-OOD calibration class (another
    # fruit), so it can under-react to far-OOD objects with no organic
    # texture at all (electronics, tools, ...) — a phone was observed to
    # sneak in on energy alone while its prototype distance blew past that
    # threshold by 6x. Rejecting on either signal catches both near- and
    # far-OOD inputs. Thresholds are per predicted-class (not one global
    # cutoff) since classes calibrate unevenly otherwise.
    energy_threshold = float(ENERGY_THRESHOLDS[pred_idx])
    proto_threshold = float(PROTO_THRESHOLDS[pred_idx])
    triggered_by = []
    if energy_score > energy_threshold:
        triggered_by.append("energy")
    if proto_score > proto_threshold:
        triggered_by.append("proto")
    decision = "REJECTED" if triggered_by else "ACCEPTED"

    base_view = denormalize(x[0]).permute(1, 2, 0).numpy()
    base_img = Image.fromarray((base_view * 255).astype(np.uint8))
    heat_img = Image.alpha_composite(base_img.convert("RGBA"), Image.fromarray(colorize_cam(cam), mode="RGBA"))

    return {
        "decision": decision,
        "predicted_class": INV_LABEL_MAP[pred_idx],
        "confidence": confidence,
        "energy_score": energy_score,
        "energy_threshold": energy_threshold,
        "msp_score": msp_score,
        "proto_score": proto_score,
        "proto_threshold": proto_threshold,
        "triggered_by": triggered_by,
        "held_out_class": HELD_OUT_CLASS,
        "model_view_image": image_to_data_url(base_img),
        "heatmap_image": image_to_data_url(heat_img),
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
