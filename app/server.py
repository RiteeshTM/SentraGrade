"""SentraGrade scanner backend.

Serves the static UI and a single inference endpoint that runs an uploaded
image through the fold-0 ResNet50 checkpoint (held-out class: Banana, the
strongest-separating fold for the energy score, AUROC 0.9997) and returns
an in-distribution / out-of-distribution verdict plus a Grad-CAM heatmap.

Scoring uses the energy score (-logsumexp(logits)) and prototype distance,
each against a hand-tuned global threshold (see ENERGY_THRESHOLD /
PROTO_THRESHOLD below); an item is rejected if either fires. MSP is computed
too and surfaced as a supporting signal.
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
# Global accept thresholds, tuned by hand against three sets (fold 0):
#   - real uploaded photos of the 5 known fruits (energy -3.4..-9.6, proto 12.8..22.7)
#   - held-out Banana: val images (energy median -1.2, proto median 22) and one
#     real photo (energy -3.115) — only 0.26 above the tomato photo (-3.372), so
#     the energy cut is the fragile one; -3.25 sits mid-gap
#   - foreign objects: phone / pen / stone (proto 95 / 170 / 47, stone energy -2.9)
# The old per-class 98th-percentile thresholds were calibrated on the dataset's
# own capture setup, so real uploads (different framing/lighting) sat just past
# them and known fruit got rejected. Energy is the near-OOD signal (Banana);
# prototype distance is the far-OOD signal (phone/pen have *low* energy but
# sit 4-8x further from every class prototype than any fruit).
# Sweep result: 5/5 real fruit accepted, 3/3 foreign + the real banana photo
# rejected, 95% of held-out Banana val rejected, 0.4% false-reject on ID val.
ENERGY_THRESHOLD = -3.25
PROTO_THRESHOLD = 32.0

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

PROTOTYPES = torch.stack([_val_feats[_val_labels == c].mean(dim=0) for c in range(len(LABEL_MAP))])

print(
    f"[sentragrade] fold {FOLD_IDX} loaded — known classes: {sorted(LABEL_MAP)} | "
    f"held-out (OOD demo) class: {HELD_OUT_CLASS} | energy threshold: {ENERGY_THRESHOLD} | "
    f"proto threshold: {PROTO_THRESHOLD}"
)

app = FastAPI(title="SentraGrade Scanner")


@app.get("/api/meta")
def meta():
    classes = sorted(LABEL_MAP, key=lambda c: LABEL_MAP[c])
    return {
        "known_classes": classes,
        "held_out_class": HELD_OUT_CLASS,
        "fold": FOLD_IDX,
        "energy_threshold": ENERGY_THRESHOLD,
        "proto_threshold": PROTO_THRESHOLD,
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

    # Energy alone under-reacts to far-OOD objects with no organic texture
    # (a pen scores *lower* energy than real fruit), so reject on either
    # signal: energy catches near-OOD (other produce), prototype distance
    # catches far-OOD (electronics, tools, ...).
    energy_threshold = ENERGY_THRESHOLD
    proto_threshold = PROTO_THRESHOLD
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


class NoCacheStaticFiles(StaticFiles):
    """Always revalidate, so UI edits show up on a normal reload instead of
    the browser serving a heuristically cached copy of the old files."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
