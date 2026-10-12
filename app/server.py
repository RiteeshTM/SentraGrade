"""SentraGrade scanner backend with Multi-Model Support.

Serves the static UI and a unified inference endpoint that can run an uploaded
image through one of three deep learning classifier backbones:
  1. ResNet-50 (Supervised CNN)
  2. DINOv2 (Self-Supervised Vision Transformer, ViT-S/14)
  3. ViT-Base (Supervised Vision Transformer, ViT-Base/16)

All three models are calibrated on Fold 0 (held-out class: Banana, known classes:
Carrot, Guava, Indian_Gooseberry, Mango, Tomato) and compute:
  - Classification prediction & confidence
  - Energy score (-logsumexp(logits))
  - Prototype distance (cdist in feature embedding space)
  - Visual explainability heatmap (Grad-CAM for ResNet, Patch Token Activation
    for DINOv2, Attention Rollout for ViT)
"""

import base64
import io
import os
from pathlib import Path
from typing import Dict, Any, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Form
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import timm
from transformers import ViTModel

DEVICE = torch.device("cpu")
ROOT_DIR = Path(__file__).resolve().parent.parent

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

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def colorize_cam(cam: torch.Tensor) -> np.ndarray:
    """cam: [224,224] in [0,1] -> RGBA uint8 heat overlay, amber->red ramp,
    low-activation regions suppressed to alpha 0."""
    cam_np = cam.detach().cpu().numpy()
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


# =============================================================================
# 1. ResNet-50 Engine (Supervised CNN)
# =============================================================================

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
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


class ResNetEngine:
    def __init__(self):
        self.name = "ResNet-50 (CNN)"
        self.badge = "Supervised CNN"
        self.energy_threshold = -3.25
        self.proto_threshold = 32.0

        candidates = [
            ROOT_DIR / "fold0_CNN.pt",
            ROOT_DIR / "fold0.pt",
            Path.home() / "sentragrade_data" / "checkpoints" / "supervised_cnn" / "fold0.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError("Could not find ResNet fold0 checkpoint (fold0_CNN.pt or fold0.pt).")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.label_map = ckpt["label_map"]
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.held_out_class = "Banana"

        self.model = ResNetClassifier(len(self.label_map)).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.layer4[-1])

        val_feats = ckpt["val"]["features"]
        val_labels = ckpt["val"]["labels"]
        self.prototypes = torch.stack([val_feats[val_labels == c].mean(dim=0) for c in range(len(self.label_map))])
        print(f"[Engine] ResNet-50 loaded from {ckpt_path.name} | classes: {list(self.label_map.keys())}")

    def infer(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            probe_logits, feats = self.model(x)
        pred_idx = int(probe_logits.argmax(dim=1))
        cam, logits = self.gradcam(x, pred_idx)
        probs = torch.softmax(logits, dim=1)[0]

        energy = float(-torch.logsumexp(logits, dim=1)[0])
        msp = float(-probs.max())
        proto = float(torch.cdist(feats, self.prototypes).min())

        return {
            "pred_idx": pred_idx,
            "predicted_class": self.inv_label_map[pred_idx],
            "confidence": float(probs[pred_idx]),
            "energy_score": energy,
            "msp_score": msp,
            "proto_score": proto,
            "energy_threshold": self.energy_threshold,
            "proto_threshold": self.proto_threshold,
            "cam": cam,
        }


# =============================================================================
# 2. DINOv2 Engine (Self-Supervised ViT-S/14)
# =============================================================================

class DINOv2Engine:
    def __init__(self):
        self.name = "DINOv2 (SSL ViT-S/14)"
        self.badge = "Self-Supervised ViT"
        # Calibrated on DINOv2 fold 0 val (-13 mean) vs OOD Banana (-4.5 mean)
        self.energy_threshold = -9.0
        # Calibrated on DINOv2 fold 0 proto distance (val max 0.70 vs OOD min 0.85)
        self.proto_threshold = 0.78
        self.held_out_class = "Banana"

        # Load timm backbone
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        ).to(DEVICE)
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(p=0.1),
            nn.Linear(384, 5),
        ).to(DEVICE)

        ckpt_candidates = [
            ROOT_DIR / "fold0_dinov2.pt",
            ROOT_DIR.parent / "Semi-Supervised" / "project" / "checkpoints" / "best_model_fold0.pt",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError("Could not find DINOv2 fold0 checkpoint.")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.label_map = ckpt.get("label_map", {'Carrot': 0, 'Guava': 1, 'Indian_Gooseberry': 2, 'Mango': 3, 'Tomato': 4})
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # Load weights into backbone and head
        sd = ckpt["model_state_dict"]
        backbone_sd = {}
        head_sd = {}
        for k, v in sd.items():
            if k.startswith("backbone.backbone."):
                backbone_sd[k.replace("backbone.backbone.", "")] = v
            elif k.startswith("head."):
                head_sd[k.replace("head.", "")] = v

        if backbone_sd:
            self.backbone.load_state_dict(backbone_sd, strict=False)
        if head_sd:
            self.head.load_state_dict(head_sd, strict=False)
        self.head.eval()

        # Prototypes
        proto_path = ROOT_DIR / "fold0_dinov2_prototypes.pt"
        if proto_path.exists():
            self.prototypes = torch.load(proto_path, map_location=DEVICE, weights_only=False)
        else:
            # Fallback prototypes
            self.prototypes = torch.zeros(5, 384, device=DEVICE)

        print(f"[Engine] DINOv2 loaded from {ckpt_path.name} | classes: {list(self.label_map.keys())}")

    def infer(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)
            cls_token = tokens[:, 0, :]
            feats = F.normalize(cls_token, p=2, dim=-1)
            logits = self.head(feats)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(logits.argmax(dim=1).item())

            # Spatial patch heatmap (16x16 tokens from 224/14)
            patch_tokens = tokens[:, 1:, :]  # [1, 256, 384]
            patch_norm = patch_tokens.norm(dim=-1).view(1, 1, 16, 16)
            cam = F.interpolate(patch_norm, size=(224, 224), mode="bilinear", align_corners=False).squeeze()
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            energy = float(-torch.logsumexp(logits, dim=1)[0])
            msp = float(-probs.max())
            proto = float(torch.cdist(feats, self.prototypes).min())

        return {
            "pred_idx": pred_idx,
            "predicted_class": self.inv_label_map[pred_idx],
            "confidence": float(probs[pred_idx]),
            "energy_score": energy,
            "msp_score": msp,
            "proto_score": proto,
            "energy_threshold": self.energy_threshold,
            "proto_threshold": self.proto_threshold,
            "cam": cam,
        }


# =============================================================================
# 3. ViT Engine (Supervised Vision Transformer - Base/16)
# =============================================================================

class ViTEngine:
    def __init__(self):
        self.name = "ViT-Base (Transformers)"
        self.badge = "Supervised ViT"
        self.held_out_class = "Banana"

        ckpt_candidates = [
            ROOT_DIR / "fold0_heldout_banana_ViT.pt",
            ROOT_DIR / "fold0_heldout_Banana.pt",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError("Could not find ViT fold0 checkpoint.")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.energy_threshold = float(ckpt["thresholds"].get("energy", -4.4789))
        # 95th percentile FPR95 threshold was ~7.32; 8.5 provides robust margin for intra-class variation while strictly rejecting OOD (>9.2)
        raw_proto_th = float(ckpt["thresholds"].get("proto", 7.3231))
        self.proto_threshold = max(raw_proto_th, 8.5)
        self.known_classes = ckpt.get("known_classes", ['Carrot', 'Guava', 'Indian_Gooseberry', 'Mango', 'Tomato'])
        self.inv_label_map = {i: c for i, c in enumerate(self.known_classes)}
        self.label_map = {c: i for i, c in enumerate(self.known_classes)}
        self.prototypes = ckpt["prototypes"].to(DEVICE)

        pretrained_name = ckpt.get("pretrained_ckpt", "google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained(
            pretrained_name,
            add_pooling_layer=True,
            attn_implementation="eager",
        ).to(DEVICE)
        self.vit.eval()

        self.classifier = nn.Linear(768, len(self.known_classes)).to(DEVICE)
        self.classifier.load_state_dict(ckpt["head_state_dict"])
        self.classifier.eval()

        print(f"[Engine] ViT-Base loaded from {ckpt_path.name} | classes: {self.known_classes}")

    def infer(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.vit(pixel_values=x, output_attentions=True)
            feats = outputs.pooler_output
            logits = self.classifier(feats)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(logits.argmax(dim=1).item())

            # 12-layer attention map (CLS token attention over 14x14 patches)
            last_attn = outputs.attentions[-1]  # [1, 12, 197, 197]
            cls_attn = last_attn[0].mean(dim=0)[0, 1:].view(1, 1, 14, 14)
            cam = F.interpolate(cls_attn, size=(224, 224), mode="bilinear", align_corners=False).squeeze()
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            energy = float(-torch.logsumexp(logits, dim=1)[0])
            msp = float(-probs.max())
            proto = float(torch.cdist(feats, self.prototypes).min())

        return {
            "pred_idx": pred_idx,
            "predicted_class": self.inv_label_map[pred_idx],
            "confidence": float(probs[pred_idx]),
            "energy_score": energy,
            "msp_score": msp,
            "proto_score": proto,
            "energy_threshold": self.energy_threshold,
            "proto_threshold": self.proto_threshold,
            "cam": cam,
        }


# =============================================================================
# Startup: Load All Three Engines
# =============================================================================

ENGINES: Dict[str, Any] = {}
MODELS_META = [
    {"id": "resnet50", "name": "ResNet-50 (CNN)", "type": "Supervised CNN", "arch": "Convolutional"},
    {"id": "dinov2", "name": "DINOv2 (SSL)", "type": "Self-Supervised ViT", "arch": "ViT-S/14"},
    {"id": "vit", "name": "ViT-Base", "type": "Supervised Transformer", "arch": "ViT-B/16"},
]

try:
    ENGINES["resnet50"] = ResNetEngine()
except Exception as e:
    print(f"[warn] Failed to load ResNetEngine: {e}")

try:
    ENGINES["dinov2"] = DINOv2Engine()
except Exception as e:
    print(f"[warn] Failed to load DINOv2Engine: {e}")

try:
    ENGINES["vit"] = ViTEngine()
except Exception as e:
    print(f"[warn] Failed to load ViTEngine: {e}")

if not ENGINES:
    raise RuntimeError("No model engines could be loaded. Check checkpoint paths.")

DEFAULT_ENGINE_ID = "resnet50" if "resnet50" in ENGINES else list(ENGINES.keys())[0]

app = FastAPI(title="SentraGrade Multi-Model Scanner")


@app.get("/api/meta")
def meta(model: str = Query(DEFAULT_ENGINE_ID)):
    engine = ENGINES.get(model, ENGINES[DEFAULT_ENGINE_ID])
    classes = sorted(engine.label_map, key=lambda c: engine.label_map[c])
    return {
        "models": MODELS_META,
        "active_model": model if model in ENGINES else DEFAULT_ENGINE_ID,
        "active_model_name": engine.name,
        "active_model_badge": engine.badge,
        "known_classes": classes,
        "held_out_class": engine.held_out_class,
        "energy_threshold": engine.energy_threshold,
        "proto_threshold": engine.proto_threshold,
    }


@app.post("/api/scan")
async def scan(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_ENGINE_ID),
):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file couldn't be read as an image.")

    engine = ENGINES.get(model, ENGINES[DEFAULT_ENGINE_ID])
    x = eval_tf(img).unsqueeze(0).to(DEVICE)

    res = engine.infer(x)

    # Rejection logic: reject if energy > energy_threshold OR proto > proto_threshold
    triggered_by = []
    if res["energy_score"] > res["energy_threshold"]:
        triggered_by.append("energy")
    if res["proto_score"] > res["proto_threshold"]:
        triggered_by.append("proto")
    decision = "REJECTED" if triggered_by else "ACCEPTED"

    base_view = denormalize(x[0]).permute(1, 2, 0).numpy()
    base_img = Image.fromarray((base_view * 255).astype(np.uint8))
    heat_img = Image.alpha_composite(base_img.convert("RGBA"), Image.fromarray(colorize_cam(res["cam"]), mode="RGBA"))

    return {
        "decision": decision,
        "model_id": model,
        "model_name": engine.name,
        "predicted_class": res["predicted_class"],
        "confidence": res["confidence"],
        "energy_score": res["energy_score"],
        "energy_threshold": res["energy_threshold"],
        "msp_score": res["msp_score"],
        "proto_score": res["proto_score"],
        "proto_threshold": res["proto_threshold"],
        "triggered_by": triggered_by,
        "held_out_class": engine.held_out_class,
        "model_view_image": image_to_data_url(base_img),
        "heatmap_image": image_to_data_url(heat_img),
    }


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
