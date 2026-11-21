"""
SentraGrade — Time and space complexity for the three backbones.

Read-only / no training: builds each architecture with random (untrained)
weights just to measure its shape — parameter counts, FLOPs and CPU latency
don't depend on which checkpoint's weights are loaded. These are the numbers
behind the "Time and Space Complexity" slide.

Measures, per architecture, on this machine:
  - total and trainable parameter counts (with the same freezing/head setup
    used in the training notebooks: ResNet50 stem..layer2 frozen with a
    5-class head; ViT-B/16 backbone frozen with a linear head; DINOv2
    ViT-S/14 fully unfrozen, i.e. the fine-tuned stage)
  - GFLOPs for one 224x224 image (batch size 1)
  - FP32 weight size in MB
  - CPU latency (mean of 30 forward passes, after 5 warm-up passes)

Usage:
    python analysis/model_complexity.py
"""

import time

import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import resnet50

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def gflops(model):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def latency_ms(model, n=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000


rows = []

# 1) ResNet50, 5-class head, stem..layer2 frozen (matches
#    notebooks/02_supervised_cnn_baseline.ipynb's ResNetClassifier(freeze_until="layer2"))
resnet = resnet50(weights=None)
resnet.fc = nn.Linear(2048, 5)
for module in [resnet.conv1, resnet.bn1, resnet.layer1, resnet.layer2]:
    for p in module.parameters():
        p.requires_grad = False
rows.append(("ResNet50 (5-class head)", count_params(resnet), count_params(resnet, True), gflops(resnet), latency_ms(resnet), 2048))

# 2) ViT-B/16, frozen backbone + linear head (matches notebooks/ViT_Model.ipynb's
#    ViTBackbone with FREEZE_BACKBONE=True; timm's vit_base_patch16_224 is
#    architecturally identical to google/vit-base-patch16-224-in21k)
vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=5)
for name, p in vit.named_parameters():
    if not name.startswith("head"):
        p.requires_grad = False
rows.append(("ViT-B/16 (5-class head)", count_params(vit), count_params(vit, True), gflops(vit), latency_ms(vit), 768))


# 3) DINOv2 ViT-S/14 backbone + LayerNorm/Dropout/Linear head, fully
#    unfrozen (matches notebooks/SentraGrade_DINOv2_Notebook.ipynb's FineTunedClassifier)
class DINOv2WithHead(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224)
        self.head = nn.Sequential(nn.LayerNorm(384), nn.Dropout(0.1), nn.Linear(384, num_classes))

    def forward(self, z):
        return self.head(self.backbone(z))


dinov2 = DINOv2WithHead()
rows.append(("DINOv2 ViT-S/14 (fine-tuned)", count_params(dinov2), count_params(dinov2, True), gflops(dinov2), latency_ms(dinov2), 384))
probe_trainable = sum(p.numel() for p in dinov2.head.parameters())

print(f"DINOv2 linear-probe trainable params (head only, backbone frozen): {probe_trainable:,}\n")
print(f"{'model':32s} {'total params':>14s} {'trainable':>12s} {'GFLOPs':>8s} {'FP32 MB':>8s} {'CPU ms/img':>11s} {'emb dim':>7s}")
for name, total, trainable, flops, ms, emb in rows:
    print(f"{name:32s} {total:14,d} {trainable:12,d} {flops:8.2f} {total * 4 / 1e6:8.1f} {ms:11.1f} {emb:7d}")

print(f"\nThreads: {torch.get_num_threads()} | device: CPU | batch 1 | 224x224 | FLOPs = 2 x MACs (torch.utils.flop_counter)")
print(f"ViT-B/16 tokens: {(224 // 16) ** 2 + 1} | DINOv2 ViT-S/14 tokens: {(224 // 14) ** 2 + 1}")
