# Kaggle experiments

Notebooks that run on Kaggle (GPU T4, Internet on) with the `sentragrade-data` dataset attached
(it must contain `spoilage_labels.csv`, also kept here).

| Notebook | What it does | Needs as input |
|---|---|---|
| `leakage_split_resnet50.ipynb` | Retrains ResNet50 on all 6 leave-one-class-out folds under the original day-grouped split and the leakage-corrected stage-matched split; MSP / energy / prototype scores, session-bootstrap CIs, fresh vs spoiled false rejects | `sentragrade-data` |
| `vit_dinov2_probe.ipynb` | Frozen ViT-B/16 and DINOv2 ViT-S/14 with linear heads on the same splits | `sentragrade-data` |
| `extra_baselines.ipynb` | Post-hoc baselines on the saved models, no retraining: MaxLogit, Mahalanobis, KNN, ReAct, ViM (plus MSP/energy/prototype as a check) | `sentragrade-data`, outputs of the leakage and ViT/DINOv2 notebooks |
| `leakage_seeds_resnet50.ipynb` | Same as the leakage notebook for extra seeds 43 and 44 (seed 42 is the main run); CSVs only | `sentragrade-data` |
| `vit_dinov2_seeds.ipynb` | Same as the ViT/DINOv2 notebook for seeds 42, 43, 44 | `sentragrade-data` |
| `novelty_maps.ipynb` | Energy-CAM, class-conditional DINOv2 patch maps, deletion/insertion faithfulness, map spread, spoiled/unknown/foreign triage, COIL-100 and DTD far-OOD | `sentragrade-data`, output of the leakage notebook, COIL-100 |

## Results

- `results/run2_stage_matched/` — the current results (stage-matched corrected split). Every number in the paper traces to these CSVs; `all_backbones_stage_matched_summary.csv` combines the three backbones; `baselines/` holds the extra post-hoc baselines; `seeds/` holds the seed 43/44 (ResNet50) and seed 42-44 (ViT/DINOv2) runs, and `seeds/summary_across_seeds_all.csv` combines all three seeds for every backbone.
- `results/run1_time_blocked/` — the first corrected run with a plain time-blocked split. Kept because it shows the spoilage confound (training mostly fresh, testing mostly spoiled). Its novelty folder uses an earlier deletion test with a flat-colour fill that was later found to be invalid; do not use those faithfulness numbers.

Large archives (`*.zip`), checkpoints (`*.pt`) and map arrays (`*.npz`) are not committed.
