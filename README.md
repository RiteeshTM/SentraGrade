# SentraGrade
Explainable Out-of-Distribution Detection for Foreign Object Rejection in Automated Fruit and Vegetable Grading Systems

## Repository layout

| Path | What is in it |
|---|---|
| `preprocessing.py` | Builds `~/sentragrade_data/` from TR-6: de-duplication, resizing, leave-one-class-out folds |
| `main.py` | Placeholder entry point from the `uv` project template |
| `app/` | FastAPI scanner web app (multi-model OOD detection with heatmaps); start it with `start.sh` (local) or `share.sh` |
| `notebooks/` | Model notebooks: ResNet50 baseline and OOD/Grad-CAM evaluation (`02_*`, `03_*`), ViT-B/16 (`ViT_Model.ipynb`), DINOv2 (`SentraGrade_DINOv2_Notebook.ipynb`), Keras ResNet50 (`supervised_ood_eval_gradcam_tf.ipynb`) |
| `notebooks/outputs/` | Per-fold training summaries saved by the notebooks |
| `kaggle/` | Kaggle notebooks for the leakage-corrected split, ViT/DINOv2 probes, novelty maps, post-hoc baselines and multi-seed runs, with their results (see `kaggle/README.md`) |
| `analysis/` | Scripts for the result graphs and the model-complexity table |
| `experiments/gan/` | GAN experiments and the GAN lab evaluations (notebooks, report and outputs) |
| `Test_Images/` | Sample images for trying the scanner: known produce (`valid/`) and foreign objects (`Foreign/`) |

The TR-6 dataset (`TR-6/`), the preprocessed data, model checkpoints and large Kaggle archives are not stored in the repository.

## Running the scanner

```bash
./start.sh
```
