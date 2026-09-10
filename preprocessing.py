"""
SentraGrade — Step 1: Preprocessing
Run this LOCALLY on your Mac (CPU-bound, needs the raw TR-6 folder on disk).

What it does:
  1. Walks the TR-6 folder and builds one manifest of every sRGB image with
     its class label, source bucket, and parsed capture timestamp.
  2. De-duplicates: TR-6 stores every capture under BOTH `Normal/<class>/`
     AND `Classified/<class>/{Not_spoiled,Spoiled}/` — the per-class counts
     in the project report are ~exactly half/half (e.g. Tomato: 2244 Normal
     + 2244 Classified), which is a strong signal these are the same photos
     filed twice, not independent images. We check this by filename
     collision and keep only one copy per (class, filename).
  3. Builds a leakage-safe grouping key so the same physical fruit's photos
     never land on both sides of a train/val split (see NOTE below).
  4. Builds the 6 leave-one-class-out folds (train / val / ood manifests).
  5. Resizes every kept image (shorter side -> IMG_SIZE) and caches it to
     OUTPUT_DIR/resized/, then tells you what to zip and upload to Drive.

NOTE on leakage grouping — read this before trusting the split:
  Filenames are `YYYYMMDD_HHMMSS.jpg` — there is no explicit specimen ID.
  Each fruit was shot ~3 sessions/day (bursts of ~17 photos seconds apart)
  across its whole decomposition period. We group by (class, calendar day)
  and never split a day across train/val. This kills the most obvious
  leakage (near-duplicate burst photos landing on both sides), but it is
  NOT a true specimen-level split — if one physical fruit was tracked
  across many days, different days of that same fruit can still end up on
  both sides. Given the data has no specimen ID, day-level grouping is the
  most defensible thing we can do automatically. If you want to be
  stricter, set GROUP_BY = "folder" below (whole leaf folder = one
  specimen) — but check EDA output first, because for small classes like
  Banana that may leave too few groups to split at all.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("TR-6").expanduser()   # <-- set to your actual path
OUTPUT_DIR = Path("~/sentragrade_data").expanduser()    # manifests + resized cache go here
IMG_SIZE = 256          # shorter side, in pixels, after resize (224 crop happens at train time)
VAL_FRACTION = 0.2      # fraction of groups (days) held out for validation, per known class
RANDOM_SEED = 42
GROUP_BY = "day"        # "day" (recommended default) or "folder" (stricter, see note above)

CLASSES = ["Banana", "Carrot", "Guava", "Indian_Gooseberry", "Mango", "Tomato"]

FNAME_RE = re.compile(r"^(\d{8})_(\d{6})\.jpg$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 1. Build manifest
# ---------------------------------------------------------------------------

def find_srgb_dirs(dataset_root: Path):
    """Yield (class_name, source, spoil_status, sRGB_images_dir) for every leaf."""
    normal_root = dataset_root / "Normal"
    for cls in CLASSES:
        d = normal_root / cls / "sRGB_images"
        if d.is_dir():
            yield cls, "Normal", None, d

    classified_root = dataset_root / "Classified"
    for cls in CLASSES:
        for spoil in ["Not_spoiled", "Spoiled"]:
            d = classified_root / cls / spoil / "sRGB_images"
            if d.is_dir():
                yield cls, "Classified", spoil, d


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for cls, source, spoil, d in find_srgb_dirs(dataset_root):
        for fp in sorted(d.glob("*.jpg")):
            m = FNAME_RE.match(fp.name)
            if not m:
                print(f"  [skip] unrecognized filename pattern: {fp}")
                continue
            date_str, time_str = m.groups()
            rows.append(
                {
                    "filepath": str(fp),
                    "filename": fp.name,
                    "class_name": cls,
                    "source": source,
                    "spoil_status": spoil,
                    "date": date_str,
                    "time": time_str,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No sRGB images found under {dataset_root}. Check DATASET_ROOT."
        )
    return df


def dedupe_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Same (class, filename) appearing under both Normal/ and Classified/ is
    almost certainly the same physical photo filed twice. Keep the
    Classified copy when both exist (it carries the extra spoilage label,
    unused here but free), else keep whatever's there."""
    before = len(df)
    df = df.sort_values("source", ascending=False)  # "Normal" < "Classified" alphabetically desc puts Classified first
    df = df.drop_duplicates(subset=["class_name", "filename"], keep="first")
    after = len(df)
    print(f"Deduplication: {before} -> {after} images ({before - after} duplicate filings removed)")
    return df.reset_index(drop=True)


def add_group_id(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    if group_by == "day":
        df["group_id"] = df["class_name"] + "_" + df["date"]
    elif group_by == "folder":
        df["group_id"] = df["class_name"] + "_" + df["source"] + "_" + df["spoil_status"].fillna("na")
    else:
        raise ValueError(f"Unknown GROUP_BY: {group_by}")
    return df


def print_eda(df: pd.DataFrame):
    print("\n=== Class counts (after dedup) ===")
    print(df["class_name"].value_counts())
    print("\n=== Distinct groups per class (GROUP_BY={}) ===".format(GROUP_BY))
    print(df.groupby("class_name")["group_id"].nunique())
    print("\nIf any class shows fewer than ~5 groups, the train/val split below")
    print("may be too coarse for that class — consider GROUP_BY='day' if you")
    print("were using 'folder', or accept a larger VAL_FRACTION variance.\n")


# ---------------------------------------------------------------------------
# 2. Leave-one-class-out folds
# ---------------------------------------------------------------------------

def split_class_groups(class_df: pd.DataFrame, class_name: str):
    """Split one class's groups into train/val. Plain GroupShuffleSplit over
    *all* known classes pooled together can, by chance, put every one of a
    class's groups on the same side — e.g. a class with few groups (like
    Banana) could end up with zero validation images, or worse, zero
    training images for that fold. Splitting per class guarantees every
    known class is represented on both sides."""
    n_groups = class_df["group_id"].nunique()
    if n_groups < 2:
        print(
            f"  [warn] class '{class_name}' has only {n_groups} group(s) "
            f"(GROUP_BY={GROUP_BY}) — can't split without leakage, putting all "
            f"of it in train and none in val for this fold."
        )
        return class_df, class_df.iloc[0:0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(class_df, groups=class_df["group_id"]))
    return class_df.iloc[train_idx], class_df.iloc[val_idx]


def build_folds(df: pd.DataFrame, output_dir: Path):
    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for fold_idx, held_out_class in enumerate(CLASSES):
        known_df = df[df["class_name"] != held_out_class].reset_index(drop=True)
        ood_df = df[df["class_name"] == held_out_class].reset_index(drop=True)

        train_parts, val_parts = [], []
        for cls in sorted(known_df["class_name"].unique()):
            cls_df = known_df[known_df["class_name"] == cls]
            tr, va = split_class_groups(cls_df, cls)
            train_parts.append(tr)
            val_parts.append(va)
        train_df = pd.concat(train_parts).reset_index(drop=True)
        val_df = pd.concat(val_parts).reset_index(drop=True)

        # sanity: no group leakage, and every known class present on both sides
        overlap = set(train_df["group_id"]) & set(val_df["group_id"])
        assert not overlap, f"Fold {fold_idx}: group leakage detected: {overlap}"
        missing_from_train = set(known_df["class_name"]) - set(train_df["class_name"])
        assert not missing_from_train, f"Fold {fold_idx}: class(es) missing from train: {missing_from_train}"

        known_classes = sorted(known_df["class_name"].unique())
        label_map = {c: i for i, c in enumerate(known_classes)}

        train_df.to_csv(fold_dir / f"fold{fold_idx}_train.csv", index=False)
        val_df.to_csv(fold_dir / f"fold{fold_idx}_val.csv", index=False)
        ood_df.to_csv(fold_dir / f"fold{fold_idx}_ood.csv", index=False)
        with open(fold_dir / f"fold{fold_idx}_label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        summary.append(
            {
                "fold": fold_idx,
                "held_out_class": held_out_class,
                "train_n": len(train_df),
                "val_n": len(val_df),
                "ood_n": len(ood_df),
            }
        )
        print(
            f"Fold {fold_idx} (held out: {held_out_class:20s}) "
            f"train={len(train_df):5d}  val={len(val_df):5d}  ood={len(ood_df):5d}"
        )

    pd.DataFrame(summary).to_csv(fold_dir / "fold_summary.csv", index=False)
    print(f"\nFold manifests written to {fold_dir}")


# ---------------------------------------------------------------------------
# 3. Resize + cache
# ---------------------------------------------------------------------------

def resize_and_cache(df: pd.DataFrame, output_dir: Path, img_size: int) -> pd.DataFrame:
    cache_root = output_dir / "resized"
    df = df.copy()
    new_paths = []
    n_ok, n_fail = 0, 0
    for row in df.itertuples():
        out_dir = cache_root / row.class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / row.filename
        new_paths.append(str(out_path))
        if out_path.exists():
            n_ok += 1
            continue
        try:
            with Image.open(row.filepath) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w < h:
                    new_w, new_h = img_size, int(h * img_size / w)
                else:
                    new_w, new_h = int(w * img_size / h), img_size
                im = im.resize((new_w, new_h), Image.BILINEAR)
                im.save(out_path, "JPEG", quality=90)
            n_ok += 1
        except Exception as e:
            print(f"  [fail] {row.filepath}: {e}")
            n_fail += 1
    df["resized_path"] = new_paths
    print(f"\nResize cache: {n_ok} ok, {n_fail} failed, written to {cache_root}")
    return df


def rewrite_fold_paths(output_dir: Path, path_lookup: dict):
    """Add a resized_path column to each fold's train/val/ood CSV using the
    filename as key. Named explicitly (not globbed) so this doesn't also
    pick up fold_summary.csv, which has no class_name column."""
    fold_dir = output_dir / "folds"
    for fold_idx in range(len(CLASSES)):
        for split in ["train", "val", "ood"]:
            csv_path = fold_dir / f"fold{fold_idx}_{split}.csv"
            if not csv_path.exists():
                continue
            fdf = pd.read_csv(csv_path)
            fdf["resized_path"] = (fdf["class_name"] + "/" + fdf["filename"]).map(
                lambda k: path_lookup.get(k)
            )
            fdf.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning {DATASET_ROOT} ...")
    df = build_manifest(DATASET_ROOT)
    df = dedupe_manifest(df)
    df = add_group_id(df, GROUP_BY)
    print_eda(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "manifest_full.csv", index=False)

    build_folds(df, OUTPUT_DIR)

    df = resize_and_cache(df, OUTPUT_DIR, IMG_SIZE)
    path_lookup = {f"{r.class_name}/{r.filename}": r.resized_path for r in df.itertuples()}
    rewrite_fold_paths(OUTPUT_DIR, path_lookup)

    print("\nDone. Next step:")
    print(f"  cd {OUTPUT_DIR}")
    print("  zip -r sentragrade_data.zip resized folds manifest_full.csv")
    print("  # upload sentragrade_data.zip to Google Drive, then open the Colab notebook")


if __name__ == "__main__":
    main()