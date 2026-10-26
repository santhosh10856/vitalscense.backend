"""
train_all_models.py — VitalSense AI · Module 3
===============================================
Master training launcher for all three scan modalities:
  • X-Ray   (chest_xray — already split into train/val/test)
  • MRI     (brain_mri_dataset — flat Normal/Tumar folders, auto-split)
  • CT Scan (kidney ct scan data sets — flat Cyst/Normal/Stone/Tumor, auto-split)

Usage:
    # Train all three models (run from project root)
    python module3/train_all_models.py

    # Train only specific models
    python module3/train_all_models.py --models xray mri
    python module3/train_all_models.py --models ct

    # Custom epochs
    python module3/train_all_models.py --phase1-epochs 15 --phase2-epochs 10

    # Skip fine-tuning (Phase 2)
    python module3/train_all_models.py --no-finetune

How it works:
    1. X-Ray  -> uses existing train/val/test directory structure directly.
    2. MRI    -> auto-splits the flat Normal/Tumar folders (70/15/15).
    3. CT     -> auto-splits the flat Cyst/Normal/Stone/Tumor folders (70/15/15)
       from 'kidney ct scan data sets/'.
    4. Each model saved to model_store/<scan_type>_model/
    5. A training_summary.json is written at the end with all metrics.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

# -- Suppress TensorFlow noise BEFORE importing TF --------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# -- Make sure module3 is on the path ---------------------------------------
MODULE3_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(MODULE3_DIR))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train_image_model import (
    build_model,
    evaluate_model,
    fine_tune_model,
    save_model_artifacts,
    train_model,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = MODULE3_DIR.parent

XRAY_DATASET = PROJECT_ROOT / "x ray datasets" / "chest_xray"
MRI_DATASET  = PROJECT_ROOT / "brain_mri_dataset"
CT_DATASET   = PROJECT_ROOT / "kidney ct scan data sets"

MODEL_STORE  = MODULE3_DIR / "model_store"

# Temporary split directories (auto-generated, cleaned up after training)
SPLIT_TMP    = MODULE3_DIR / "_tmp_splits"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# Class-label remaps: folder name -> display name saved in class_labels.json
# Maps the actual dataset folder names → display labels saved in class_labels.json.
# Dataset layout: brain_mri_dataset/Normal/ (no tumor) and /Tumar/ (tumor, typo in source).
MRI_CLASS_REMAP = {"Normal": "NORMAL", "Tumar": "TUMOR"}
CT_CLASS_REMAP  = {}   # CT folder names are already descriptive


# ===========================================================================
# Helper: auto-split a flat dataset
# ===========================================================================

def split_flat_dataset(
    src_dir,
    dest_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    class_remap=None,
    seed=SEED,
):
    """
    Copy images from a flat class-folder layout into train/val/test splits.

    Source layout:
        src_dir/
            class_A/   <- all images for class A
            class_B/   <- all images for class B
            ...

    Destination layout:
        dest_dir/
            train/class_A/  class_B/  ...
            val/  class_A/  class_B/  ...
            test/ class_A/  class_B/  ...
    """
    src_dir  = Path(src_dir)
    dest_dir = Path(dest_dir)
    remap    = class_remap or {}
    rng      = random.Random(seed)

    # Wipe old split directory so we always start fresh
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    class_dirs = sorted(
        [d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )
    if not class_dirs:
        raise FileNotFoundError(f"No class sub-folders found in {src_dir}")

    for cls_dir in class_dirs:
        display_name = remap.get(cls_dir.name, cls_dir.name)

        images = sorted([
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        if not images:
            logger.warning("Class '%s' has no images -- skipping.", cls_dir.name)
            continue

        rng.shuffle(images)
        n       = len(images)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_test  = max(1, n - n_train - n_val)
        # Clamp in case rounding pushed us over
        while n_train + n_val + n_test > n:
            n_test -= 1

        splits = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val: n_train + n_val + n_test],
        }

        for split_name, files in splits.items():
            target_dir = dest_dir / split_name / display_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                shutil.copy2(img_path, target_dir / img_path.name)
            logger.info(
                "  %s / %-10s -> %d images", split_name, display_name, len(files)
            )


# ===========================================================================
# Helper: build Keras generators from a split directory
# ===========================================================================

def make_generators(split_dir, batch_size=BATCH_SIZE):
    """
    Build augmented train / plain val / plain test ImageDataGenerators.

    Augmentation (train only):
        Horizontal flip, rotation +-15 deg, width/height shift 10%,
        zoom 10%, brightness 80-120%.

    Returns (train_gen, val_gen, test_gen, class_names)
    """
    split_dir = Path(split_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        brightness_range=[0.80, 1.20],
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # First pass with binary to detect class count
    probe = eval_datagen.flow_from_directory(
        str(split_dir / "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        seed=SEED,
    )
    num_classes = len(probe.class_indices)
    class_mode  = "binary" if num_classes == 2 else "categorical"
    class_names = [k for k, _ in sorted(probe.class_indices.items(), key=lambda x: x[1])]

    common = dict(target_size=IMG_SIZE, batch_size=batch_size,
                  class_mode=class_mode, seed=SEED)

    train_gen = train_datagen.flow_from_directory(
        str(split_dir / "train"), shuffle=True,  **common
    )
    val_gen = eval_datagen.flow_from_directory(
        str(split_dir / "val"),   shuffle=False, **common
    )
    test_gen = eval_datagen.flow_from_directory(
        str(split_dir / "test"),  shuffle=False, **common
    )

    logger.info(
        "  Generators ready | classes=%s | train=%d  val=%d  test=%d",
        class_names, train_gen.samples, val_gen.samples, test_gen.samples,
    )
    return train_gen, val_gen, test_gen, class_names


# ===========================================================================
# Per-model training routines
# ===========================================================================

def train_xray(args):
    """Train the Chest X-Ray model (NORMAL / PNEUMONIA)."""
    logger.info("=" * 60)
    logger.info("  TRAINING: CHEST X-RAY MODEL")
    logger.info("=" * 60)

    model_dir = str(MODEL_STORE / "xray_model")
    train_gen, val_gen, test_gen, class_names = make_generators(XRAY_DATASET, args.batch_size)

    model = build_model(len(class_names), IMG_SIZE)
    h1    = train_model(model, train_gen, val_gen, args.phase1_epochs, model_dir)

    h2 = {}
    if not args.no_finetune:
        h2 = fine_tune_model(model, train_gen, val_gen,
                             epochs=args.phase2_epochs, model_dir=model_dir)

    metrics = evaluate_model(model, test_gen, model_dir, class_names)
    save_model_artifacts(model, class_names, model_dir)

    logger.info("  X-RAY done -> accuracy=%.4f  AUC=%s",
                metrics["accuracy"], metrics["roc_auc"])
    return {"scan_type": "xray", "classes": class_names,
            "metrics": metrics, "phase1_history": h1, "phase2_history": h2}


def train_mri(args):
    """
    Train the Brain MRI model (NORMAL / TUMOR).
    Source: brain_mri_dataset/Normal (no tumor) and /Tumar (tumor — note: folder
    name is misspelled in the source dataset but remapped correctly via
    MRI_CLASS_REMAP: Normal → NORMAL, Tumar → TUMOR).
    Auto-splits 70/15/15.
    """
    logger.info("=" * 60)
    logger.info("  TRAINING: BRAIN MRI MODEL")
    logger.info("=" * 60)

    split_dir = SPLIT_TMP / "mri"
    model_dir = str(MODEL_STORE / "mri_model")

    logger.info("  Auto-splitting MRI dataset (70 / 15 / 15)...")
    split_flat_dataset(MRI_DATASET, split_dir, class_remap=MRI_CLASS_REMAP)

    train_gen, val_gen, test_gen, class_names = make_generators(split_dir, args.batch_size)

    model = build_model(len(class_names), IMG_SIZE)
    h1    = train_model(model, train_gen, val_gen, args.phase1_epochs, model_dir)

    h2 = {}
    if not args.no_finetune:
        h2 = fine_tune_model(model, train_gen, val_gen,
                             epochs=args.phase2_epochs, model_dir=model_dir)

    metrics = evaluate_model(model, test_gen, model_dir, class_names)
    save_model_artifacts(model, class_names, model_dir)

    logger.info("  MRI done -> accuracy=%.4f  AUC=%s",
                metrics["accuracy"], metrics["roc_auc"])
    return {"scan_type": "mri", "classes": class_names,
            "metrics": metrics, "phase1_history": h1, "phase2_history": h2}


def train_ct(args):
    """
    Train the CT Scan model (Cyst / Normal / Stone / Tumor) -- 4-class.
    Source: ct scan data sets/ with flat per-class folders.
    Auto-splits 70/15/15.
    """
    logger.info("=" * 60)
    logger.info("  TRAINING: CT SCAN MODEL")
    logger.info("=" * 60)

    split_dir = SPLIT_TMP / "ct"
    model_dir = str(MODEL_STORE / "ct_model")

    logger.info("  Auto-splitting CT dataset (70 / 15 / 15)...")
    split_flat_dataset(CT_DATASET, split_dir, class_remap=CT_CLASS_REMAP)

    train_gen, val_gen, test_gen, class_names = make_generators(split_dir, args.batch_size)

    model = build_model(len(class_names), IMG_SIZE)
    h1    = train_model(model, train_gen, val_gen, args.phase1_epochs, model_dir)

    h2 = {}
    if not args.no_finetune:
        h2 = fine_tune_model(model, train_gen, val_gen,
                             epochs=args.phase2_epochs, model_dir=model_dir)

    metrics = evaluate_model(model, test_gen, model_dir, class_names)
    save_model_artifacts(model, class_names, model_dir)

    logger.info("  CT done -> accuracy=%.4f  AUC=%s",
                metrics["accuracy"], metrics["roc_auc"])
    return {"scan_type": "ct", "classes": class_names,
            "metrics": metrics, "phase1_history": h1, "phase2_history": h2}


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VitalSense AI -- Train X-Ray, MRI, and CT Scan models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python module3/train_all_models.py
  python module3/train_all_models.py --models xray mri
  python module3/train_all_models.py --models ct --phase1-epochs 20
  python module3/train_all_models.py --no-finetune --batch-size 16
        """,
    )
    parser.add_argument(
        "--models", nargs="+", choices=["xray", "mri", "ct"],
        default=["xray", "mri", "ct"],
        help="Which model(s) to train. Default: all three.",
    )
    parser.add_argument("--phase1-epochs", type=int, default=10,
                        help="Max epochs for Phase 1 (frozen base). Default: 10.")
    parser.add_argument("--phase2-epochs", type=int, default=10,
                        help="Max epochs for Phase 2 fine-tuning. Default: 10.")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip Phase 2 fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size. Default: {BATCH_SIZE}.")
    parser.add_argument("--keep-splits", action="store_true",
                        help="Keep temporary split directories after training.")
    return parser.parse_args()


# ===========================================================================
# Utilities
# ===========================================================================

def _fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Validate dataset availability
    dataset_map = {
        "xray": XRAY_DATASET,
        "mri":  MRI_DATASET,
        "ct":   CT_DATASET,
    }
    for key in args.models:
        path = dataset_map[key]
        if not path.exists():
            logger.error("Dataset not found for '%s': %s", key, path)
            sys.exit(1)

    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    SPLIT_TMP.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 56)
    logger.info("  VitalSense AI -- Multi-Modal Image Training")
    logger.info("  Models    : %s", args.models)
    logger.info("  Phase 1   : %d epochs (frozen DenseNet121)", args.phase1_epochs)
    if args.no_finetune:
        logger.info("  Phase 2   : SKIPPED")
    else:
        logger.info("  Phase 2   : %d epochs (fine-tune)", args.phase2_epochs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("=" * 56)

    trainer_map = {"xray": train_xray, "mri": train_mri, "ct": train_ct}
    all_results = []
    start_wall  = time.time()

    for key in args.models:
        t0 = time.time()
        try:
            result = trainer_map[key](args)
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            result["status"] = "success"
        except Exception as exc:
            logger.exception("Training failed for '%s': %s", key, exc)
            result = {"scan_type": key, "status": "error", "error": str(exc)}
        all_results.append(result)

    # Clean up temporary splits
    if not args.keep_splits and SPLIT_TMP.exists():
        shutil.rmtree(SPLIT_TMP)
        logger.info("Temporary split directories removed.")

    total_elapsed = round(time.time() - start_wall, 1)

    # Write summary JSON
    summary = {
        "total_elapsed_seconds": total_elapsed,
        "models_trained": args.models,
        "results": all_results,
    }
    summary_path = MODEL_STORE / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    logger.info("")
    logger.info("=" * 56)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 56)
    for r in all_results:
        if r.get("status") == "success":
            m = r.get("metrics", {})
            logger.info(
                "  %-6s  acc=%.4f  AUC=%s  classes=%s",
                r["scan_type"].upper(),
                m.get("accuracy", 0),
                str(m.get("roc_auc", "N/A")),
                r.get("classes", []),
            )
        else:
            logger.info("  %-6s  FAILED: %s", r["scan_type"].upper(), r.get("error", ""))
    logger.info("  Total time : %s", _fmt_elapsed(total_elapsed))
    logger.info("  Summary    : %s", summary_path)
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
