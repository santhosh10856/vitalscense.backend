"""
train_image_model.py — VitalSense AI · Module 3 (Image Analysis)
=================================================================
Transfer-learning pipeline using DenseNet121 pretrained on ImageNet.

TWO-PHASE TRAINING STRATEGY (key concept for viva):
────────────────────────────────────────────────────
Phase 1 — Feature Extraction (all base layers FROZEN):
    The DenseNet121 base already learned rich feature representations from
    1.2 million ImageNet images (edges, textures, shapes, object parts).
    We FREEZE it so those weights are not corrupted by our small medical
    dataset during early training. Only the new classification HEAD is
    trained. This is fast and protects the pretrained knowledge.

Phase 2 — Fine-Tuning (last N base layers UNFROZEN):
    After the head has learned to use the base's features, we UNFREEZE
    the last N layers of DenseNet121 and retrain at a 10× smaller learning
    rate. This allows the top blocks of DenseNet (which encode high-level,
    ImageNet-specific semantics) to adapt to medical image patterns without
    destroying the stable low-level features in earlier blocks.

Why DenseNet121?
    Dense connections (each layer receives feature maps from all previous
    layers) promote feature reuse, reduce vanishing gradients, and produce
    compact models. It was the backbone of CheXNet, the landmark chest X-ray
    classification model (Rajpurkar et al., 2017), making it a natural choice
    for medical imaging.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — MUST be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402 — pyplot must follow matplotlib.use()
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — override with environment variables for experimentation
# ---------------------------------------------------------------------------
DEFAULT_DROPOUT = float(os.environ.get("IMG_DROPOUT", "0.5"))
DEFAULT_DENSE_UNITS = int(os.environ.get("IMG_DENSE_UNITS", "256"))
DEFAULT_UNFREEZE_N = int(os.environ.get("IMG_UNFREEZE_N", "30"))
DEFAULT_FINETUNE_LR = float(os.environ.get("IMG_FINETUNE_LR", "1e-5"))
DEFAULT_PHASE1_LR = float(os.environ.get("IMG_PHASE1_LR", "1e-4"))


# ===========================================================================
# Model construction
# ===========================================================================

def build_model(
    num_classes: int,
    img_size: tuple[int, int] = (224, 224),
) -> Model:
    """
    Build the transfer-learning model.

    Architecture:
        Input (224×224×3)
          └─ DenseNet121 (ImageNet, include_top=False) ← FROZEN in Phase 1
               └─ GlobalAveragePooling2D               ← squashes spatial dims
                    └─ Dense(256, relu) + Dropout(0.5) ← classification head
                         └─ Dense(num_classes, activation) ← output

    Output activation:
        sigmoid  — for 2-class binary classification
        softmax  — for N-class multi-class classification

    Parameters
    ----------
    num_classes : int
        Number of output classes detected from the dataset folders.
    img_size : tuple[int, int]
        (height, width) — must match training generator target_size.

    Returns
    -------
    tf.keras.Model (keras.Model)
        Compiled-ready model (not yet compiled — caller calls compile).
    """
    input_shape = (*img_size, 3)  # height × width × RGB channels

    # ── Base model: DenseNet121 pretrained on ImageNet ────────────────────
    # include_top=False removes the original 1000-class head.
    # weights="imagenet" downloads pretrained weights automatically on
    # first call (cached in ~/.keras/models/).
    base_model = DenseNet121(
        input_shape=input_shape,
        include_top=False,       # remove ImageNet classification head
        weights="imagenet",      # use pretrained ImageNet weights
        pooling=None,            # we add our own pooling below
    )

    # ── PHASE 1: Freeze all base model layers ─────────────────────────────
    # Rationale: We want to train only the new head first. If we let gradients
    # flow through the entire base immediately, the high learning rate will
    # corrupt the carefully learned ImageNet representations.
    base_model.trainable = False
    logger.info(
        "Base DenseNet121 FROZEN: %d layers, %d trainable params (should be 0).",
        len(base_model.layers),
        sum(1 for layer in base_model.layers if layer.trainable),
    )

    # ── Classification head ────────────────────────────────────────────────
    inputs = keras.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)
    # training=False: keep BatchNorm in inference mode even during Phase 1
    # training. This prevents the small medical dataset from updating the
    # ImageNet BN running statistics inside the frozen base.

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # Squashes the (7×7×1024) feature map → (1024,) vector by averaging
    # spatial positions. More robust to positional shift than Flatten().

    x = layers.Dense(DEFAULT_DENSE_UNITS, activation="relu", name="fc1")(x)
    x = layers.Dropout(DEFAULT_DROPOUT, name="dropout1")(x)
    # Dropout(0.5): randomly zeroes 50% of units during training to prevent
    # the small head from memorising individual training images.

    # Output activation depends on number of classes:
    if num_classes == 2:
        # Binary: single sigmoid neuron; threshold at 0.5.
        # class_labels[1] is the "positive" class (e.g., PNEUMONIA).
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        # Multi-class: softmax over num_classes neurons.
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="VitalSense_ImageModel")

    logger.info(
        "Model built: %d total params, %d trainable (head only).",
        model.count_params(),
        sum(np.prod(v.shape) for v in model.trainable_variables),
    )
    return model


# ===========================================================================
# Phase 1 — Training (frozen base)
# ===========================================================================

def train_model(
    model: Model,
    train_gen,
    val_gen,
    epochs: int = 10,
    model_dir: str = "./model_store",
) -> dict:
    """
    Phase 1 training: train only the classification head with frozen base.

    Callbacks:
        EarlyStopping  — stops when val_accuracy plateaus for 5 epochs,
                         restores best weights automatically.
        ModelCheckpoint — saves checkpoint only when val_accuracy improves.

    Parameters
    ----------
    model     : Keras model from build_model().
    train_gen : Training ImageDataGenerator (with augmentation).
    val_gen   : Validation ImageDataGenerator (rescale only).
    epochs    : Maximum number of Phase 1 epochs. Default: 10.
    model_dir : Directory to save the best checkpoint. Default: ./model_store.

    Returns
    -------
    dict
        History dict with keys: "accuracy", "val_accuracy", "loss",
        "val_loss", plus AUC metrics if applicable.
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(model_dir_path / "best_checkpoint.keras")

    num_classes = _detect_num_classes(model)
    loss_fn, metrics = _get_loss_and_metrics(num_classes)

    # Compile with Adam at phase-1 learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_PHASE1_LR),
        loss=loss_fn,
        metrics=metrics,
    )
    logger.info("Phase 1 compiled: lr=%.0e, loss=%s", DEFAULT_PHASE1_LR, loss_fn)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,               # wait 5 epochs with no improvement
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,      # only overwrite if val_accuracy improves
            verbose=1,
        ),
    ]

    logger.info("Starting Phase 1 training (up to %d epochs)...", epochs)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    logger.info(
        "Phase 1 complete. Best val_accuracy: %.4f",
        max(history_dict.get("val_accuracy", [0])),
    )
    return history_dict


# ===========================================================================
# Phase 2 — Fine-tuning (partial unfreeze)
# ===========================================================================

def fine_tune_model(
    model: Model,
    train_gen,
    val_gen,
    unfreeze_n: int = DEFAULT_UNFREEZE_N,
    epochs: int = 10,
    model_dir: str = "./model_store",
) -> dict:
    """
    Phase 2 fine-tuning: unfreeze the last ``unfreeze_n`` layers of the
    DenseNet121 base and continue training at a much lower learning rate.

    WHY A LOWER LR?
        The head has already stabilised. If we use the same Phase 1 LR,
        large weight updates would destroy the carefully learned DenseNet
        representations. A 10× smaller LR (1e-5) allows gentle adaptation.

    WHY ONLY THE LAST N LAYERS?
        Early DenseNet layers detect low-level features (edges, corners)
        that are universal across image domains — there is no benefit to
        retraining them. Only the final blocks encode high-level semantics
        (ImageNet object parts) that may need to shift toward medical
        image patterns.

    Parameters
    ----------
    model      : Keras model (after Phase 1 training).
    train_gen  : Training generator.
    val_gen    : Validation generator.
    unfreeze_n : Number of layers to unfreeze from the END of the base.
                 Default: 30 (last ~2 dense blocks of DenseNet121).
    epochs     : Maximum fine-tune epochs. Default: 10.
    model_dir  : Directory to save the fine-tuned checkpoint.

    Returns
    -------
    dict
        History dict from fine-tune training (same format as train_model).
    """
    # ── Locate the DenseNet121 base sub-model ─────────────────────────────
    base_model = _get_base_model(model)
    if base_model is None:
        logger.warning("Could not locate DenseNet121 base — skipping fine-tune.")
        return {}

    total_layers = len(base_model.layers)

    # Unfreeze last `unfreeze_n` layers; keep everything before frozen.
    # This is the precise surgical unfreeze described in the training strategy.
    base_model.trainable = True
    for layer in base_model.layers[: total_layers - unfreeze_n]:
        layer.trainable = False   # refreeze early layers

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    logger.info(
        "Phase 2: unfroze last %d of %d base layers (%d now trainable).",
        unfreeze_n,
        total_layers,
        trainable_count,
    )

    num_classes = _detect_num_classes(model)
    loss_fn, metrics = _get_loss_and_metrics(num_classes)

    # Re-compile with 10× smaller LR to protect pretrained weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_FINETUNE_LR),
        loss=loss_fn,
        metrics=metrics,
    )
    logger.info("Phase 2 compiled: lr=%.0e", DEFAULT_FINETUNE_LR)

    model_dir_path = Path(model_dir)
    ft_checkpoint = str(model_dir_path / "best_finetune_checkpoint.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ft_checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Starting Phase 2 fine-tuning (up to %d epochs)...", epochs)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    logger.info(
        "Phase 2 complete. Best val_accuracy: %.4f",
        max(history_dict.get("val_accuracy", [0])),
    )
    return history_dict


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_model(
    model: Model,
    test_gen,
    model_dir: str = "./model_store",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate the trained model on the test generator and save metrics.

    Produces:
        • Classification report (precision, recall, F1 per class)
        • Confusion matrix (printed)
        • ROC-AUC score + ROC curve PNG saved to model_dir/

    Parameters
    ----------
    model       : Trained Keras model.
    test_gen    : Test ImageDataGenerator (shuffle=False, rescale only).
    model_dir   : Directory to save plots.
    class_names : Optional list of class names for readable report.

    Returns
    -------
    dict
        {"accuracy", "precision", "recall", "f1", "roc_auc",
         "confusion_matrix", "roc_curve_path"}
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    num_classes = _detect_num_classes(model)

    logger.info("Running inference on test set (%d batches)...", len(test_gen))
    # Reset generator so predictions start from the first image
    test_gen.reset()

    # ── Get predictions ───────────────────────────────────────────────────
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes  # ground-truth integer labels (in order)

    if num_classes == 2:
        # Binary: single sigmoid output → squeeze to 1D probability of class 1
        y_pred_proba_flat = y_pred_proba.squeeze()
        y_pred_labels = (y_pred_proba_flat >= 0.5).astype(int)
        y_score = y_pred_proba_flat           # for ROC-AUC
        roc_auc = roc_auc_score(y_true, y_score)
    else:
        # Multi-class: softmax → argmax for label; OvR AUC
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        y_score = y_pred_proba
        try:
            roc_auc = roc_auc_score(y_true, y_score, multi_class="ovr")
        except ValueError as e:
            logger.warning("ROC-AUC could not be computed: %s", e)
            roc_auc = None

    # ── Classification report ──────────────────────────────────────────────
    names = class_names or [str(i) for i in range(num_classes)]
    report_str = classification_report(y_true, y_pred_labels, target_names=names)
    print("\n-- Classification Report -----------------------------------------")
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred_labels, target_names=names, output_dict=True
    )

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred_labels)
    print("\n-- Confusion Matrix ----------------------------------------------")
    print(f"Classes: {names}")
    print(cm)

    # Accuracy from confusion matrix (sum of diagonal / total)
    accuracy = float(np.trace(cm) / np.sum(cm))

    # ── ROC curve plot ─────────────────────────────────────────────────────
    roc_curve_path = None
    try:
        roc_curve_path = _save_roc_plot(
            y_true, y_score, names, num_classes, model_dir_path
        )
    except Exception as exc:
        logger.warning("Failed to generate ROC curve plot: %s", exc)

    if roc_auc is not None:
        print(f"\n-- ROC-AUC: {roc_auc:.4f} -----------------------------------")

    metrics_out = {
        "accuracy": round(accuracy, 4),
        "precision": round(report_dict.get("weighted avg", {}).get("precision", 0), 4),
        "recall": round(report_dict.get("weighted avg", {}).get("recall", 0), 4),
        "f1": round(report_dict.get("weighted avg", {}).get("f1-score", 0), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "roc_curve_path": str(roc_curve_path) if roc_curve_path else None,
        "per_class": {
            cls: {
                k: round(v, 4)
                for k, v in report_dict.get(cls, {}).items()
                if isinstance(v, float)
            }
            for cls in names
        },
    }

    logger.info("Evaluation complete: accuracy=%.4f, AUC=%s", accuracy, roc_auc)
    return metrics_out


# ===========================================================================
# Save / Load helpers
# ===========================================================================

def save_model_artifacts(
    model: Model,
    class_names: list[str],
    model_dir: str = "./model_store",
) -> tuple[str, str]:
    """
    Save the trained model and class-label mapping to disk.

    Saves:
        <model_dir>/image_model.keras — full model (architecture + weights)
        <model_dir>/class_labels.json — {index: class_name} mapping

    The .keras format (Keras 3 native) is preferred over legacy .h5 because
    it stores the full model graph reliably, including custom layers.

    Returns
    -------
    tuple[str, str]
        (model_path, labels_path)
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = model_dir_path / "image_model.keras"
    model.save(str(model_path))
    logger.info("Model saved: %s", model_path)

    # ── Save class label mapping ───────────────────────────────────────────
    # Stored as {str(index): class_name} so JSON is portable.
    labels_map = {str(i): name for i, name in enumerate(class_names)}
    labels_path = model_dir_path / "class_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, indent=2)
    logger.info("Class labels saved: %s  —  %s", labels_path, labels_map)

    return str(model_path), str(labels_path)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _get_loss_and_metrics(num_classes: int) -> tuple[str, list]:
    """
    Select the appropriate loss function and evaluation metrics.

    Binary (2 classes):
        Loss    — binary_crossentropy (single sigmoid output)
        Metrics — accuracy + AUC

    Multi-class (>2 classes):
        Loss    — categorical_crossentropy (softmax over N classes)
        Metrics — accuracy + AUC (one-vs-rest)
    """
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = [
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ]
    else:
        loss = "categorical_crossentropy"
        metrics = [
            "accuracy",
            keras.metrics.AUC(name="auc", multi_label=False),
        ]
    return loss, metrics


def _detect_num_classes(model: Model) -> int:
    """
    Infer the number of output classes from the model's output layer shape.
    Binary models have output shape (None, 1); multi-class is (None, N).
    """
    out_shape = model.output_shape
    last_dim = out_shape[-1]
    return 2 if last_dim == 1 else last_dim


def _get_base_model(model: Model):
    """
    Find the DenseNet121 sub-model inside the top-level functional model.
    Returns None if not found.
    """
    for layer in model.layers:
        if isinstance(layer, Model) and "densenet" in layer.name.lower():
            return layer
    return None


def _save_roc_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    num_classes: int,
    model_dir_path: Path,
) -> Path:
    """
    Generate and save ROC curve(s) as a PNG.

    Binary: single curve.
    Multi-class: one curve per class (One-vs-Rest).

    Returns the saved PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[1]} (AUC = {auc_val:.3f})")
    else:
        for i, cls in enumerate(class_names):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
            auc_val = roc_auc_score(y_true_bin, y_score[:, i])
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — VitalSense AI Image Model", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    roc_path = model_dir_path / "roc_curve.png"
    fig.savefig(str(roc_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved: %s", roc_path)
    return roc_path
