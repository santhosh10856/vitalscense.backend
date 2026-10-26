"""
predict_image.py — VitalSense AI · Module 3 (Image Analysis)
=============================================================
Inference engine for the trained medical image screening model.

Includes:
    - Grad-CAM heatmap generation (identifies WHICH region drove the prediction)
    - Confidence thresholding (low-confidence predictions flagged as inconclusive)
    - DICOM file support (requires pydicom, optional)
    - Mandatory assistive disclaimer in every response

GRAD-CAM EXPLANATION (for viva):
    Grad-CAM (Gradient-weighted Class Activation Mapping) answers:
    "Which spatial regions of the input image contributed most to
    this classification decision?"

    Algorithm:
    1. Forward-pass the image through the model.
    2. Compute the gradient of the predicted class score with respect
       to the feature maps of the LAST convolutional layer.
    3. Average the gradients across spatial dimensions (global average)
       to get one importance weight per feature map channel.
    4. Multiply each channel's feature map by its importance weight
       and sum → gives a single spatial heatmap (same spatial size as
       the conv layer output, e.g., 7×7 for DenseNet121 final block).
    5. Apply ReLU (keep only positive contributions), then upsample to
       the original image size.
    6. Overlay as a transparent colour heatmap on the original image.

    This is "gradient-weighted" because we use gradients (not just
    activations) to weight the feature maps — making it class-discriminative.
    The same technique works for any CNN without architecture changes.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for Flask server threads
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

# Stub blocked TFLite DLLs BEFORE importing tensorflow.
# Windows App Control blocks several TFLite .pyd wrappers on this machine.
# tf_lite_stub installs a sys.meta_path hook that replaces them with empty
# modules so the main TF inference engine (not blocked) can load normally.
import tf_lite_stub  # noqa: F401  — side-effect only, must be before tensorflow

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Assistive disclaimer — required on every response
# ---------------------------------------------------------------------------
ASSISTIVE_DISCLAIMER = (
    "This is an AI-generated preliminary screening result based on image patterns. "
    "It is NOT a medical diagnosis. A qualified radiologist or physician must review "
    "the original image and confirm any finding before any medical decision is made."
)

# ---------------------------------------------------------------------------
# Supported image extensions (lowercase)
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DICOM_EXT = ".dcm"

# ---------------------------------------------------------------------------
# In-memory model cache — avoids reloading model on every request
# ---------------------------------------------------------------------------
_model_cache: dict = {
    "model": None,
    "class_labels": None,
    "model_dir": None,
}


# ===========================================================================
# Public API
# ===========================================================================

def predict_from_image(
    image_path: str,
    model_dir: str = "./model_store",
    conf_threshold: float = 0.60,
) -> dict:
    """
    Run inference on a single medical image and return a structured result.

    Parameters
    ----------
    image_path     : str
        Absolute or relative path to the image file
        (.jpg / .png / .bmp / .dcm if pydicom is installed).
    model_dir      : str
        Directory containing ``image_model.keras`` and ``class_labels.json``.
        Default: ``./model_store``.
    conf_threshold : float
        Confidence threshold below which ``low_confidence_warning`` is True.
        A low-confidence result should be treated as inconclusive, not
        "probably normal". Default: 0.60.

    Returns
    -------
    dict with keys:
        predicted_class       : str   — winning class name
        confidence            : float — softmax/sigmoid probability [0, 1]
        all_probabilities     : dict  — {class_name: probability} for all classes
        low_confidence_warning: bool  — True if confidence < conf_threshold
        flagged_region        : str|null — path to Grad-CAM heatmap PNG, or null
        assistive_disclaimer  : str   — mandatory disclaimer text

    Raises
    ------
    FileNotFoundError : if image_path or model artifacts do not exist.
    RuntimeError      : if model or labels file cannot be loaded.
    ValueError        : if the file type is not supported.
    """
    img_path = Path(image_path).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    ext = img_path.suffix.lower()
    if ext not in SUPPORTED_EXTS and ext != DICOM_EXT:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTS | {DICOM_EXT}}"
        )

    # ── Load model & labels (cached after first call) ─────────────────────
    model, class_labels = _load_model_and_labels(model_dir)
    num_classes = len(class_labels)
    img_size = model.input_shape[1:3]  # (height, width) from model spec

    # ── Preprocess image ───────────────────────────────────────────────────
    img_array, original_img = _load_and_preprocess(img_path, img_size)
    # img_array shape: (1, H, W, 3) — batch of one, normalised to [0,1]

    # ── Run inference ──────────────────────────────────────────────────────
    predictions = model.predict(img_array, verbose=0)

    # ── Decode predictions ─────────────────────────────────────────────────
    if num_classes == 2:
        # Binary sigmoid: single probability for class index 1
        prob_class1 = float(predictions[0][0])
        probs = {class_labels["0"]: round(1.0 - prob_class1, 4),
                 class_labels["1"]: round(prob_class1, 4)}
        predicted_idx = "1" if prob_class1 >= 0.5 else "0"
        confidence = prob_class1 if prob_class1 >= 0.5 else (1.0 - prob_class1)
    else:
        # Multi-class softmax: probability per class
        prob_vector = predictions[0]
        predicted_idx_int = int(np.argmax(prob_vector))
        predicted_idx = str(predicted_idx_int)
        confidence = float(prob_vector[predicted_idx_int])
        probs = {
            class_labels[str(i)]: round(float(p), 4)
            for i, p in enumerate(prob_vector)
        }

    predicted_class = class_labels[predicted_idx]
    low_confidence = confidence < conf_threshold

    logger.info(
        "Prediction: class='%s', confidence=%.4f, low_confidence=%s",
        predicted_class,
        confidence,
        low_confidence,
    )

    # ── Grad-CAM heatmap ───────────────────────────────────────────────────
    heatmap_path = None
    try:
        heatmap_path = _generate_gradcam(
            model=model,
            img_array=img_array,
            original_img=original_img,
            predicted_idx=int(predicted_idx),
            num_classes=num_classes,
            model_dir=model_dir,
        )
    except Exception as exc:
        logger.warning("Grad-CAM generation failed (non-fatal): %s", exc)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": probs,
        "low_confidence_warning": low_confidence,
        "flagged_region": str(heatmap_path) if heatmap_path else None,
        "assistive_disclaimer": ASSISTIVE_DISCLAIMER,
    }


# ===========================================================================
# Grad-CAM Implementation
# ===========================================================================

def _generate_gradcam(
    model: keras.Model,
    img_array: np.ndarray,
    original_img: Image.Image,
    predicted_idx: int,
    num_classes: int,
    model_dir: str,
    last_conv_layer_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate a Grad-CAM heatmap and save it as a PNG.

    The heatmap visually highlights the image regions that most strongly
    influenced the model's prediction — making the "assistive" label concrete.

    Algorithm:
        1. Build a sub-model that outputs both the last conv layer's feature
           maps AND the final classification score.
        2. Record gradients of the target class score w.r.t. the conv maps.
        3. Pool gradients spatially → per-channel importance weights.
        4. Weighted sum of feature maps → raw heatmap.
        5. ReLU + normalise → [0, 1] heatmap.
        6. Resize to input image size, apply colour map, overlay on original.

    Parameters
    ----------
    model              : Trained Keras model.
    img_array          : Preprocessed input batch (1, H, W, 3).
    original_img       : PIL Image of the original (unprocessed) image.
    predicted_idx      : Integer index of the predicted class.
    num_classes        : Total number of classes (2 → binary sigmoid output).
    model_dir          : Where to save the heatmap PNG.
    last_conv_layer_name: Name of the target conv layer. Auto-detected if None.

    Returns
    -------
    Path to the saved heatmap PNG, or None if generation fails.
    """
    # ── Locate the last convolutional layer ───────────────────────────────
    # For DenseNet121: the final conv layer is inside a sub-model.
    # We unwrap to find the innermost DenseNet base's last Conv2D.
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)
        if last_conv_layer_name is None:
            logger.warning("No Conv2D layer found — Grad-CAM unavailable.")
            return None

    logger.debug("Grad-CAM target layer: %s", last_conv_layer_name)

    # ── Build gradient model ───────────────────────────────────────────────
    # We need two outputs simultaneously:
    #   1. The feature maps of the last conv layer (spatial activations)
    #   2. The final model predictions (to compute the target class score)
    try:
        grad_model = _build_grad_model(model, last_conv_layer_name)
    except Exception as exc:
        logger.warning("Could not build Grad-CAM model: %s", exc)
        return None

    # ── Compute gradients via GradientTape ────────────────────────────────
    # GradientTape records operations on tensors so we can later call
    # tape.gradient(output, input) to get ∂output/∂input.
    #
    # IMPORTANT: tape.watch(conv_outputs) MUST be called before the operations
    # that link conv_outputs → predictions → class_score are recorded.
    # We therefore run the forward pass in two steps:
    #   Step 1: obtain conv_outputs from grad_model and immediately watch it.
    #   Step 2: compute class_score from predictions inside the tape context.
    # This guarantees the full conv_outputs → class_score gradient path is
    # captured; without this ordering, tape.gradient() returns None.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        # Watch conv_outputs NOW — before class_score is derived from predictions.
        # Since conv_outputs and predictions are produced in the same call, TF2
        # has already recorded the conv_outputs → predictions dependency inside
        # this tape context. Watching here (before class_score ops) ensures the
        # full path conv_outputs → predictions → class_score is differentiable.
        tape.watch(conv_outputs)

        if num_classes == 2:
            # Binary sigmoid: single output neuron → probability of class 1.
            # For class 0: use (1 - sigmoid_output) so gradient flows correctly.
            if predicted_idx == 1:
                class_score = predictions[:, 0]
            else:
                class_score = tf.constant(1.0, dtype=predictions.dtype) - predictions[:, 0]
        else:
            # Multi-class softmax: select the predicted class channel.
            class_score = predictions[:, predicted_idx]

    # Gradient of the class score w.r.t. each conv feature map
    grads = tape.gradient(class_score, conv_outputs)  # shape: (1, h, w, channels)

    if grads is None:
        logger.warning("Grad-CAM: gradient is None — layer may not be differentiable.")
        return None

    # ── Pool gradients → importance weights ────────────────────────────────
    # Global average pooling over spatial dims → one weight per channel.
    # These weights tell us: "how important is channel k to this class?"
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

    # ── Weighted sum of feature maps ───────────────────────────────────────
    conv_outputs_np = conv_outputs[0].numpy()  # (h, w, channels)
    pooled_grads_np = pooled_grads.numpy()     # (channels,)

    # Multiply each channel map by its importance weight and sum
    for i in range(pooled_grads_np.shape[-1]):
        conv_outputs_np[:, :, i] *= pooled_grads_np[i]

    heatmap = np.mean(conv_outputs_np, axis=-1)   # (h, w)

    # ── ReLU + normalise ───────────────────────────────────────────────────
    # ReLU: keep only regions that POSITIVELY contribute to this class.
    # Regions with negative contributions would highlight "anti-class" areas.
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()  # normalise to [0, 1]

    # ── Save overlay ───────────────────────────────────────────────────────
    heatmap_dir = Path(model_dir) / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    heatmap_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
    heatmap_path = heatmap_dir / heatmap_filename

    _save_gradcam_overlay(heatmap, original_img, heatmap_path)
    logger.info("Grad-CAM heatmap saved: %s", heatmap_path)
    return heatmap_path


def _save_gradcam_overlay(
    heatmap: np.ndarray,
    original_img: Image.Image,
    save_path: Path,
    alpha: float = 0.4,
) -> None:
    """
    Overlay the Grad-CAM heatmap on the original image and save as PNG.

    Parameters
    ----------
    heatmap      : 2D float array in [0, 1], shape (h, w).
    original_img : PIL Image of the original input.
    save_path    : Where to save the overlaid PNG.
    alpha        : Heatmap opacity (0=invisible, 1=opaque). Default: 0.4.
    """
    orig_w, orig_h = original_img.size

    # Resize heatmap to match original image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(
        (orig_w, orig_h), resample=Image.LANCZOS
    )

    # Apply JET colormap (blue=low importance, red=high importance)
    # matplotlib 3.7+ uses colormaps[]; cm.get_cmap() was removed in 3.9
    try:
        import matplotlib
        colormap = matplotlib.colormaps["jet"]   # new API (matplotlib >= 3.7)
    except AttributeError:
        colormap = cm.get_cmap("jet")            # fallback for older matplotlib
    heatmap_coloured = colormap(np.array(heatmap_pil) / 255.0)  # (H, W, 4) RGBA
    heatmap_coloured = np.uint8(heatmap_coloured[:, :, :3] * 255)  # RGB only

    # Composite: alpha-blend heatmap over original
    orig_rgb = original_img.convert("RGB")
    orig_arr = np.array(orig_rgb, dtype=np.float32)
    heat_arr = heatmap_coloured.astype(np.float32)

    blended = (1 - alpha) * orig_arr + alpha * heat_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Save with title
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig_rgb);          axes[0].set_title("Original Image");      axes[0].axis("off")
    axes[1].imshow(heatmap_coloured);  axes[1].set_title("Grad-CAM Heatmap");    axes[1].axis("off")
    axes[2].imshow(blended);           axes[2].set_title("Overlay (α=0.4)");     axes[2].axis("off")

    fig.suptitle("VitalSense AI — Grad-CAM Activation Map", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _load_model_and_labels(model_dir: str) -> tuple[keras.Model, dict]:
    """
    Load the Keras model and class-label mapping from disk.
    Caches both in memory so subsequent calls within the same process
    do not re-read from disk (saves ~2s per request).

    Returns
    -------
    tuple (model, class_labels_dict)
    """
    global _model_cache

    # Return cached version if available and same model_dir
    if (
        _model_cache["model"] is not None
        and _model_cache["model_dir"] == model_dir
    ):
        return _model_cache["model"], _model_cache["class_labels"]

    model_dir_path = Path(model_dir).resolve()

    # ── Find model file (.keras preferred, .h5 fallback) ──────────────────
    model_path = model_dir_path / "image_model.keras"
    if not model_path.exists():
        model_path = model_dir_path / "image_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found in '{model_dir_path}'. "
            "Run POST /module3/image/train first."
        )

    # ── Find class labels ──────────────────────────────────────────────────
    labels_path = model_dir_path / "class_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"class_labels.json not found in '{model_dir_path}'. "
            "Re-run training to regenerate it."
        )

    logger.info("Loading model from: %s", model_path)
    model = keras.models.load_model(str(model_path))

    with open(labels_path, "r", encoding="utf-8") as f:
        class_labels = json.load(f)  # {"0": "NORMAL", "1": "PNEUMONIA", ...}

    _model_cache["model"] = model
    _model_cache["class_labels"] = class_labels
    _model_cache["model_dir"] = model_dir

    logger.info("Model and labels loaded. Classes: %s", class_labels)
    return model, class_labels


def _load_and_preprocess(
    img_path: Path,
    img_size: tuple[int, int],
) -> tuple[np.ndarray, Image.Image]:
    """
    Load an image file and preprocess it to match training conditions.

    Preprocessing steps (MUST match image_preprocessing.py exactly):
        1. Load as RGB (3-channel)
        2. Resize to img_size (height, width)
        3. Normalise pixel values: divide by 255.0 → [0, 1]
        4. Add batch dimension: (H, W, 3) → (1, H, W, 3)

    Supports .jpg/.png/.bmp natively via Pillow.
    Supports .dcm (DICOM) if pydicom is installed.

    Returns
    -------
    tuple (img_array, original_pil_image)
        img_array: float32 numpy array (1, H, W, 3) in [0, 1]
        original_pil_image: unmodified PIL Image for Grad-CAM overlay
    """
    ext = img_path.suffix.lower()

    if ext == DICOM_EXT:
        # ── DICOM path (optional dependency) ──────────────────────────────
        try:
            import pydicom
            dicom_data = pydicom.dcmread(str(img_path))
            pixel_array = dicom_data.pixel_array.astype(np.float32)
            # Normalise DICOM pixel values (may be 12-bit or 16-bit)
            pixel_array = (pixel_array - pixel_array.min()) / (
                pixel_array.max() - pixel_array.min() + 1e-8
            )
            pixel_array = (pixel_array * 255).astype(np.uint8)
            if pixel_array.ndim == 2:
                # Grayscale DICOM → convert to RGB by repeating channels
                pil_img = Image.fromarray(pixel_array).convert("RGB")
            else:
                pil_img = Image.fromarray(pixel_array).convert("RGB")
        except ImportError:
            raise ImportError(
                "DICOM file detected but pydicom is not installed. "
                "Install it with: pip install pydicom"
            )
    else:
        # ── Standard image formats via Pillow ─────────────────────────────
        pil_img = Image.open(str(img_path)).convert("RGB")

    original_img = pil_img.copy()

    # Resize to match training input size (height, width)
    pil_resized = pil_img.resize((img_size[1], img_size[0]), Image.LANCZOS)

    # Convert to numpy array and normalise [0, 255] → [0.0, 1.0]
    # This EXACTLY mirrors the rescale=1/255 in ImageDataGenerator
    img_array = np.array(pil_resized, dtype=np.float32) / 255.0

    # Add batch dimension: (H, W, 3) → (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img


def _find_last_conv_layer(model: keras.Model) -> Optional[str]:
    """
    Find the name of the last Conv2D layer in the model (including sub-models).

    For DenseNet121, the final conv layer is inside the base sub-model.
    We search recursively by unwrapping nested Model layers.

    Returns
    -------
    str or None
        Layer name usable with model.get_layer(), or None if not found.
    """
    last_conv_name = None

    def _search(m: keras.Model):
        nonlocal last_conv_name
        for layer in m.layers:
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_name = layer.name
            elif isinstance(layer, keras.Model):
                _search(layer)  # recurse into sub-models

    _search(model)
    return last_conv_name


def _build_grad_model(
    model: keras.Model,
    last_conv_layer_name: str,
) -> keras.Model:
    """
    Build a new Model that simultaneously outputs:
        1. The feature maps of the specified conv layer
        2. The final model predictions

    This is required for GradientTape to record the computation graph
    from the conv layer's output to the final score.

    We search both the top-level model and any nested sub-model.
    """
    # Try top-level first
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        return keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
    except (ValueError, AttributeError):
        pass

    # Search inside nested sub-model (DenseNet121 is a sub-model)
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            try:
                conv_layer = layer.get_layer(last_conv_layer_name)
                # Build grad model using the sub-model's output as intermediate
                inner_grad_model = keras.Model(
                    inputs=layer.inputs,
                    outputs=[conv_layer.output, layer.output],
                )
                # Wrap: top-level input → inner_grad_model → classification head
                inp = model.inputs[0]
                conv_out, inner_out = inner_grad_model(inp, training=False)

                # Get the remaining layers after the base sub-model
                remaining_output = inner_out
                base_found = False
                for top_layer in model.layers:
                    if top_layer == layer:
                        base_found = True
                        continue
                    if base_found:
                        remaining_output = top_layer(remaining_output)

                return keras.Model(inputs=inp, outputs=[conv_out, remaining_output])
            except (ValueError, AttributeError):
                continue

    raise ValueError(
        f"Could not build Grad-CAM model for layer '{last_conv_layer_name}'."
    )
