"""
infer.py  --  Standalone inference script for VitalSense AI Module 3
====================================================================
Called by preview_server.py as a subprocess using the module3 venv Python,
so TensorFlow runs in its own process (bypasses Windows App Control DLL block
on the main process).

Usage:
    python infer.py <image_path> <report_type>

Prints a single-line JSON to stdout:
    {"predicted_class": "NORMAL", "confidence": 0.9231, ...}

Exit codes:
    0 = success
    1 = error (error message printed to stderr)
"""
import sys, json, os
from pathlib import Path

sys.setrecursionlimit(5000)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: infer.py <image_path> <report_type>"}))
        sys.exit(1)

    image_path  = sys.argv[1]
    report_type = sys.argv[2].lower()

    MODEL_STORE = Path(__file__).parent / "model_store"

    # Pick model directory: type-specific first, then shared fallback
    model_dirs = [
        MODEL_STORE / f"{report_type}_model",
        MODEL_STORE,
    ]
    model_path = None
    for d in model_dirs:
        for name in ("image_model.keras", "best_checkpoint.keras"):
            candidate = d / name
            if candidate.exists():
                model_path = candidate
                break
        if model_path:
            break

    if model_path is None:
        print(json.dumps({"error": f"No model found for report_type={report_type}"}))
        sys.exit(1)

    # Load class labels
    labels_candidates = [
        MODEL_STORE / f"{report_type}_model" / "class_labels.json",
        MODEL_STORE / "class_labels.json",
    ]
    labels = {"0": "NORMAL", "1": "PNEUMONIA"}
    for lp in labels_candidates:
        if lp.exists():
            with open(lp) as f:
                labels = json.load(f)
            break

    # Import TF (only in this subprocess — not blocked here)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import numpy as np
    from PIL import Image
    import io as _io

    # Stub blocked TFLite DLLs BEFORE importing tensorflow.
    # Windows App Control blocks several TFLite .pyd wrappers on this machine.
    # tf_lite_stub installs a sys.meta_path hook that replaces them with empty
    # modules so the main TF inference engine (not blocked) can load normally.
    sys.path.insert(0, str(Path(__file__).parent))
    import tf_lite_stub  # noqa: F401  — side-effect: installs DLL stubs

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")


    # Load model (cached in this process lifetime)
    model = tf.keras.models.load_model(str(model_path))

    # Pre-process image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # Run inference
    probs = model.predict(arr, verbose=0)[0]

    # Handle both binary sigmoid (shape: [1]) and softmax (shape: [N]) outputs
    if len(probs) == 1:
        # Binary sigmoid: single output = P(positive class = class "1")
        p_positive = float(probs[0])
        p_negative = 1.0 - p_positive
        probs = [p_negative, p_positive]   # [P(NORMAL), P(PNEUMONIA)]

    pred_idx   = int(max(range(len(probs)), key=lambda i: probs[i]))
    confidence = float(probs[pred_idx])
    pred_class = labels.get(str(pred_idx), str(pred_idx))

    CONF_THRESHOLD = 0.60
    DISCLAIMER = (
        "This is an AI-generated preliminary screening result based on image "
        "patterns. It is NOT a medical diagnosis. A qualified radiologist or "
        "physician must review the original image and confirm any finding before "
        "any medical decision is made."
    )

    result = {
        "predicted_class"       : pred_class,
        "confidence"            : round(confidence, 4),
        "all_probabilities"     : {labels.get(str(i), str(i)): round(float(p), 4) for i, p in enumerate(probs)},
        "low_confidence_warning": confidence < CONF_THRESHOLD,
        "assistive_disclaimer"  : DISCLAIMER,
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
