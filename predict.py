"""
Inference for fruit classification: batch prediction, TTA, and Flask/CLI API.
"""

import argparse
import hashlib
import io
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import train

train.configure_tensorflow()

NORMALIZE_SIZE = 512
DEFAULT_CROP_FRACTION = 0.85

# ponytail: process-global singleton; upgrade path: explicit reload API
_model = None
_class_names = None


def load_model(model_path):
    """Load a saved Keras model for inference (weights only, no optimizer)."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects=train.get_custom_objects(),
        compile=False,
    )


def load_model_and_labels(model_path=None):
    """Load Keras model and class names (cached)."""
    global _model, _class_names
    model_path = model_path or train.MODEL_PATH
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run: python train.py train --backbone {train.APP_BACKBONE}"
            )
        _model = load_model(model_path)
    if _class_names is None:
        _class_names = train.load_class_names()
    return _model, _class_names


def load_rgb_image(image_source):
    """Open image bytes or path as upright RGB."""
    if isinstance(image_source, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(image_source))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                "Use a real file path, for example:\n"
                '  python predict.py --image "dataset\\raw\\Strawberi\\r0_0.jpg"'
            )
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image_source))).convert(
        "RGB"
    )


def default_center_crop_box(img, fraction=DEFAULT_CROP_FRACTION):
    """Center square crop box in pixel coordinates."""
    width, height = img.size
    side = min(width, height) * fraction
    left = (width - side) / 2
    top = (height - side) / 2
    return left, top, left + side, top + side


def validate_crop_fractions(crop):
    """Require positive fractional crop box in [0, 1]."""
    for key in ("x", "y", "w", "h"):
        val = crop.get(key)
        if not isinstance(val, (int, float)):
            raise ValueError("Invalid crop region.")
    if not (0 <= crop["x"] <= 1 and 0 <= crop["y"] <= 1):
        raise ValueError("Invalid crop region.")
    if (
        crop["w"] <= 0
        or crop["h"] <= 0
        or crop["x"] + crop["w"] > 1.001
        or crop["y"] + crop["h"] > 1.001
    ):
        raise ValueError("Invalid crop region.")


def apply_crop_fractions(img, crop):
    """Crop using fractions {x, y, w, h} relative to image width/height."""
    validate_crop_fractions(crop)
    width, height = img.size
    left = crop["x"] * width
    top = crop["y"] * height
    right = left + crop["w"] * width
    bottom = top + crop["h"] * height
    cropped = img.crop((left, top, right, bottom))
    if cropped.size[0] < 1 or cropped.size[1] < 1:
        raise ValueError("Invalid crop region.")
    return cropped


def normalize_square(img, size=NORMALIZE_SIZE):
    """Letterbox to a square white canvas (device-independent export)."""
    width, height = img.size
    width = max(1, width)
    height = max(1, height)
    scale = min(size / width, size / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return out


def prepare_model_image(image_source, crop=None, use_full=False):
    """
    EXIF upright -> optional crop -> 512 square normalize.
    Returns PIL image ready for 224 resize.
    """
    img = load_rgb_image(image_source)
    if crop:
        img = apply_crop_fractions(img, crop)
    elif use_full:
        img = img.crop(default_center_crop_box(img))
    img = normalize_square(img)
    return img


def image_input_meta(img):
    """Hash and size metadata for the normalized square input."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    width, height = img.size
    return {
        "input_width": width,
        "input_height": height,
        "input_hash": hashlib.sha256(data).hexdigest()[:8],
    }


def preprocess_image(image_source, crop=None, use_full=False):
    """
    Load and resize an image for the saved Keras model.
    Returns float32 array (1, H, W, 3) with pixel values in [0, 255].

    Backbone preprocess_input is baked into the saved model graph — do not
    apply it here (double scaling caused wrong predictions).
    """
    if crop is not None or use_full or not isinstance(image_source, (str, os.PathLike)):
        img = prepare_model_image(image_source, crop=crop, use_full=use_full)
    else:
        img = load_rgb_image(image_source)

    img = img.resize(train.IMG_SIZE, Image.Resampling.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def tta_transforms(images):
    """Generate TTA views: original, h-flip, v-flip, both flips, slight crop."""
    h, w = train.IMG_SIZE
    views = [
        images,
        tf.image.flip_left_right(images),
        tf.image.flip_up_down(images),
        tf.image.flip_up_down(tf.image.flip_left_right(images)),
        tf.image.resize(tf.image.central_crop(images, 0.9), (h, w)),
    ]
    return views[: train.TTA_VIEWS]


def predict_batch(model, images, use_tta=False):
    """Predict probabilities for a batch, optionally with TTA averaging."""
    if not use_tta:
        return model.predict(images, verbose=0)

    views = tta_transforms(images)
    accum = None
    for view in views:
        probs = model.predict(view, verbose=0)
        accum = probs if accum is None else accum + probs
    return accum / len(views)


def collect_predictions(model, dataset, use_tta=False):
    """Run model on dataset; return y_true, y_pred, y_prob arrays."""
    y_true, y_pred, y_prob = [], [], []
    for images, labels in dataset:
        probs = predict_batch(model, images, use_tta=use_tta)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def load_fruit_info():
    """Load fruit metadata JSON for the information panel."""
    if not os.path.exists(train.FRUIT_INFO_PATH):
        return {}
    with open(train.FRUIT_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


def confidence_message(confidence):
    """User-facing status message based on confidence score."""
    if confidence >= train.CONFIDENCE_HIGH:
        return "High confidence! The fruit has been identified successfully."
    if confidence >= train.CONFIDENCE_MEDIUM:
        return "Moderate confidence. Consider retaking the photo with better lighting."
    return "Low confidence. Please upload a clearer image following the tips below."


def predict(image_source, top_k=3, crop=None, use_full=False):
    """
    Predict fruit class from an image.
    Returns dict with prediction, confidence, top3, message, and fruit_info.
    """
    model, class_names = load_model_and_labels()
    if crop is not None or use_full or not isinstance(image_source, (str, os.PathLike)):
        model_img = prepare_model_image(image_source, crop=crop, use_full=use_full)
        meta = image_input_meta(model_img)
        tensor = preprocess_image(image_source, crop=crop, use_full=use_full)
    else:
        meta = {}
        tensor = preprocess_image(image_source)

    probs = predict_batch(model, tensor, use_tta=train.USE_TTA)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    top3 = [
        {"class": train.to_local_name(class_names[i]), "confidence": float(probs[i])}
        for i in top_indices
    ]

    best_idx = int(top_indices[0])
    local_name = train.to_local_name(class_names[best_idx])
    best_conf = float(probs[best_idx])

    fruit_info_db = load_fruit_info()
    fruit_info = fruit_info_db.get(
        local_name,
        {
            "description": f"{local_name} is a local fruit included in this dataset.",
            "nutrition": [],
            "season": "Availability varies by region in Malaysia.",
            "price": "Price varies by market and season.",
            "uses": "Commonly consumed fresh or used in local dishes.",
        },
    )

    return {
        "prediction": local_name,
        "confidence": best_conf,
        "top3": top3,
        "message": confidence_message(best_conf),
        "fruit_info": fruit_info,
        **meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict fruit class from an image")
    parser.add_argument("--image", required=True, help="Path to fruit image")
    args = parser.parse_args()

    result = predict(args.image)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nTop 3 predictions:")
    for item in result["top3"]:
        print(f"  {item['class']}: {item['confidence'] * 100:.2f}%")
    print(f"\n{result['message']}")


def _self_check():
    """Same image bytes must yield identical preprocessed tensors."""
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(120, 80, 40)).save(buf, format="JPEG")
    data = buf.getvalue()
    a = preprocess_image(data)
    b = preprocess_image(data)
    assert np.array_equal(a, b), "preprocess_image must be deterministic"
    assert a.shape == (1, *train.IMG_SIZE, 3)

    crop = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
    c1 = preprocess_image(data, crop=crop)
    c2 = preprocess_image(data, crop=crop)
    assert np.array_equal(c1, c2), "crop fractions must be deterministic"
    full1 = preprocess_image(data, use_full=True)
    full2 = preprocess_image(data, use_full=True)
    assert np.array_equal(full1, full2), "use_full must be deterministic"

    try:
        validate_crop_fractions({"x": 0, "y": 0, "w": 0, "h": 0})
        raise AssertionError("zero crop should be rejected")
    except ValueError:
        pass
    normalize_square(Image.new("RGB", (1, 1), color=(1, 2, 3)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        _self_check()
        print("predict.py self-check ok")
        sys.exit(0)
    main()
