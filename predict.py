"""
Inference for fruit classification: batch prediction, TTA, and Flask/CLI API.
"""

import argparse
import io
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import train

train.configure_tensorflow()

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


def preprocess_image(image_source):
    """
    Load and resize an image for the saved Keras model.
    Returns float32 array (1, H, W, 3) with pixel values in [0, 255].

    Backbone preprocess_input is baked into the saved model graph — do not
    apply it here (double scaling caused wrong predictions).
    """
    if isinstance(image_source, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(image_source))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                "Use a real file path, for example:\n"
                '  python predict.py --image "dataset\\raw\\Strawberi\\r0_0.jpg"'
            )
        img = Image.open(path).convert("RGB")
    else:
        img = Image.open(io.BytesIO(image_source)).convert("RGB")

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


def predict(image_source, top_k=3):
    """
    Predict fruit class from an image.
    Returns dict with prediction, confidence, top3, message, and fruit_info.
    """
    model, class_names = load_model_and_labels()
    probs = predict_batch(model, preprocess_image(image_source), use_tta=train.USE_TTA)[0]

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


if __name__ == "__main__":
    main()
