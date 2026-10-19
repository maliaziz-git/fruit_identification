"""
Flask web application for the Local Fruit Identification System.
Serves the upload GUI and /predict API endpoint.

Inference always uses MobileNetV3 (train.APP_BACKBONE) trained on the
80:10:10 split (train.APP_SPLIT). Compare backbones or split ratios with
evaluate.py --compare-models and evaluate.py --compare-splits.
"""

import io
import json
import os
import re
import socket

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

import train
import predict as predictor
import poster as poster_gen

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = train.MAX_CONTENT_LENGTH

FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


def _react_build_ready():
    return os.path.isfile(os.path.join(FRONTEND_DIST, "index.html"))


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in train.ALLOWED_UPLOAD_EXTENSIONS


def _validate_upload(file):
    """Return (image_bytes, None) or (None, error_response_tuple)."""
    if file.filename == "":
        return None, (jsonify({"error": "No file selected."}), 400)
    if not allowed_file(file.filename):
        return None, (
            jsonify(
                {"error": "Invalid file type. Use JPG, PNG, or WEBP (max 8 MB)."}
            ),
            400,
        )
    image_bytes = file.read()
    if len(image_bytes) > train.MAX_CONTENT_LENGTH:
        return None, (
            jsonify({"error": f"File exceeds {train.MAX_UPLOAD_MB} MB limit."}),
            400,
        )
    return image_bytes, None


@app.route("/api/fruits")
def api_fruits():
    return jsonify(_load_encyclopedia_fruits())


@app.route("/")
def index():
    if _react_build_ready():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return render_template("index.html")


def _load_encyclopedia_fruits():
    """Sorted fruit list for encyclopedia template."""
    with open(train.FRUIT_INFO_PATH, encoding="utf-8") as f:
        info = json.load(f)
    fruits = []
    for local_name in sorted(info.keys()):
        entry = info[local_name]
        fruits.append(
            {
                "id": local_name,
                "local_name": local_name,
                "english_name": entry.get("english_name", local_name),
                "botanical_name": entry.get("botanical_name", ""),
                "description": entry.get("description", ""),
                "nutrition": entry.get("nutrition") or [],
                "season": entry.get("season", ""),
            }
        )
    return fruits


def _find_sample_image(local_name):
    """First image in dataset/raw/<local_name>/, or None."""
    if local_name not in train.get_allowed_class_names():
        return None
    folder = os.path.join(train.RAW_DIR, local_name)
    if not os.path.isdir(folder):
        return None
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in train.IMAGE_EXTENSIONS:
            return os.path.join(folder, fname)
    return None


@app.route("/encyclopedia")
def encyclopedia():
    if _react_build_ready():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return render_template("encyclopedia.html", fruits=_load_encyclopedia_fruits())


@app.route("/fruit-sample/<local_name>")
def fruit_sample(local_name):
    path = _find_sample_image(local_name)
    if not path:
        abort(404)
    return send_file(path)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    image_bytes, err = _validate_upload(file)
    if err:
        return err

    try:
        crop = None
        use_full = request.form.get("use_full") == "1"
        crop_raw = request.form.get("crop")
        if crop_raw:
            crop = json.loads(crop_raw)
            if not all(k in crop for k in ("x", "y", "w", "h")):
                return jsonify({"error": "Invalid crop JSON."}), 400
            predictor.validate_crop_fractions(crop)
        result = predictor.predict(image_bytes, crop=crop, use_full=use_full)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


def _safe_poster_filename(name):
    """Sanitize fruit name for download filename."""
    safe = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    safe = re.sub(r"[-\s]+", "-", safe)
    return (safe or "fruit") + "-poster.png"


@app.route("/generate-poster", methods=["POST"])
def generate_poster_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    image_bytes, err = _validate_upload(file)
    if err:
        return err

    metadata_raw = request.form.get("metadata")
    if not metadata_raw:
        return jsonify({"error": "Missing metadata."}), 400

    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid metadata JSON."}), 400

    if "prediction" not in metadata:
        return jsonify({"error": "Metadata must include prediction."}), 400

    try:
        png_bytes = poster_gen.build_poster(image_bytes, metadata)
    except Exception as e:
        return jsonify({"error": f"Poster generation failed: {e}"}), 500

    filename = _safe_poster_filename(metadata.get("prediction", "fruit"))
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=True,
        download_name=filename,
    )


if _react_build_ready():

    @app.route("/assets/<path:filename>")
    def react_assets(filename):
        return send_from_directory(os.path.join(FRONTEND_DIST, "assets"), filename)

    @app.route("/<path:path>")
    def spa_fallback(path):
        if path in ("predict", "generate-poster", "api", "fruit-sample"):
            abort(404)
        full = os.path.join(FRONTEND_DIST, path)
        if os.path.isfile(full):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")


def _local_lan_ip():
    """Best-effort LAN IP for phone/tablet access on the same Wi-Fi."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    train.ensure_directories()
    port = 5000
    lan_ip = _local_lan_ip()
    print("Local Fruit Identification System")
    print(f"Model: {train.BACKBONES[train.APP_BACKBONE]['display_name']}")
    print(f"Split: {train.SPLIT_PRESETS[train.APP_SPLIT]['label']}")
    print(f"Model path: {train.MODEL_PATH}")
    print(f"On this PC:  http://127.0.0.1:{port}")
    if lan_ip:
        print(f"On your phone (same Wi-Fi): http://{lan_ip}:{port}")
    else:
        print("On your phone: use this PC's Wi-Fi IP instead of 127.0.0.1")
    app.run(host="0.0.0.0", port=port, debug=False)
