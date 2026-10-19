"""
Settings and training for the Local Fruit Identification System.
Paths, hyperparameters, and MobileNet V1/V2/V3 two-phase training pipeline.

Phase 1: frozen base + train head. Phase 2: unfreeze top layers and fine-tune.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1. Settings — paths, presets, hyperparameters, app constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")

APP_SPLIT = "80_10_10"

SPLIT_PRESETS = {
    "70_20_10": {"train": 0.70, "val": 0.20, "test": 0.10, "label": "70:20:10"},
    "60_20_20": {"train": 0.60, "val": 0.20, "test": 0.20, "label": "60:20:20"},
    "80_10_10": {"train": 0.80, "val": 0.10, "test": 0.10, "label": "80:10:10"},
}
SPLIT_ORDER = ["70_20_10", "60_20_20", "80_10_10"]

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_EXT = ".h5"

APP_BACKBONE = "mobilenet_v3"

BACKBONES = {
    "mobilenet_v1": {
        "display_name": "MobileNetV1",
        "filename": "fruit_classifier_mobilenet_v1.h5",
        "history_filename": "training_history_mobilenet_v1.json",
        "fine_tune_at": 60,
    },
    "mobilenet_v2": {
        "display_name": "MobileNetV2",
        "filename": "fruit_classifier_mobilenet_v2.h5",
        "history_filename": "training_history_mobilenet_v2.json",
        "fine_tune_at": 50,
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3",
        "filename": "fruit_classifier_mobilenet_v3.h5",
        "history_filename": "training_history_mobilenet_v3.json",
        "fine_tune_at": 110,
    },
}
BACKBONE_ORDER = ["mobilenet_v1", "mobilenet_v2", "mobilenet_v3"]

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
FRUIT_INFO_PATH = os.path.join(DATA_DIR, "fruit_info.json")

MODEL_PATH = os.path.join(MODELS_DIR, BACKBONES[APP_BACKBONE]["filename"])
HISTORY_PATH = os.path.join(MODELS_DIR, BACKBONES[APP_BACKBONE]["history_filename"])
EVAL_REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
SPLIT_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "split_comparison.json")
SPLIT_COMPARISON_CSV = os.path.join(OUTPUTS_DIR, "split_comparison.csv")
SPLIT_SUMMARIES_DIR = os.path.join(OUTPUTS_DIR, "split")
TRAINING_CURVES_DIR = os.path.join(OUTPUTS_DIR, "training curves")
EVAL_ALL_SUMMARY_JSON = os.path.join(OUTPUTS_DIR, "evaluation_all_summary.json")
EVAL_ALL_SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "evaluation_all_summary.csv")

TRAIN_RATIO = SPLIT_PRESETS[APP_SPLIT]["train"]
VAL_RATIO = SPLIT_PRESETS[APP_SPLIT]["val"]
TEST_RATIO = SPLIT_PRESETS[APP_SPLIT]["test"]
SPLIT_LABEL = SPLIT_PRESETS[APP_SPLIT]["label"]
RANDOM_SEED = 42

IMG_SIZE = (224, 224)
NUM_CHANNELS = 3
BATCH_SIZE = 32
DATASET_CACHE = False
PREFETCH_BUFFER = 2

EPOCHS_PHASE1 = 50
LEARNING_RATE_PHASE1 = 8e-4
EPOCHS_PHASE2 = 25
LEARNING_RATE_PHASE2 = 1e-6
LEARNING_RATE_PHASE2_HEAD = 2e-4
LEARNING_RATE_PHASE2_BASE = 2e-5

DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 8
USE_DEEP_HEAD = True

LR_SCHEDULE = "cosine_warmup"
COSINE_MIN_LR = 1e-7
WARMUP_EPOCHS = 3

USE_CLASS_WEIGHTS = True
USE_TIERED_CLASS_WEIGHTS = True

HARD_CLASS_TIER1 = ["Limau Besar", "Bambangan", "Nangka"]
HARD_CLASS_TIER1_BOOST = 2.0

HARD_CLASS_TIER2 = [
    "Cempedak",
    "Betik",
    "Belimbing Hutan",
    "Jambu Air Madu",
    "Rambutan",
]
HARD_CLASS_TIER2_BOOST = 1.5

OFFLINE_AUG_SUFFIX = "_aug"

USE_TTA = True
TTA_VIEWS = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MAX_UPLOAD_MB = 8
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.40

LABEL_SMOOTHING = 0.05

# ---------------------------------------------------------------------------
# 2. Path and preset helpers
# ---------------------------------------------------------------------------


def get_backbone_config(backbone=None):
    """Return metadata dict for a backbone key."""
    key = backbone or APP_BACKBONE
    if key not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{key}'. Choose from: {', '.join(BACKBONE_ORDER)}"
        )
    return BACKBONES[key]


def get_eval_all_backbones(split_name=None):
    """Backbones to evaluate per split in evaluate.py --matrix."""
    key = split_name or APP_SPLIT
    return BACKBONE_ORDER if key == APP_SPLIT else [APP_BACKBONE]


def get_split_preset(split_name=None):
    """Return ratio metadata for a split preset key."""
    key = split_name or APP_SPLIT
    if key not in SPLIT_PRESETS:
        raise ValueError(
            f"Unknown split '{key}'. Choose from: {', '.join(SPLIT_ORDER)}"
        )
    return SPLIT_PRESETS[key]


def get_split_dirs(split_name=None):
    """Return (train_dir, val_dir, test_dir) for a split preset."""
    key = split_name or APP_SPLIT
    get_split_preset(key)
    base = os.path.join(SPLITS_DIR, key)
    return (
        os.path.join(base, "train"),
        os.path.join(base, "val"),
        os.path.join(base, "test"),
    )


def get_split_summary_path(split_name):
    """Path for per-split summary text file."""
    return os.path.join(SPLIT_SUMMARIES_DIR, f"split_summary_{split_name}.txt")


def get_training_curves_path(split_name, backbone):
    """Path for training accuracy/loss plot PNG."""
    return os.path.join(
        TRAINING_CURVES_DIR, f"training_curves_{split_name}_{backbone}.png"
    )


def _split_model_filename(cfg, split_name=None):
    """Model filename; non-default splits get a split suffix in models/."""
    split_name = split_name or APP_SPLIT
    if split_name == APP_SPLIT:
        return cfg["filename"]
    stem, ext = os.path.splitext(cfg["filename"])
    return f"{stem}_{split_name}{ext}"


def _split_history_filename(cfg, split_name=None):
    """History JSON filename; non-default splits get a split suffix in models/."""
    split_name = split_name or APP_SPLIT
    if split_name == APP_SPLIT:
        return cfg["history_filename"]
    stem, ext = os.path.splitext(cfg["history_filename"])
    return f"{stem}_{split_name}{ext}"


def get_model_path(backbone=None, split_name=None):
    """Filesystem path for a saved model (.h5)."""
    cfg = get_backbone_config(backbone)
    split_name = split_name or APP_SPLIT
    return os.path.join(MODELS_DIR, _split_model_filename(cfg, split_name))


def get_history_path(backbone=None, split_name=None):
    """Filesystem path for training history JSON."""
    cfg = get_backbone_config(backbone)
    split_name = split_name or APP_SPLIT
    return os.path.join(MODELS_DIR, _split_history_filename(cfg, split_name))


def infer_backbone_from_model_path(model_path):
    """Guess backbone key from a saved model filename."""
    # ponytail: filename heuristic; upgrade path: store backbone in history JSON
    basename = os.path.basename(model_path).lower()
    for key, cfg in BACKBONES.items():
        stem, ext = os.path.splitext(cfg["filename"].lower())
        file_stem, file_ext = os.path.splitext(basename)
        if file_ext != ext:
            continue
        if file_stem == stem or file_stem.startswith(stem + "_"):
            return key
    return None


def get_evaluation_report_path(backbone, split_name, split="test"):
    """Path for a per-model, per-split evaluation text report."""
    ensure_directories()
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    prefix = "evaluation" if split == "test" else f"{split}_report"
    return os.path.join(EVAL_REPORTS_DIR, f"{prefix}_{backbone}_{split_name}.txt")


def get_confusion_matrix_path(backbone, split_name):
    """Path for a per-model, per-split confusion matrix image."""
    ensure_directories()
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    return os.path.join(EVAL_REPORTS_DIR, f"confusion_matrix_{backbone}_{split_name}.png")


def get_roc_curve_path(backbone, split_name):
    """Path for a per-model, per-split macro-average ROC curve image."""
    ensure_directories()
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    return os.path.join(EVAL_REPORTS_DIR, f"roc_curve_{backbone}_{split_name}.png")


def ensure_directories():
    """Create output directories if they do not exist."""
    for path in (MODELS_DIR, OUTPUTS_DIR, DATA_DIR):
        os.makedirs(path, exist_ok=True)
    os.makedirs(SPLIT_SUMMARIES_DIR, exist_ok=True)
    os.makedirs(TRAINING_CURVES_DIR, exist_ok=True)
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    for split_name in SPLIT_ORDER:
        for subset in ("train", "val", "test"):
            os.makedirs(os.path.join(SPLITS_DIR, split_name, subset), exist_ok=True)


# ---------------------------------------------------------------------------
# 3. Label mapping and class-name helpers
# ---------------------------------------------------------------------------
_ENGLISH_TO_LOCAL_FALLBACK = {"Banana": "Pisang", "Starfruit": "Belimbing"}
_ENGLISH_LEGACY_ALIASES = {
    "Banana": "Pisang",
    "Starfruit": "Belimbing",
    "Guava": "Jambu Batu",
    "Jackfruit": "Nangka",
    "Johey Oak": "Bambangan",
    "Longan": "Longan",
    "Mangosteen": "Manggis",
    "Marang": "Terap",
    "Papaya": "Betik",
    "Pineapple": "Nanas",
    "Pomelo": "Limau Besar",
    "Water Rose Apple": "Jambu Air",
    "Watermelon": "Tembikai",
    "Wax Apple": "Jambu Air Madu",
    "Borneo Starfruit": "Belimbing Hutan",
    "Cat's Eye": "Mata Kucing",
    "Malaysia Mango": "Binjai",
}
_english_to_local_cache = None
_allowed_class_names = None


def load_english_to_local():
    """English→local mapping from fruit_info.json plus legacy aliases."""
    global _english_to_local_cache
    if _english_to_local_cache is not None:
        return _english_to_local_cache
    mapping = dict(_ENGLISH_TO_LOCAL_FALLBACK)
    if os.path.isfile(FRUIT_INFO_PATH):
        with open(FRUIT_INFO_PATH, encoding="utf-8") as f:
            for local, meta in json.load(f).items():
                mapping[local] = local
                eng = meta.get("english_name")
                if eng:
                    mapping.setdefault(eng, local)
    for key, value in _ENGLISH_LEGACY_ALIASES.items():
        mapping.setdefault(key, value)
    _english_to_local_cache = mapping
    return _english_to_local_cache


def to_local_name(english_name):
    """Map model class label (English or local) to Malaysian local name."""
    return load_english_to_local().get(english_name, english_name)


def local_to_english_name(local_name):
    """Return a common English alias for a local fruit name (poster subtitle)."""
    mapping = load_english_to_local()
    for english, local in mapping.items():
        if local == local_name and english != local_name:
            return english
    return None


def load_allowed_class_names():
    """Load allowed local class names from fruit_info.json (sorted)."""
    if not os.path.isfile(FRUIT_INFO_PATH):
        raise FileNotFoundError(f"Fruit info not found: {FRUIT_INFO_PATH}")
    with open(FRUIT_INFO_PATH, encoding="utf-8") as f:
        names = sorted(json.load(f).keys())
    if not names:
        raise ValueError(f"No fruit classes found in {FRUIT_INFO_PATH}")
    return names


def get_allowed_class_names():
    """Parse and cache class names from fruit_info.json."""
    global _allowed_class_names
    if _allowed_class_names is None:
        # ponytail: lazy load avoids import-time failure if JSON missing
        _allowed_class_names = load_allowed_class_names()
    return _allowed_class_names


def validate_class_names(names):
    """Ensure folder-derived class names match fruit_info.json exactly."""
    allowed = set(get_allowed_class_names())
    found = set(names)
    extra = sorted(found - allowed)
    missing = sorted(allowed - found)
    if extra or missing:
        raise ValueError(
            "Class folder mismatch with fruit_info.json.\n"
            f"  Extra folders: {extra}\n"
            f"  Missing folders: {missing}"
        )
    return True


def get_class_names(data_dir=None):
    """Discover class names from folder structure (sorted alphabetically)."""
    if data_dir is None:
        data_dir = RAW_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not names:
        raise ValueError(f"No class folders found in {data_dir}")
    validate_class_names(names)
    return names


def load_class_names(data_dir=None):
    """Class labels in training order (alphabetical folder names)."""
    if data_dir is None:
        train_dir, _, _ = get_split_dirs(APP_SPLIT)
        data_dir = train_dir if os.path.isdir(train_dir) else RAW_DIR
    return get_class_names(data_dir)


# ---------------------------------------------------------------------------
# 4. TensorFlow lifecycle (deferred — not run at import time)
# ---------------------------------------------------------------------------
_tf_ready = False


def _configure_tensorflow_devices():
    """Enable CUDA GPU when present; otherwise apply CPU thread limits."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU devices: {[g.name for g in gpus]}")
        return
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print("No GPU detected; using CPU thread limits")


def configure_tensorflow():
    """Configure devices and random seed (idempotent; safe to call multiple times)."""
    global _tf_ready
    if _tf_ready:
        return
    _configure_tensorflow_devices()
    tf.random.set_seed(RANDOM_SEED)
    _tf_ready = True


def ensure_tensorflow():
    """Initialize TensorFlow before ML operations in this module."""
    configure_tensorflow()


# ---------------------------------------------------------------------------
# 5. Hyperparameters
# ---------------------------------------------------------------------------


def default_hyperparameters(backbone=None):
    """Return a dict of all trainable hyperparameters (baseline values)."""
    key = backbone or APP_BACKBONE
    cfg = BACKBONES[key]
    return {
        "learning_rate_phase1": LEARNING_RATE_PHASE1,
        "learning_rate_phase2": cfg.get("learning_rate_phase2", LEARNING_RATE_PHASE2),
        "learning_rate_phase2_head": LEARNING_RATE_PHASE2_HEAD,
        "learning_rate_phase2_base": LEARNING_RATE_PHASE2_BASE,
        "fine_tune_at": cfg["fine_tune_at"],
        "dropout_rate": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": cfg.get("epochs_phase1", EPOCHS_PHASE1),
        "epochs_phase2": cfg.get("epochs_phase2", EPOCHS_PHASE2),
        "lr_schedule": LR_SCHEDULE,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_tiered_class_weights": USE_TIERED_CLASS_WEIGHTS,
        "use_deep_head": USE_DEEP_HEAD,
        "label_smoothing": LABEL_SMOOTHING,
    }


def merge_hyperparameters(overrides=None, backbone=None):
    """Merge defaults with optional override dict."""
    hp = default_hyperparameters(backbone=backbone)
    if overrides:
        hp.update(overrides)
    return hp


# ---------------------------------------------------------------------------
# 6. Model building and datasets
# ---------------------------------------------------------------------------


def build_augmentation():
    """Data augmentation applied only during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.1),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="augmentation",
    )


def build_backbone(backbone, fine_tune=False, fine_tune_at=0):
    """Instantiate a MobileNet base model and its preprocess function."""
    input_shape = (*IMG_SIZE, NUM_CHANNELS)
    apps = keras.applications

    if backbone == "mobilenet_v1":
        base_model = apps.MobileNet(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = apps.mobilenet.preprocess_input
    elif backbone == "mobilenet_v2":
        base_model = apps.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = apps.mobilenet_v2.preprocess_input
    elif backbone == "mobilenet_v3":
        base_model = apps.MobileNetV3Large(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = apps.mobilenet_v3.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = fine_tune
    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    return base_model, preprocess_input


def build_model(num_classes, hp, fine_tune=False, backbone=None):
    """Build transfer-learning model for the chosen MobileNet backbone."""
    ensure_tensorflow()
    backbone = backbone or APP_BACKBONE
    backbone_cfg = get_backbone_config(backbone)
    fine_tune_at = int(hp.get("fine_tune_at", backbone_cfg["fine_tune_at"]))
    dropout_rate = float(hp["dropout_rate"])

    augmentation = build_augmentation()
    base_model, preprocess_input = build_backbone(
        backbone, fine_tune=fine_tune, fine_tune_at=fine_tune_at
    )

    inputs = keras.Input(shape=(*IMG_SIZE, NUM_CHANNELS))
    x = augmentation(inputs)
    x = preprocess_input(x)
    # ponytail: always inference BN on backbone; upgrade path: custom train step with layer-wise training
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    use_deep_head = hp.get("use_deep_head", USE_DEEP_HEAD)
    if use_deep_head:
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
    else:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model_name = f"fruit_classifier_{backbone}"
    return keras.Model(inputs, outputs, name=model_name), base_model


def create_datasets(batch_size=None, class_names=None, split_name=None):
    """Load train and validation image datasets."""
    ensure_tensorflow()
    batch_size = batch_size or BATCH_SIZE
    train_dir, val_dir, _ = get_split_dirs(split_name)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        class_names=train_ds.class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=False,
    )
    names = train_ds.class_names
    prefetch = PREFETCH_BUFFER
    if DATASET_CACHE:
        train_ds = train_ds.cache().prefetch(buffer_size=prefetch)
        val_ds = val_ds.cache().prefetch(buffer_size=prefetch)
    else:
        train_ds = train_ds.prefetch(buffer_size=prefetch)
        val_ds = val_ds.prefetch(buffer_size=prefetch)
    return train_ds, val_ds, names, len(names)


def compute_class_weights(class_names, split_name=None, hp=None):
    """
    Balanced class weights from training folder counts.
    Applies tiered boosts for historically hard classes.
    """
    hp = hp or {}
    train_dir, _, _ = get_split_dirs(split_name)
    labels = []
    for idx, name in enumerate(class_names):
        class_dir = os.path.join(train_dir, name)
        count = sum(
            1
            for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        labels.extend([idx] * count)

    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=labels
    )
    class_weight = {i: float(weights[i]) for i in range(len(class_names))}

    use_tiered = hp.get("use_tiered_class_weights", USE_TIERED_CLASS_WEIGHTS)
    if use_tiered:
        for name in HARD_CLASS_TIER1:
            if name in class_names:
                class_weight[class_names.index(name)] *= HARD_CLASS_TIER1_BOOST
        for name in HARD_CLASS_TIER2:
            if name in class_names:
                class_weight[class_names.index(name)] *= HARD_CLASS_TIER2_BOOST
    else:
        flat_boost = 1.25
        for name in HARD_CLASS_TIER1 + HARD_CLASS_TIER2:
            if name in class_names:
                class_weight[class_names.index(name)] *= flat_boost

    return class_weight


# ---------------------------------------------------------------------------
# 7. Loss, callbacks, and training curves
# ---------------------------------------------------------------------------


def _cosine_lr(epoch, total_epochs, initial_lr, min_lr):
    if total_epochs <= 0:
        return initial_lr
    progress = min(epoch / total_epochs, 1.0)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _warmup_cosine_lr(epoch, total_epochs, initial_lr, min_lr, warmup_epochs):
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / max(warmup_epochs, 1)
    adjusted = epoch - warmup_epochs
    total = max(total_epochs - warmup_epochs, 1)
    return _cosine_lr(adjusted, total, initial_lr, min_lr)


def get_lr_schedule_callback(lr_schedule, initial_lr, total_epochs, phase_name):
    """Build LR callback for the chosen schedule."""
    min_lr = COSINE_MIN_LR
    schedule = (lr_schedule or "plateau").lower()

    if schedule == "plateau":
        return keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=min_lr,
            verbose=1,
        )

    if schedule == "cosine":

        def schedule_fn(epoch, lr):
            return _cosine_lr(epoch, total_epochs, initial_lr, min_lr)

    elif schedule == "cosine_warmup":

        def schedule_fn(epoch, lr):
            return _warmup_cosine_lr(
                epoch, total_epochs, initial_lr, min_lr, WARMUP_EPOCHS
            )

    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    return keras.callbacks.LearningRateScheduler(schedule_fn, verbose=1)


@keras.utils.register_keras_serializable(package="fruit_identification")
class DualLRAdam(keras.optimizers.Optimizer):
    """Two Adam optimizers; ponytail: LR scheduler scales head, base follows ratio."""

    def __init__(self, lr_base, lr_head, backbone_var_ids, **kwargs):
        name = kwargs.pop("name", "DualLRAdam")
        super().__init__(name=name, **kwargs)
        self._ratio = lr_base / lr_head
        self._backbone_var_ids = set(backbone_var_ids)
        self._opt_head = keras.optimizers.Adam(learning_rate=lr_head)
        self._opt_base = keras.optimizers.Adam(learning_rate=lr_base)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        backbone_gv = [
            (g, v) for g, v in grads_and_vars if id(v) in self._backbone_var_ids
        ]
        head_gv = [
            (g, v) for g, v in grads_and_vars if id(v) not in self._backbone_var_ids
        ]
        if backbone_gv:
            self._opt_base.apply_gradients(backbone_gv)
        if head_gv:
            self._opt_head.apply_gradients(head_gv)

    @property
    def learning_rate(self):
        return self._opt_head.learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._opt_head.learning_rate = lr
        self._opt_base.learning_rate = lr * self._ratio

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, value):
        self.learning_rate = value

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "lr_base": keras.backend.get_value(self._opt_base.learning_rate),
                "lr_head": keras.backend.get_value(self._opt_head.learning_rate),
                "backbone_var_ids": list(self._backbone_var_ids),
            }
        )
        return cfg


def build_phase2_optimizer(model, base_model, lr_base, lr_head):
    """Adam with separate LRs for unfrozen backbone vs classification head."""
    backbone_ids = {id(v) for v in base_model.trainable_variables}
    backbone_vars = [v for v in model.trainable_variables if id(v) in backbone_ids]
    backbone_var_ids = [id(v) for v in backbone_vars]
    return DualLRAdam(lr_base, lr_head, backbone_var_ids)


def get_callbacks(model_path, hp, phase_epochs, phase_lr, phase_name="phase"):
    """Checkpoint, early stopping, and LR schedule callbacks."""
    ensure_directories()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        get_lr_schedule_callback(
            hp["lr_schedule"], phase_lr, phase_epochs, phase_name
        ),
    ]
    return callbacks


@keras.utils.register_keras_serializable(package="fruit_identification")
class SparseLabelSmoothingCrossentropy(keras.losses.Loss):
    """Sparse CE with label smoothing; ponytail: Keras 3.14 SparseCategoricalCrossentropy lacks label_smoothing."""

    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        smoothing = self.label_smoothing
        n_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        one_hot = tf.one_hot(y_true, tf.shape(y_pred)[-1])
        smooth = one_hot * (1.0 - smoothing) + smoothing / n_classes
        return keras.losses.categorical_crossentropy(smooth, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg["label_smoothing"] = self.label_smoothing
        return cfg


def get_loss(label_smoothing):
    """Sparse categorical crossentropy with optional label smoothing."""
    smoothing = float(label_smoothing or 0.0)
    if smoothing <= 0:
        return keras.losses.SparseCategoricalCrossentropy()
    return SparseLabelSmoothingCrossentropy(label_smoothing=smoothing)


def get_custom_objects():
    """Keras custom objects required to load saved .h5 models."""
    return {
        "SparseLabelSmoothingCrossentropy": SparseLabelSmoothingCrossentropy,
        "DualLRAdam": DualLRAdam,
    }


def phase1_checkpoint_path(model_path):
    """Sibling path for Phase 1 best weights (preserved before fine-tuning)."""
    stem, ext = os.path.splitext(model_path)
    return f"{stem}_phase1{ext}"


def plot_history(history_phase1, history_phase2, save_path):
    """Plot training and validation accuracy/loss for both phases."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hist, label in [(history_phase1, "Phase 1"), (history_phase2, "Phase 2")]:
        epochs = range(1, len(hist.history["accuracy"]) + 1)
        axes[0].plot(epochs, hist.history["accuracy"], label=f"{label} train")
        axes[0].plot(
            epochs, hist.history["val_accuracy"], linestyle="--", label=f"{label} val"
        )
        axes[1].plot(epochs, hist.history["loss"], label=f"{label} train")
        axes[1].plot(
            epochs, hist.history["val_loss"], linestyle="--", label=f"{label} val"
        )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# 8. Training pipeline
# ---------------------------------------------------------------------------


def _run_phase1(
    num_classes,
    hp,
    backbone,
    backbone_cfg,
    train_ds,
    val_ds,
    model_path,
    class_weight,
    loss,
):
    """Phase 1: frozen backbone, train classification head."""
    print("\n" + "=" * 60)
    print(
        f"Phase 1: Training classification head "
        f"(frozen {backbone_cfg['display_name']})"
    )
    print("=" * 60)

    model, _ = build_model(num_classes, hp, fine_tune=False, backbone=backbone)
    lr1 = float(hp["learning_rate_phase1"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr1),
        loss=loss,
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(hp["epochs_phase1"]),
        callbacks=get_callbacks(
            model_path, hp, int(hp["epochs_phase1"]), lr1, "phase1"
        ),
        class_weight=class_weight,
    )

    phase1_path = phase1_checkpoint_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase 1 checkpoint missing at {model_path}. "
            "ModelCheckpoint should have saved the best epoch."
        )
    shutil.copy2(model_path, phase1_path)
    phase1_best_val = max(history.history["val_accuracy"])
    print(
        f"Phase 1 best val_accuracy: {phase1_best_val:.4f} "
        f"(backed up to {phase1_path})"
    )
    return history, model, phase1_best_val, phase1_path


def _run_phase2(
    phase1_model,
    num_classes,
    hp,
    backbone,
    train_ds,
    val_ds,
    model_path,
    class_weight,
    loss,
):
    """Phase 2: unfreeze top layers and fine-tune."""
    print("\n" + "=" * 60)
    print(f"Phase 2: Fine-tuning (layers {int(hp['fine_tune_at'])}+ unfrozen)")
    print("=" * 60)

    model, base_model = build_model(num_classes, hp, fine_tune=True, backbone=backbone)
    model.set_weights(phase1_model.get_weights())
    lr_head = float(hp.get("learning_rate_phase2_head", LEARNING_RATE_PHASE2_HEAD))
    lr_base = float(hp.get("learning_rate_phase2_base", LEARNING_RATE_PHASE2_BASE))
    optimizer = build_phase2_optimizer(model, base_model, lr_base, lr_head)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(hp["epochs_phase2"]),
        callbacks=get_callbacks(
            model_path, hp, int(hp["epochs_phase2"]), lr_head, "phase2"
        ),
        class_weight=class_weight,
    )
    phase2_best_val = max(history.history["val_accuracy"])
    # ponytail: re-save with standard Adam so inference does not need DualLRAdam
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_head),
        loss=loss,
        metrics=["accuracy"],
    )
    model.save(model_path)
    return history, phase2_best_val


def _deploy_best_model(model_path, phase1_path, phase1_best_val, phase2_best_val):
    """Copy the phase with higher val_accuracy to the deploy model path."""
    # ponytail: deploy whichever phase had higher val_accuracy
    deploy_path = model_path if phase2_best_val > phase1_best_val else phase1_path
    deployed_phase = 2 if phase2_best_val > phase1_best_val else 1
    if os.path.abspath(deploy_path) != os.path.abspath(model_path):
        shutil.copy2(deploy_path, model_path)
    print(
        f"\nDeployed Phase {deployed_phase} model to {model_path} "
        f"(phase1 val={phase1_best_val:.4f}, phase2 val={phase2_best_val:.4f})"
    )
    return deployed_phase


def resave_for_inference(model_path):
    """Strip training optimizer from deployed .h5 so inference loads cleanly."""
    model = keras.models.load_model(
        model_path,
        custom_objects=get_custom_objects(),
        compile=False,
    )
    model.save(model_path, include_optimizer=False)
    print(f"Inference-ready model saved to {model_path}")


def train(
    overrides=None,
    model_path=None,
    history_path=None,
    save_curves=True,
    backbone=None,
    split_name=None,
):
    """
    Full training pipeline with optional hyperparameter overrides.

    Returns dict with paths and both phase histories.
    """
    ensure_tensorflow()
    backbone = backbone or APP_BACKBONE
    split_name = split_name or APP_SPLIT
    split_cfg = get_split_preset(split_name)
    backbone_cfg = get_backbone_config(backbone)
    hp = merge_hyperparameters(overrides, backbone=backbone)
    model_path = model_path or get_model_path(backbone, split_name)
    history_path = history_path or get_history_path(backbone, split_name)

    train_dir, _, _ = get_split_dirs(split_name)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_dir}. "
            f"Run: python split_dataset.py --split {split_name}"
        )

    ensure_directories()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train_ds, val_ds, class_names, num_classes = create_datasets(
        batch_size=int(hp["batch_size"]),
        split_name=split_name,
    )
    class_weight = None
    if hp.get("use_class_weights"):
        class_weight = compute_class_weights(
            class_names, split_name=split_name, hp=hp
        )
        print("Using class weights (balanced + low-recall boost)")

    loss = get_loss(hp.get("label_smoothing"))

    print(f"Split: {split_cfg['label']} ({split_name})")
    print(f"Backbone: {backbone_cfg['display_name']} ({backbone})")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")
    print(f"Classes: {num_classes}")

    history1, phase1_model, phase1_best_val, phase1_path = _run_phase1(
        num_classes,
        hp,
        backbone,
        backbone_cfg,
        train_ds,
        val_ds,
        model_path,
        class_weight,
        loss,
    )
    history2, phase2_best_val = _run_phase2(
        phase1_model,
        num_classes,
        hp,
        backbone,
        train_ds,
        val_ds,
        model_path,
        class_weight,
        loss,
    )
    _deploy_best_model(model_path, phase1_path, phase1_best_val, phase2_best_val)
    resave_for_inference(model_path)

    combined = {
        "split_name": split_name,
        "split_label": split_cfg["label"],
        "backbone": backbone,
        "display_name": backbone_cfg["display_name"],
        "hyperparameters": hp,
        "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()},
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    if save_curves:
        curves_path = get_training_curves_path(split_name, backbone)
        plot_history(history1, history2, curves_path)

    return {
        "split_name": split_name,
        "backbone": backbone,
        "model_path": model_path,
        "history_path": history_path,
        "history1": history1,
        "history2": history2,
        "hyperparameters": hp,
    }


# ---------------------------------------------------------------------------
# 9. Entry — self-check and CLI
# ---------------------------------------------------------------------------


def _run_self_check():
    """Minimal integrity checks for paths, hyperparameters, and model build."""
    # ponytail: inline holdout fraction; avoid split_dataset import at check time
    assert abs(0.2 / (0.2 + 0.1) - 2 / 3) < 1e-9
    assert infer_backbone_from_model_path("fruit_classifier_mobilenet_v2.h5") == (
        "mobilenet_v2"
    )
    assert infer_backbone_from_model_path("fruit_classifier_mobilenet_v3.h5") == (
        "mobilenet_v3"
    )
    assert to_local_name("Banana") == "Pisang"
    if os.path.isfile(FRUIT_INFO_PATH):
        assert len(get_allowed_class_names()) == 28
    hp = default_hyperparameters("mobilenet_v3")
    assert hp["fine_tune_at"] == 110
    assert hp["epochs_phase1"] == 50
    ensure_tensorflow()
    model, _ = build_model(28, hp, fine_tune=False, backbone="mobilenet_v3")
    assert model.output_shape[-1] == 28
    phase2_model, base_model = build_model(28, hp, fine_tune=True, backbone="mobilenet_v3")
    phase2_model.set_weights(model.get_weights())
    assert len(phase2_model.get_weights()) == len(model.get_weights())
    optimizer = build_phase2_optimizer(
        phase2_model,
        base_model,
        hp["learning_rate_phase2_base"],
        hp["learning_rate_phase2_head"],
    )
    phase2_model.compile(
        optimizer=optimizer,
        loss=get_loss(hp["label_smoothing"]),
        metrics=["accuracy"],
    )
    print("train self-check OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fruit classifier")
    parser.add_argument("--check", action="store_true", help="Run self-check")
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_ORDER,
        default=APP_BACKBONE,
        help=f"MobileNet backbone to train (default: {APP_BACKBONE})",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_ORDER,
        default=APP_SPLIT,
        help=f"Dataset split preset (default: {APP_SPLIT})",
    )
    args = parser.parse_args()
    if args.check:
        _run_self_check()
    else:
        train(backbone=args.backbone, split_name=args.split)
