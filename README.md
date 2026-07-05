# Local Fruit Identification System (CNN)

Final-year project: classify **28 Malaysian local fruit classes** using **MobileNetV3** transfer learning (TensorFlow 2.10 + CUDA GPU on Windows), a **Flask** API backend, and a **React** SaaS-style web UI.

## Project structure

```
fruit_identification/
├── split_dataset.py       # Train/val/test splits (70:20:10, 60:20:20, 80:10:10)
├── train.py               # Paths, hyperparameters, MobileNet V1/V2/V3 training
├── evaluate.py            # Metrics, comparisons, error analysis
├── predict.py             # Inference (CLI + Flask)
├── app.py                 # Flask API + optional React static serve
├── frontend/              # React + Vite SaaS UI (MagicPath-style components)
│   └── src/
├── requirements.txt
├── dataset/
│   ├── raw/               # Original images (28 class folders)
│   └── splits/            # Created by split_dataset.py
│       ├── 70_20_10/      # App default (train / val / test)
│       ├── 60_20_20/
│       └── 80_10_10/
├── models/
│   ├── fruit_classifier_mobilenet_v1.h5
│   ├── fruit_classifier_mobilenet_v2.h5
│   └── fruit_classifier_mobilenet_v3.h5   # used by app.py
├── outputs/               # Reports and plots
│   ├── split/             # split_summary_*.txt
│   ├── training curves/   # training_curves_*_*.png
│   └── reports/           # evaluation reports per model × split
├── data/
│   └── fruit_info.json    # Fruit metadata for GUI and English→local label mapping
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Dataset layout

Place filtered images under `dataset/raw/` with one folder per class:

```
dataset/raw/
├── Bambangan/
├── Pisang/
├── Tembikai/
└── ... (28 folders, ~1000 images each)
```

Class folder names must match the Malaysian **Local Names** (keys) in `data/fruit_info.json`, which also holds GUI metadata (description, nutrition, etc.).

After adding or changing classes, update `data/fruit_info.json`, then:

```powershell
python split_dataset.py --split 70_20_10 --force
```

## Setup

Requires **Python 3.10**, **tensorflow 2.10**, and **CUDA 11.2 + cuDNN 8.1** for GPU training on native Windows (TF 2.10 is the last Windows release with CUDA GPU support).

```powershell
cd c:\fruit_identification
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print('CUDA build:', tf.test.is_built_with_cuda()); print(tf.config.list_physical_devices('GPU'))"
```

The last command should print `2.10.0`, `CUDA build: True`, and at least one `GPU` device. If `GPU` is empty, check CUDA/cuDNN on PATH (including `cudnn64_8.dll` and `zlibwapi.dll` in `CUDA\v11.2\bin`), update your GPU driver, then reinstall: `pip uninstall tensorflow tensorflow-gpu tensorflow-cpu -y` then `pip install -r requirements.txt`.

**Note:** Models saved with TensorFlow 2.21 (`.keras` / Keras 3) are incompatible. Retrain after switching stacks (`python train.py --backbone mobilenet_v3`).

If `py -3.10` is missing, install Python 3.10 from [python.org](https://www.python.org/downloads/release/python-31011/).

## Run pipeline

### 1. Split dataset

Build the **70:20:10** split used by the web app:

```powershell
python split_dataset.py
# same as: python split_dataset.py --split 70_20_10
```

Build all three presets for comparison:

```powershell
python split_dataset.py --all
```

Other presets:

```powershell
python split_dataset.py --split 60_20_20
python split_dataset.py --split 80_10_10
```

Rebuild a preset:

```powershell
python split_dataset.py --split 70_20_10 --force
python split_dataset.py --all --force
```

### 2. Train model

Train the **MobileNetV3** model used by the web app:

```powershell
python train.py
# same as: python train.py --backbone mobilenet_v3
```

Train other backbones for comparison:

```powershell
python train.py --backbone mobilenet_v1
python train.py --backbone mobilenet_v2
```

- **Phase 1:** Frozen base, train classification head  
- **Phase 2:** Unfreeze top layers, fine-tune with lower learning rate  
- Saves `models/fruit_classifier_mobilenet_<v>.h5` and `outputs/training curves/training_curves_<preset>_<v>.png`

Train on a non-default split preset:

```powershell
python train.py --split 60_20_20
```

### 2b. Compare split Macro-Average AUC

Train and evaluate **MobileNetV3** on each split preset, then compare `macro_auc`:

```powershell
python evaluate.py --compare-splits --build-missing-splits --train-missing
```

Train all presets from scratch and compare:

```powershell
python evaluate.py --compare-splits --build-missing-splits --train-all
```

Results: `outputs/split_comparison.json` and `outputs/split_comparison.csv`.

### 2c. Compare model accuracy

Evaluate all trained backbones on the test set and print `accuracy_score`:

```powershell
python evaluate.py --compare-models
```

Train any missing models first, then compare:

```powershell
python evaluate.py --compare-models --train-missing
```

Train all three from scratch and compare:

```powershell
python evaluate.py --compare-models --train-all
```

Results are printed to the console.

### 3. Evaluate on test set

Single model (app default — MobileNetV3 + 70:20:10):

```powershell
python evaluate.py --backbone mobilenet_v3 --split-preset 70_20_10
```

**All models on default split + MobileNetV3 on other splits** (saves reports under `outputs/reports/`):

```powershell
python evaluate.py --matrix
```

Produces per combo:

- `outputs/reports/evaluation_{backbone}_{split}.txt` — full classification report  
- `outputs/reports/confusion_matrix_{backbone}_{split}.png`  
- `outputs/evaluation_all_summary.csv` — overview table

`--compare-models` and `--compare-splits` also save individual reports when run.

**Error analysis** (confused pairs, per-class error rates):

```powershell
python evaluate.py --errors
```

### 4. Web GUI (React + Flask API)

**Development** — run both servers (API on port 5000, UI on port 5173 with proxy):

```powershell
# Terminal 1 — Flask API
python app.py

# Terminal 2 — React dev server
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** — landing page, **Identify** (upload, crop, predict, poster), and **Encyclopedia**.

**Production** — build the frontend, then Flask serves the SPA from `frontend/dist/`:

```powershell
cd frontend
npm run build
cd ..
python app.py
```

Open **http://127.0.0.1:5000** — same React UI, API routes unchanged (`/predict`, `/generate-poster`, `/api/fruits`, `/fruit-sample/...`).

The Flask app always uses **MobileNetV3** on the **70:20:10** split (`fruit_classifier_mobilenet_v3.h5`). Legacy Jinja templates under `templates/` are used only when `frontend/dist/` is not built.

**MagicPath:** Run `npx -y magicpath-ai login` to install designs from [MagicPath](https://www.magicpath.ai) into `frontend/src/components/magicpath/` via `magicpath-ai add`. Until logged in, the bundled SaaS components in that folder are used.

### 5. CLI prediction (optional)

```powershell
python predict.py --image path\to\fruit.jpg
```

## File descriptions

| File | Purpose |
|------|---------|
| `train.py` | Paths, split ratios, hyperparameters, MobileNet V1/V2/V3 two-phase training |
| `split_dataset.py` | Stratified splits: 70:20:10, 60:20:20, 80:10:10 under `dataset/splits/` |
| `evaluate.py` | Test metrics, confusion matrix, model/split comparisons, error analysis |
| `predict.py` | Loads model, returns top-3 predictions and fruit info |
| `app.py` | Flask API (`/predict`, `/generate-poster`, `/api/fruits`) + React static serve when built |
| `frontend/` | React + Vite SaaS UI (`/`, `/identify`, `/encyclopedia`) |
| `poster.py` | Share poster PNG generation |
| `templates/index.html` | Legacy upload UI (used when React build absent) |
| `static/style.css` | Legacy green-themed styling |
| `static/app.js` | Legacy upload, crop (Cropper.js), and prediction UI logic |
| `data/fruit_info.json` | Descriptions, nutrition, season, price keyed by Local Name (English class labels mapped at inference) |

## Model summary

- **Backbones (training/comparison):** MobileNetV1, MobileNetV2, MobileNetV3Large
- **Web app backbone:** MobileNetV3 only (`train.APP_BACKBONE`)
- **Head:** GlobalAveragePooling → Dense(512) → BN → Dropout → Dense(256) → Dropout → Dense(N, softmax) (when `USE_DEEP_HEAD=True`)
- **Preprocessing:** backbone-specific `preprocess_input` baked into each saved model
- **Augmentation:** flip, rotation, zoom, contrast, brightness, translation (training only)

## Training options (`train.py`)

- **Early stopping** and **ModelCheckpoint** on validation metrics
- **LR schedule** via `LR_SCHEDULE`:
  - `plateau` — `ReduceLROnPlateau`
  - `cosine` — cosine decay per phase
  - `cosine_warmup` — warmup then cosine (default)
- **Tiered class weights** — `USE_TIERED_CLASS_WEIGHTS` boosts hard classes (`HARD_CLASS_TIER1`, `HARD_CLASS_TIER2`)

If accuracy plateaus, diagnose confused pairs (`python evaluate.py --errors`), fix raw data, and resplit.

## Notes

- First run `split_dataset.py` before `train.py`.
- Training uses the GPU automatically when CUDA is available (`configure_tensorflow()` in `train.py`).
- The GUI requires a trained model at `models/fruit_classifier_mobilenet_v3.h5`.

---

Final Year Project — CNN Fruit Identification (MobileNetV3)
