"""
Evaluate trained fruit classifiers on validation or test sets.

CLI modes: default single-model eval, --compare-models, --compare-splits,
--matrix (model x split), --errors (misclassification analysis).
"""

import argparse
import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import predict as inference
import train

train.configure_tensorflow()


def load_dataset(split="test", class_names=None, batch_size=None, split_name=None):
    """Load val or test image dataset for a split preset."""
    batch_size = batch_size or train.BATCH_SIZE
    train_dir, val_dir, test_dir = train.get_split_dirs(split_name)
    if split == "val":
        data_dir = val_dir
    elif split == "test":
        data_dir = test_dir
    else:
        raise ValueError("split must be 'val' or 'test'")

    if class_names is None:
        class_names = train.load_class_names()

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=train.IMG_SIZE,
        shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE), class_names


def compute_metrics(y_true, y_pred, y_prob=None):
    """Return dict of overall classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "num_samples": int(len(y_true)),
    }
    if y_prob is not None and len(y_true) > 0:
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["macro_auc"] = None
    else:
        metrics["macro_auc"] = None
    return metrics


def _format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_metric(value, fmt=".4f"):
    return format(value, fmt) if value is not None else "—"


def _print_results_table(title, header, rows, row_fmt):
    print("\n" + "=" * len(header))
    print(title)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row_fmt(row))
    print("=" * len(header))


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix (Test Set)"):
    """Save confusion matrix heatmap."""
    fig_size = max(12, len(class_names) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val < 10:
                continue
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=max(6, min(9, 200 // len(class_names))),
            )
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, save_path, macro_auc=None, title="Macro Average"):
    """Save macro-average one-vs-rest ROC curve."""
    n_classes = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr = {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    if macro_auc is None:
        macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        all_fpr,
        mean_tpr,
        color="#f5c542",
        linewidth=2.5,
        label=f"Macro AUC = {macro_auc:.3f}",
    )
    ax.fill_between(all_fpr, mean_tpr, color="#f5c542", alpha=0.25)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def evaluate_model(
    model_path=None,
    split="test",
    save_report=True,
    save_cm=True,
    split_name=None,
    use_tta=None,
    backbone=None,
):
    """Load model and evaluate on val or test split. Returns metrics dict."""
    model_path = model_path or train.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not backbone:
        backbone = train.infer_backbone_from_model_path(model_path) or train.APP_BACKBONE

    train.ensure_directories()
    class_names = train.load_class_names()
    model = inference.load_model(model_path)
    split_name = split_name or train.APP_SPLIT
    split_cfg = train.get_split_preset(split_name)
    dataset, _ = load_dataset(split=split, class_names=class_names, split_name=split_name)

    use_tta = train.USE_TTA if use_tta is None else use_tta
    print(
        f"Evaluating {model_path} on {split} set ({split_cfg['label']})..."
        + (" [TTA]" if use_tta else "")
    )
    y_true, y_pred, y_prob = inference.collect_predictions(
        model, dataset, use_tta=use_tta
    )
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics.update(
        {
            "split": split,
            "split_name": split_name,
            "split_label": split_cfg["label"],
            "model_path": model_path,
            "backbone": backbone,
            "display_name": train.get_backbone_config(backbone)["display_name"],
        }
    )
    backbone_cfg = train.get_backbone_config(backbone)

    if save_report:
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        lines = [
            f"Model Evaluation Report ({split.upper()} Set)",
            "=" * 60,
            f"Model: {model_path}",
            f"Backbone: {backbone_cfg['display_name']} ({backbone})",
            f"Split preset: {split_cfg['label']} ({split_name})",
            f"Samples: {metrics['num_samples']}",
            "",
            f"Accuracy:           {metrics['accuracy']:.4f}",
            f"Precision (macro):  {metrics['precision_macro']:.4f}",
            f"Recall (macro):     {metrics['recall_macro']:.4f}",
            f"F1-score (macro):   {metrics['f1_macro']:.4f}",
            f"Macro-Average AUC:  {_format_metric(metrics.get('macro_auc'))}",
            "",
            report,
        ]
        report_path = train.get_evaluation_report_path(backbone, split_name, split)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Report saved to {report_path}")

    if save_cm and split == "test":
        cm = confusion_matrix(y_true, y_pred)
        cm_path = train.get_confusion_matrix_path(backbone, split_name)
        title = (
            f"Confusion Matrix — {backbone_cfg['display_name']} "
            f"({split_cfg['label']} test)"
        )
        plot_confusion_matrix(cm, class_names, cm_path, title=title)
        plot_roc_curve(
            y_true,
            y_prob,
            train.get_roc_curve_path(backbone, split_name),
            macro_auc=metrics.get("macro_auc"),
            title="Macro Average",
        )

    print(
        f"  {split} accuracy: {metrics['accuracy']:.4f}  "
        f"macro F1: {metrics['f1_macro']:.4f}  "
        f"macro AUC: {_format_metric(metrics.get('macro_auc'))}"
    )
    return metrics


def compare_models(train_missing=False, split="test"):
    """Evaluate each backbone on APP_SPLIT; optionally train missing models."""
    train.ensure_directories()
    rows = []

    for backbone in train.BACKBONE_ORDER:
        cfg = train.get_backbone_config(backbone)
        model_path = train.get_model_path(backbone)

        if train_missing and not os.path.exists(model_path):
            print(f"\nTraining missing model: {cfg['display_name']}")
            train.train(backbone=backbone, save_curves=True)

        if not os.path.exists(model_path):
            rows.append(
                {
                    "backbone": backbone,
                    "display_name": cfg["display_name"],
                    "model_path": model_path,
                    "split": split,
                    "accuracy": None,
                    "status": "missing",
                }
            )
            continue

        print(f"\nEvaluating {cfg['display_name']}...")
        metrics = evaluate_model(
            model_path=model_path,
            split=split,
            save_report=True,
            save_cm=(split == "test"),
            split_name=train.APP_SPLIT,
            backbone=backbone,
        )
        rows.append(
            {
                "backbone": backbone,
                "display_name": cfg["display_name"],
                "model_path": model_path,
                "split": split,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "num_samples": metrics["num_samples"],
                "status": "ok",
            }
        )

    header = f"{'Model':<16} {'Backbone':<14} {'Accuracy':>10} {'Status':>10}"
    _print_results_table(
        f"Model accuracy comparison ({split} set)",
        header,
        rows,
        lambda r: (
            f"{r['display_name']:<16} {r['backbone']:<14} "
            f"{_fmt_metric(r.get('accuracy')):>10} {r['status']:>10}"
        ),
    )
    return rows


def compare_splits(
    train_missing=False,
    eval_split="test",
    backbone=None,
    build_missing_splits=False,
    save=True,
):
    """Evaluate each split preset; compare Macro-Average AUC."""
    from split_dataset import run_split, split_exists
    backbone = backbone or train.APP_BACKBONE
    backbone_cfg = train.get_backbone_config(backbone)
    train.ensure_directories()
    rows = []

    for split_name in train.SPLIT_ORDER:
        split_cfg = train.get_split_preset(split_name)
        train_dir, _, _ = train.get_split_dirs(split_name)
        model_path = train.get_model_path(backbone, split_name)

        if build_missing_splits and not split_exists(train_dir):
            print(f"\nBuilding split preset: {split_cfg['label']}")
            run_split(split_name=split_name, force=False)

        if not split_exists(train_dir):
            rows.append(
                {
                    "split_name": split_name,
                    "split_label": split_cfg["label"],
                    "backbone": backbone,
                    "display_name": backbone_cfg["display_name"],
                    "eval_split": eval_split,
                    "macro_auc": None,
                    "accuracy": None,
                    "status": "split_missing",
                    "model_path": model_path,
                }
            )
            continue

        if train_missing and not os.path.exists(model_path):
            print(f"\nTraining {backbone_cfg['display_name']} on {split_cfg['label']}...")
            train.train(backbone=backbone, split_name=split_name, save_curves=True)

        if not os.path.exists(model_path):
            rows.append(
                {
                    "split_name": split_name,
                    "split_label": split_cfg["label"],
                    "backbone": backbone,
                    "display_name": backbone_cfg["display_name"],
                    "eval_split": eval_split,
                    "macro_auc": None,
                    "accuracy": None,
                    "status": "model_missing",
                    "model_path": model_path,
                }
            )
            continue

        print(f"\nEvaluating {split_cfg['label']} on {eval_split} set...")
        metrics = evaluate_model(
            model_path=model_path,
            split=eval_split,
            save_report=True,
            save_cm=(eval_split == "test"),
            split_name=split_name,
            backbone=backbone,
        )
        rows.append(
            {
                "split_name": split_name,
                "split_label": split_cfg["label"],
                "backbone": backbone,
                "display_name": backbone_cfg["display_name"],
                "eval_split": eval_split,
                "macro_auc": metrics.get("macro_auc"),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "num_samples": metrics.get("num_samples"),
                "status": "ok",
                "model_path": model_path,
            }
        )

    header = (
        f"{'Split':<12} {'Preset':<14} {'Macro AUC':>12} {'Accuracy':>12} {'Status':>10}"
    )
    _print_results_table(
        f"Split Macro-Average AUC comparison ({backbone_cfg['display_name']}, {eval_split} set)",
        header,
        rows,
        lambda r: (
            f"{r['split_label']:<12} {r['split_name']:<14} "
            f"{_fmt_metric(r.get('macro_auc')):>12} "
            f"{_fmt_metric(r.get('accuracy')):>12} {r['status']:>10}"
        ),
    )

    if save:
        with open(train.SPLIT_COMPARISON_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        with open(train.SPLIT_COMPARISON_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split_label",
                    "split_name",
                    "macro_auc",
                    "accuracy",
                    "f1_macro",
                    "num_samples",
                    "backbone",
                    "display_name",
                    "eval_split",
                    "status",
                    "model_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved comparison to {train.SPLIT_COMPARISON_PATH}")
        print(f"Saved comparison to {train.SPLIT_COMPARISON_CSV}")

    return rows


def evaluate_matrix(
    eval_split="test",
    use_tta=False,
    backbones=None,
    split_names=None,
    save=True,
):
    """Evaluate backbone/split combos; write per-combo reports and summary."""
    split_names = split_names or train.SPLIT_ORDER
    train.ensure_directories()
    rows = []

    for split_name in split_names:
        split_cfg = train.get_split_preset(split_name)
        split_backbones = backbones or train.get_eval_all_backbones(split_name)
        for backbone in split_backbones:
            cfg = train.get_backbone_config(backbone)
            model_path = train.get_model_path(backbone, split_name)

            if not os.path.exists(model_path):
                rows.append(
                    {
                        "backbone": backbone,
                        "display_name": cfg["display_name"],
                        "split_name": split_name,
                        "split_label": split_cfg["label"],
                        "eval_split": eval_split,
                        "accuracy": None,
                        "macro_auc": None,
                        "status": "model_missing",
                        "model_path": model_path,
                        "report_path": "",
                        "confusion_matrix_path": "",
                    }
                )
                continue

            print(
                f"\nEvaluating {cfg['display_name']} on {split_cfg['label']} "
                f"({eval_split})..."
            )
            metrics = evaluate_model(
                model_path=model_path,
                split=eval_split,
                save_report=True,
                save_cm=(eval_split == "test"),
                split_name=split_name,
                use_tta=use_tta,
                backbone=backbone,
            )
            report_path = train.get_evaluation_report_path(
                backbone, split_name, eval_split
            )
            cm_path = (
                train.get_confusion_matrix_path(backbone, split_name)
                if eval_split == "test"
                else ""
            )
            rows.append(
                {
                    "backbone": backbone,
                    "display_name": cfg["display_name"],
                    "split_name": split_name,
                    "split_label": split_cfg["label"],
                    "eval_split": eval_split,
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "macro_auc": metrics.get("macro_auc"),
                    "num_samples": metrics.get("num_samples"),
                    "status": "ok",
                    "model_path": model_path,
                    "report_path": report_path,
                    "confusion_matrix_path": cm_path,
                }
            )

    header = (
        f"{'Model':<14} {'Split':<12} {'Accuracy':>10} {'Macro AUC':>12} {'Status':>14}"
    )
    _print_results_table(
        f"Evaluation summary ({eval_split} set) — model × split",
        header,
        rows,
        lambda r: (
            f"{r['display_name']:<14} {r['split_label']:<12} "
            f"{_fmt_metric(r.get('accuracy')):>10} "
            f"{_fmt_metric(r.get('macro_auc')):>12} {r['status']:>14}"
        ),
    )

    if save:
        with open(train.EVAL_ALL_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        with open(train.EVAL_ALL_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "display_name",
                    "backbone",
                    "split_label",
                    "split_name",
                    "eval_split",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "macro_auc",
                    "f1_macro",
                    "num_samples",
                    "status",
                    "report_path",
                    "confusion_matrix_path",
                    "model_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary: {train.EVAL_ALL_SUMMARY_JSON}")
        print(f"Summary: {train.EVAL_ALL_SUMMARY_CSV}")
        print(f"Reports:  {train.EVAL_REPORTS_DIR}")

    return rows


def print_error_analysis(
    model_path=None,
    split_name=None,
    split="test",
    top_n=20,
):
    """Print confused pairs and per-class error rates (console only)."""
    model_path = model_path or train.MODEL_PATH
    split_name = split_name or train.APP_SPLIT

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    class_names = train.load_class_names()
    model = inference.load_model(model_path)
    dataset, _ = load_dataset(
        split=split, class_names=class_names, split_name=split_name
    )

    print(f"Analyzing errors: {model_path} ({split}, {split_name})")
    y_true, y_pred, _ = inference.collect_predictions(model, dataset)

    pair_counts = Counter()
    class_errors = Counter()
    class_totals = Counter()

    for true_idx, pred_idx in zip(y_true, y_pred):
        class_totals[int(true_idx)] += 1
        if true_idx != pred_idx:
            pair_counts[(int(true_idx), int(pred_idx))] += 1
            class_errors[int(true_idx)] += 1

    print(f"\nTop {top_n} confused pairs (true -> predicted):")
    print("-" * 60)
    for (true_idx, pred_idx), count in pair_counts.most_common(top_n):
        print(f"  {class_names[true_idx]:<20} -> {class_names[pred_idx]:<20}  {count}")

    print("\nHighest error-rate classes:")
    print("-" * 60)
    error_rows = [
        (class_names[idx], class_errors.get(idx, 0), total, class_errors.get(idx, 0) / total)
        for idx, total in class_totals.items()
    ]
    error_rows.sort(key=lambda r: r[3], reverse=True)
    for name, errors, total, rate in error_rows[:15]:
        print(f"  {name:<20} {errors:>4}/{total:<4}  {rate * 100:5.1f}%")

    return {
        "pair_counts": pair_counts,
        "class_errors": class_errors,
        "class_totals": class_totals,
        "y_true": y_true,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fruit classifier")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare accuracy across MobileNet V1/V2/V3",
    )
    mode.add_argument(
        "--compare-splits",
        action="store_true",
        help="Compare Macro-Average AUC across split presets",
    )
    mode.add_argument(
        "--matrix",
        action="store_true",
        help="Full model × split evaluation matrix",
    )
    mode.add_argument(
        "--errors",
        action="store_true",
        help="Analyze misclassifications (console only)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .h5 model (default: train.MODEL_PATH)",
    )
    parser.add_argument(
        "--split-preset",
        choices=train.SPLIT_ORDER,
        default=train.APP_SPLIT,
        dest="split_name",
        help=f"Dataset ratio preset (default: {train.APP_SPLIT})",
    )
    parser.add_argument(
        "--backbone",
        choices=train.BACKBONE_ORDER,
        default=None,
        help="Backbone tag for report filename (optional)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--train-missing",
        action="store_true",
        help="(compare modes) Train missing models before evaluation",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="(compare modes) Train all models/presets before evaluation",
    )
    parser.add_argument(
        "--build-missing-splits",
        action="store_true",
        help="(compare-splits) Build any split preset not yet created",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help="(compare-splits) Subset to score Macro-AUC on",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="(errors) Number of confused pairs to show",
    )
    args = parser.parse_args()

    if args.compare_models:
        if args.train_all:
            for backbone in train.BACKBONE_ORDER:
                cfg = train.get_backbone_config(backbone)
                print(f"\nTraining {cfg['display_name']}...")
                train.train(backbone=backbone, save_curves=True)
        compare_models(train_missing=args.train_missing, split=args.split)
    elif args.compare_splits:
        backbone = args.backbone or train.APP_BACKBONE
        if args.build_missing_splits or args.train_all:
            from split_dataset import run_split, split_exists

            for split_name in train.SPLIT_ORDER:
                train_dir, _, _ = train.get_split_dirs(split_name)
                if not split_exists(train_dir):
                    run_split(split_name=split_name, force=False)
        if args.train_all:
            for split_name in train.SPLIT_ORDER:
                cfg = train.get_split_preset(split_name)
                print(f"\nTraining on {cfg['label']}...")
                train.train(backbone=backbone, split_name=split_name, save_curves=True)
        compare_splits(
            train_missing=args.train_missing,
            eval_split=args.eval_split,
            backbone=backbone,
            build_missing_splits=False,
        )
    elif args.matrix:
        evaluate_matrix(eval_split=args.split, use_tta=args.tta)
    elif args.errors:
        print_error_analysis(
            model_path=args.model,
            split_name=args.split_name,
            split=args.split,
            top_n=args.top,
        )
    else:
        model_path = args.model or train.get_model_path(args.backbone)
        evaluate_model(
            model_path=model_path,
            split=args.split,
            split_name=args.split_name,
            use_tta=args.tta,
            backbone=args.backbone,
        )
