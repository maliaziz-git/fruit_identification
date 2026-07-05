"""
Split dataset/raw into train/val/test per class for multiple ratio presets.

Presets: 70:20:10 (app default), 60:20:20, 80:10:10.
Each preset is stored under dataset/splits/<preset>/.
"""

import argparse
import os
import shutil

from sklearn.model_selection import train_test_split

import train


def _rmtree_win(path):
    """Remove a directory tree on Windows (clears read-only files)."""
    import stat

    def _on_rm_error(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_on_rm_error)


def val_fraction_of_holdout(val_ratio, test_ratio):
    """Fraction of holdout assigned to validation (remainder goes to test)."""
    return val_ratio / (val_ratio + test_ratio)


def list_images(class_dir, original_only=False, aug_suffix=None):
    """Return full paths of image files in a class folder."""
    aug_suffix = aug_suffix if aug_suffix is not None else train.OFFLINE_AUG_SUFFIX
    paths = []
    for name in os.listdir(class_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext not in train.IMAGE_EXTENSIONS:
            continue
        if original_only:
            stem, _ = os.path.splitext(name)
            if aug_suffix in stem:
                continue
        paths.append(os.path.join(class_dir, name))
    return sorted(paths)


def copy_files(file_paths, dest_class_dir):
    """Copy image files into destination class folder."""
    os.makedirs(dest_class_dir, exist_ok=True)
    for src in file_paths:
        dest = os.path.join(dest_class_dir, os.path.basename(src))
        if not os.path.exists(dest):
            shutil.copy2(src, dest)


def split_class(class_name, raw_dir, train_dir, val_dir, test_dir, ratios):
    """Split one class folder into train, val, and test subsets."""
    # ponytail: per-class two-step split, not global stratified; OK for folder-per-class
    class_raw = os.path.join(raw_dir, class_name)
    images = list_images(class_raw)
    if len(images) < 3:
        print(f"  Warning: {class_name} has only {len(images)} images; skipping.")
        return None

    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]
    val_frac = val_fraction_of_holdout(val_ratio, test_ratio)

    train_files, holdout_files = train_test_split(
        images,
        train_size=train_ratio,
        random_state=train.RANDOM_SEED,
        shuffle=True,
    )
    val_files, test_files = train_test_split(
        holdout_files,
        train_size=val_frac,
        random_state=train.RANDOM_SEED,
        shuffle=True,
    )

    for subset_dir, files in [
        (train_dir, train_files),
        (val_dir, val_files),
        (test_dir, test_files),
    ]:
        copy_files(files, os.path.join(subset_dir, class_name))

    return {
        "class": class_name,
        "total": len(images),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def clear_split_dirs(train_dir, val_dir, test_dir):
    """Remove existing train/val/test folders for one preset."""
    for d in [train_dir, val_dir, test_dir]:
        _rmtree_win(d)


def split_exists(train_dir):
    """Return True if a split preset already has a train folder."""
    return os.path.isdir(train_dir) and any(os.listdir(train_dir))


def _migrate_flat_split(split_name=None):
    """Copy legacy dataset/train|val|test into splits/70_20_10 if preset missing."""
    split_name = split_name or train.APP_SPLIT
    if split_name != train.APP_SPLIT:
        return False
    train_dir, val_dir, test_dir = train.get_split_dirs(split_name)
    if os.path.isdir(val_dir) and any(os.listdir(val_dir)):
        return False
    flat = {
        "train": os.path.join(train.DATASET_DIR, "train"),
        "val": os.path.join(train.DATASET_DIR, "val"),
        "test": os.path.join(train.DATASET_DIR, "test"),
    }
    if not os.path.isdir(flat["train"]) or not any(os.listdir(flat["train"])):
        return False
    print("  Migrating dataset/train|val|test -> dataset/splits/70_20_10/...")
    preset_base = os.path.join(train.SPLITS_DIR, split_name)
    _rmtree_win(preset_base)
    os.makedirs(preset_base, exist_ok=True)
    for subset, dest in [
        ("train", train_dir),
        ("val", val_dir),
        ("test", test_dir),
    ]:
        shutil.copytree(flat[subset], dest)
    return True


def run_split(split_name=None, force=False):
    """Run stratified split for one preset under dataset/splits/<preset>/."""
    if not os.path.isdir(train.RAW_DIR):
        raise FileNotFoundError(f"Raw dataset not found: {train.RAW_DIR}")

    preset = train.get_split_preset(split_name)
    train_dir, val_dir, test_dir = train.get_split_dirs(split_name)
    class_names = train.get_class_names(train.RAW_DIR)
    train.validate_class_names(class_names)

    print(f"\nPreset {preset['label']} -> {train_dir}")

    if not force and _migrate_flat_split(split_name):
        print("  Migrated existing flat 70:20:10 split.")
        return {"split_name": split_name, "label": preset["label"], "migrated": True}

    if split_exists(train_dir) and not force:
        print("  Split already exists. Use --force to rebuild.")
        return None

    if force:
        print("  Removing existing split directories...")
        clear_split_dirs(train_dir, val_dir, test_dir)

    train.ensure_directories()
    stats = []

    print(
        f"  Splitting ({int(preset['train']*100)}% train / "
        f"{int(preset['val']*100)}% val / "
        f"{int(preset['test']*100)}% test)..."
    )
    for i, name in enumerate(class_names, 1):
        print(f"    [{i}/{len(class_names)}] {name}")
        row = split_class(
            name,
            train.RAW_DIR,
            train_dir,
            val_dir,
            test_dir,
            preset,
        )
        if row:
            stats.append(row)

    if split_name == train.APP_SPLIT:
        print(f"  Class names: {len(class_names)} (from train folders)")

    summary_path = train.get_split_summary_path(split_name)
    os.makedirs(train.OUTPUTS_DIR, exist_ok=True)
    total_train = sum(s["train"] for s in stats)
    total_val = sum(s["val"] for s in stats)
    total_test = sum(s["test"] for s in stats)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset Split Summary ({preset['label']})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Classes: {len(stats)}\n")
        f.write(f"Train: {total_train}  Val: {total_val}  Test: {total_test}\n")
        f.write(f"Total: {total_train + total_val + total_test}\n\n")
        f.write(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8}\n")
        f.write("-" * 60 + "\n")
        for s in stats:
            f.write(
                f"{s['class']:<25} {s['train']:>8} {s['val']:>8} {s['test']:>8}\n"
            )

    print(f"  Train: {total_train}  Val: {total_val}  Test: {total_test}")
    print(f"  Summary: {summary_path}")

    return {
        "split_name": split_name,
        "label": preset["label"],
        "train": total_train,
        "val": total_val,
        "test": total_test,
        "summary_path": summary_path,
    }


def run_all_splits(force=False):
    """Build all configured split presets."""
    results = []
    for split_name in train.SPLIT_ORDER:
        results.append(run_split(split_name=split_name, force=force))
    return [r for r in results if r]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split fruit dataset into train/val/test presets"
    )
    parser.add_argument(
        "--split",
        choices=train.SPLIT_ORDER,
        default=train.APP_SPLIT,
        help=f"Split preset to build (default: {train.APP_SPLIT} for app)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all presets (70:20:10, 60:20:20, 80:10:10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild train/val/test even if they already exist",
    )
    args = parser.parse_args()

    if args.all:
        run_all_splits(force=args.force)
    else:
        run_split(split_name=args.split, force=args.force)
