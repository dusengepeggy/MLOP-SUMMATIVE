"""Image loading, dataset building, upload handling and EDA feature extraction."""
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    CLASSES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    ROOT,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    UPLOAD_DIR,
)

EDA_STATS_PATH = ROOT / "docs" / "eda_stats.json"


# ---------------------------------------------------------------- single images
def load_image(source):
    """Load an image from a path, bytes or file-like object.

    Returns float32 array of shape (224, 224, 3) with raw 0..255 pixel values.
    Scaling to [-1, 1] happens inside the model (Rescaling layer), so training
    and prediction share exactly one preprocessing path.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def is_valid_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def list_images(root: Path, cls: str):
    d = Path(root) / cls
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


# ---------------------------------------------------------------- tf.data
def make_datasets(train_dir=TRAIN_DIR, val_split=0.15, batch_size=32, seed=SEED):
    """Train/validation tf.data datasets from <train_dir>/<class>/ folders."""
    import tensorflow as tf

    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(str(train_dir), subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(str(train_dir), subset="validation", **common)
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.cache().prefetch(autotune)


def make_test_dataset(test_dir=TEST_DIR, batch_size=32):
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    return ds, list(ds.file_paths)


def augmentation_layer():
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------- dataset bookkeeping
def class_distribution(root=TRAIN_DIR):
    return {c: len(list_images(root, c)) for c in CLASSES}


def class_weights(root=TRAIN_DIR):
    """Inverse-frequency class weights (keyed by class index) for imbalanced uploads."""
    counts = np.array([max(n, 1) for n in class_distribution(root).values()], dtype=float)
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def pending_upload_count(upload_dir=UPLOAD_DIR):
    return sum(class_distribution(upload_dir).values())


def save_upload(data: bytes, label: str, filename: str, upload_dir=UPLOAD_DIR) -> Path:
    """Save raw upload bytes as a normalised RGB JPEG under uploads/<label>/."""
    if label not in CLASSES:
        raise ValueError(f"Unknown label {label!r}")
    dest_dir = Path(upload_dir) / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem.replace(" ", "_")[:50] or "upload"
    dest = dest_dir / f"{stem}.jpg"
    i = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{i}.jpg"
        i += 1
    with Image.open(io.BytesIO(data)) as img:
        img.convert("RGB").save(dest, quality=95)
    return dest


def merge_uploads_into_train(upload_dir=UPLOAD_DIR, train_dir=TRAIN_DIR):
    """Move every pending upload into the training set. Returns {class: n_moved}."""
    moved = {}
    for c in CLASSES:
        files = list_images(upload_dir, c)
        (Path(train_dir) / c).mkdir(parents=True, exist_ok=True)
        for f in files:
            target = Path(train_dir) / c / f"upload_{f.name}"
            i = 1
            while target.exists():
                target = Path(train_dir) / c / f"upload_{i}_{f.name}"
                i += 1
            shutil.move(str(f), str(target))
        moved[c] = len(files)
    return moved


# ---------------------------------------------------------------- EDA features
def image_features(path):
    """Colour and lighting features for one image (computed at 224x224)."""
    arr = load_image(path)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    with Image.open(path) as img:
        hsv = np.asarray(img.convert("RGB").resize(IMG_SIZE).convert("HSV"), dtype=np.float32)
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "hue": float(hsv[..., 0].mean() * 360 / 255),
        "saturation": float(hsv[..., 1].mean() / 255),
        "green_ratio": float(g.mean() / (r.mean() + g.mean() + b.mean() + 1e-6)),
    }


def feature_table(root=TRAIN_DIR):
    """pandas DataFrame with one row per image and its colour/lighting features."""
    import pandas as pd

    rows = []
    for c in CLASSES:
        for p in list_images(root, c):
            rows.append({"label": c, "path": str(p), **image_features(p)})
    return pd.DataFrame(rows)


def compute_eda_stats(root=TRAIN_DIR, out_path=EDA_STATS_PATH):
    """Write per-image features to JSON so the UI can plot EDA without the dataset."""
    df = feature_table(root).drop(columns=["path"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(df.round(4).to_dict(orient="list")))
    return out_path


if __name__ == "__main__":
    print("train distribution:", class_distribution(TRAIN_DIR))
    print("test distribution:", class_distribution(TEST_DIR))
    print("EDA stats written to", compute_eda_stats())
