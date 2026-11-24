"""Split the trained Keras model into the TF-free serving artifacts used by the API.

Writes (into models/):
  backbone.tflite     rescale -> fine-tuned MobileNetV2 -> GAP  (LiteRT feature extractor)
  head.npz            trained Dense(3) weights W (1280x3) and b (3)
  features_train.npz  backbone features + labels + relative paths of data/train (for retraining)
  features_test.npz   same for the held-out data/test (for before/after evaluation)
  serving_meta.json   version, metrics on the test set, parity check results

Run with the full TensorFlow environment (requirements.txt):  python scripts/export_serving_model.py
"""
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from src import serving  # noqa: E402
from src.config import CLASSES, ROOT as REPO_ROOT, TEST_DIR, TRAIN_DIR  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import load_meta as load_keras_meta, load_model  # noqa: E402
from src.preprocessing import list_images, load_image  # noqa: E402


def convert_backbone(model):
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = model.get_layer("rescale")(inputs)
    x = model.get_layer("backbone")(x, training=False)
    outputs = model.get_layer("gap")(x)
    feature_model = tf.keras.Model(inputs, outputs, name="bean_feature_extractor")
    with tempfile.TemporaryDirectory() as tmp:
        feature_model.export(tmp)
        tflite = tf.lite.TFLiteConverter.from_saved_model(tmp).convert()
    serving.BACKBONE_PATH.write_bytes(tflite)
    return feature_model


def dataset_features(root):
    backbone = serving.Backbone()
    X, y, paths, imgs = [], [], [], []
    for label, cls in enumerate(CLASSES):
        for p in list_images(root, cls):
            img = load_image(p)
            X.append(backbone.features(img))
            imgs.append(img)
            y.append(label)
            paths.append(p.relative_to(REPO_ROOT).as_posix())
    return np.stack(X), np.array(y), paths, np.stack(imgs)


def main():
    model = load_model()
    print("Converting backbone to TFLite...")
    convert_backbone(model)
    W, b = model.get_layer("predictions").get_weights()
    serving.Head(W, b).save()

    print("Extracting features for data/train and data/test...")
    for root, out in ((TRAIN_DIR, serving.FEATURES_TRAIN_PATH), (TEST_DIR, serving.FEATURES_TEST_PATH)):
        X, y, paths, imgs = dataset_features(root)
        np.savez_compressed(out, X=X.astype(np.float32), y=y, paths=np.array(paths))
        print(f"  {out.name}: {X.shape}")
        if root == TEST_DIR:
            test_imgs, X_test, y_test = imgs, X, y

    # parity: LiteRT backbone + numpy head vs the full Keras model on the test set
    keras_prob = model.predict(test_imgs, batch_size=32, verbose=0)
    serve_prob = serving.Head.load().predict_proba(X_test)
    max_diff = float(np.abs(keras_prob - serve_prob).max())
    same_labels = bool((keras_prob.argmax(1) == serve_prob.argmax(1)).all())
    metrics = compute_metrics(y_test, serve_prob)
    print(f"Parity: max |p_keras - p_serving| = {max_diff:.2e}, identical labels: {same_labels}")
    print(f"Serving metrics on data/test: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} loss={metrics['loss']:.4f}")
    if max_diff > 1e-3 or not same_labels:
        sys.exit("Parity check failed; not writing serving_meta.json")

    keras_meta = load_keras_meta()
    serving.save_meta({
        "version": 1,
        "kind": "export",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_model": {"file": "models/bean_model.h5", "version": keras_meta.get("version"),
                         "trained_at": keras_meta.get("trained_at")},
        "classes": CLASSES,
        "metrics": metrics,
        "previous_metrics": None,
        "parity": {"max_prob_diff": max_diff, "identical_labels": same_labels, "n_images": int(len(y_test))},
    })
    shutil.rmtree(REPO_ROOT / "models" / "__pycache__", ignore_errors=True)
    print("Wrote", serving.SERVING_META_PATH)


if __name__ == "__main__":
    main()
