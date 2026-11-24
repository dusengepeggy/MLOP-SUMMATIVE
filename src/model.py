"""MobileNetV2 transfer-learning model: build, train, evaluate, save, retrain.

Usage:  python -m src.model [--head-epochs N] [--ft-epochs N]
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

from src.config import (
    CLASSES,
    H5_PATH,
    IMG_SIZE,
    META_PATH,
    MODEL_PATH,
    MODELS_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    UPLOAD_DIR,
)
from src import preprocessing as pp
from src.metrics import compute_metrics  # noqa: F401  (re-exported for notebook/back-compat)

FINE_TUNE_AT = 100  # MobileNetV2 has 154 layers; unfreeze the top ~54


def _tf():
    import tensorflow as tf

    return tf


# ---------------------------------------------------------------- build
def build_model(dropout=0.3):
    tf = _tf()
    tf.keras.utils.set_random_seed(SEED)
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    # Keras 3.3's MobileNetV2() takes no `name`, so wrap it in a named functional model.
    # Downstream code always finds it with model.get_layer("backbone"), never by index.
    backbone = tf.keras.Model(base.input, base.output, name="backbone")
    backbone.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = pp.augmentation_layer()(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)  # [0,255] -> [-1,1]
    x = backbone(x, training=False)  # keep BatchNorm in inference mode while fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="bean_mobilenetv2")


def _unfreeze_top(model, fine_tune_at=FINE_TUNE_AT):
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, _tf().keras.layers.BatchNormalization):
            layer.trainable = False


def _compile(model, lr):
    tf = _tf()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


class _ProgressCallback:
    """Factory for a Keras callback that reports (epoch, total_epochs, logs)."""

    @staticmethod
    def make(cb, total, offset=0):
        tf = _tf()

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if cb:
                    cb(offset + epoch + 1, total, {k: float(v) for k, v in (logs or {}).items()})

        return _CB()


# ---------------------------------------------------------------- evaluate
def predict_directory(model, test_dir=TEST_DIR):
    """Return (y_true, y_prob, file_paths) for every image in <test_dir>/<class>/."""
    ds, paths = pp.make_test_dataset(test_dir)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0)
    return y_true, y_prob, paths


def evaluate(model, test_dir=TEST_DIR):
    y_true, y_prob, _ = predict_directory(model, test_dir)
    return compute_metrics(y_true, y_prob)


# ---------------------------------------------------------------- persistence
def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def load_model(path=MODEL_PATH):
    return _tf().keras.models.load_model(path, compile=False)


def inference_model(model):
    """Same trained layers without the augmentation block.

    Keras 3 cannot serialise the Random* augmentation layers to legacy HDF5, and they are
    a no-op at inference anyway, so the .h5 export uses this graph.
    """
    tf = _tf()
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = model.get_layer("rescale")(inputs)
    x = model.get_layer("backbone")(x, training=False)
    x = model.get_layer("gap")(x)
    x = model.get_layer("dropout")(x)
    outputs = model.get_layer("predictions")(x)
    return tf.keras.Model(inputs, outputs, name="bean_mobilenetv2_inference")


def save(model, metrics, extra=None):
    """Back up the current model, then atomically write .keras, .h5 and meta JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    previous = load_meta()
    if MODEL_PATH.exists():
        stamp = previous.get("trained_at", now.isoformat())[:19].replace(":", "").replace("-", "")
        shutil.copy2(MODEL_PATH, MODELS_DIR / f"bean_model_{stamp}.keras")

    tmp_keras = MODELS_DIR / "bean_model.tmp.keras"
    tmp_h5 = MODELS_DIR / "bean_model.tmp.h5"
    model.save(tmp_keras)
    inference_model(model).save(tmp_h5)
    os.replace(tmp_h5, H5_PATH)
    os.replace(tmp_keras, MODEL_PATH)  # replicas reload on mtime change of this file

    meta = {
        "version": int(previous.get("version", 0)) + 1,
        "trained_at": now.isoformat(),
        "classes": CLASSES,
        "img_size": list(IMG_SIZE),
        "train_distribution": pp.class_distribution(TRAIN_DIR),
        "metrics": metrics,
        "previous_metrics": previous.get("metrics"),
        **(extra or {}),
    }
    tmp_meta = MODELS_DIR / "bean_model_meta.tmp.json"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, META_PATH)
    return meta


# ---------------------------------------------------------------- train / retrain
def train(head_epochs=10, ft_epochs=10, progress_cb=None):
    tf = _tf()
    train_ds, val_ds = pp.make_datasets(TRAIN_DIR)
    model = build_model()
    total = head_epochs + ft_epochs
    early = lambda: tf.keras.callbacks.EarlyStopping(  # noqa: E731
        monitor="val_accuracy", patience=4, restore_best_weights=True
    )

    _compile(model, 1e-3)
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=pp.class_weights(TRAIN_DIR),
        callbacks=[early(), _ProgressCallback.make(progress_cb, total)],
    )

    _unfreeze_top(model)
    _compile(model, 1e-5)
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ft_epochs,
        class_weight=pp.class_weights(TRAIN_DIR),
        callbacks=[early(), _ProgressCallback.make(progress_cb, total, offset=head_epochs)],
    )

    metrics = evaluate(model)
    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
    meta = save(
        model,
        metrics,
        extra={"kind": "train", "epochs_run": len(history["loss"]), "history": history},
    )
    return meta


def retrain(epochs=3, progress_cb=None, merge_uploads=True):
    """Fine-tune the current model on the training set plus any pending uploads."""
    tf = _tf()
    moved = pp.merge_uploads_into_train(UPLOAD_DIR, TRAIN_DIR) if merge_uploads else {}
    model = load_model() if MODEL_PATH.exists() else build_model()
    train_ds, val_ds = pp.make_datasets(TRAIN_DIR)

    _unfreeze_top(model)
    _compile(model, 1e-5)
    h = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=pp.class_weights(TRAIN_DIR),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
            _ProgressCallback.make(progress_cb, epochs),
        ],
    )
    metrics = evaluate(model)
    return save(
        model,
        metrics,
        extra={
            "kind": "retrain",
            "epochs_run": len(h.history["loss"]),
            "uploads_merged": moved,
            "history": {k: [float(v) for v in vals] for k, vals in h.history.items()},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=10)
    args = parser.parse_args()
    meta = train(args.head_epochs, args.ft_epochs)
    m = meta["metrics"]
    print(f"\nModel v{meta['version']} saved to {MODEL_PATH} and {H5_PATH}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:10s} {m[k]:.4f}")
    print("  confusion matrix:", m["confusion_matrix"])


if __name__ == "__main__":
    main()
