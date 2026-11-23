"""Download the Beans dataset (AI-Lab-Makerere/beans) into data/train and data/test.

The Hugging Face train + validation splits go to data/train/<class>/ (we make our own
validation split at training time); the test split goes to data/test/<class>/.

Usage:  python -m src.data_acquisition [--force]
"""
import argparse
import shutil

from src.config import CLASSES, TEST_DIR, TRAIN_DIR

HF_DATASET = "AI-Lab-Makerere/beans"


def count_images(root):
    """Return {class: n_images} for a <root>/<class>/ directory tree."""
    return {
        c: len([p for p in (root / c).glob("*") if p.is_file()]) if (root / c).exists() else 0
        for c in CLASSES
    }


def _already_downloaded():
    return all(v > 0 for v in count_images(TRAIN_DIR).values()) and all(
        v > 0 for v in count_images(TEST_DIR).values()
    )


def _write_split(ds, out_dir, prefix):
    names = ds.features["labels"].names
    for i, row in enumerate(ds):
        cls = names[row["labels"]]
        dest = out_dir / cls
        dest.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(dest / f"{prefix}_{i:05d}.jpg", quality=95)


def download(force=False):
    if _already_downloaded() and not force:
        print("Dataset already present, skipping download (use --force to redo).")
        return
    from datasets import load_dataset

    for d in (TRAIN_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
    ds = load_dataset(HF_DATASET)
    names = ds["train"].features["labels"].names
    assert sorted(names) == sorted(CLASSES), f"Unexpected classes: {names}"
    _write_split(ds["train"], TRAIN_DIR, "train")
    _write_split(ds["validation"], TRAIN_DIR, "val")
    _write_split(ds["test"], TEST_DIR, "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(force=args.force)
    for name, root in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        counts = count_images(root)
        print(f"{name}: {counts}  total={sum(counts.values())}")


if __name__ == "__main__":
    main()
