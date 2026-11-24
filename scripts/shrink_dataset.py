"""Resize data/train and data/test images to at most 256 px (JPEG q=90).

Originals are first copied to data/_fullres/ (git-ignored), so nothing is lost.

The model works at 224x224, so 256 px keeps all the information it uses while making the
explicit train/test splits small enough (~20 MB) to commit to the repository.
Re-download the full-size originals any time with `python -m src.data_acquisition --force`.

Usage: python scripts/shrink_dataset.py [--size 256]
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CLASSES, TEST_DIR, TRAIN_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    total = shrunk = 0
    backup_root = TRAIN_DIR.parent / "_fullres"  # git-ignored copy of the originals
    for root in (TRAIN_DIR, TEST_DIR):
        for cls in CLASSES:
            for p in sorted((root / cls).glob("*.jpg")):
                total += 1
                with Image.open(p) as img:
                    if max(img.size) <= args.size:
                        continue
                    backup = backup_root / root.name / cls / p.name
                    if not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        backup.write_bytes(p.read_bytes())
                    img = img.convert("RGB")
                    img.thumbnail((args.size, args.size), Image.LANCZOS)
                    img.save(p, quality=90)
                    shrunk += 1
    size_mb = sum(p.stat().st_size for r in (TRAIN_DIR, TEST_DIR) for p in r.rglob("*.jpg")) / 1e6
    print(f"{shrunk}/{total} images resized; dataset is now {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
