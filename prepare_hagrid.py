"""Download the HaGRID 30k sample and cut out RGB hand crops for training.

Source: https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p
        (HaGRID by Kapitanov et al., CC BY-SA 4.0; ~1 GB, downloaded once)

Output: datasets/hagrid_crops/{train,val}/<drone_gesture>/*.jpg

Class mapping:
  like    -> up
  dislike -> down
  palm    -> hover
  fist    -> land
  one     -> left  (crop rotated 90 deg counter-clockwise: finger points left)
          -> right (crop rotated 90 deg clockwise: finger points right)

The train/val split is by person (HaGRID user_id), so validation measures how
well the model handles people it has never seen.
"""
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter

import cv2
import numpy as np

from gesture_common import CROP_STORE_SIZE, HAGRID_CROPS_DIR, ROOT, crop_square, square_box

URL = ("https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/"
       "resolve/main/hagrid-sample-30k-384p.zip")
ZIP_PATH = ROOT / "datasets" / "hagrid_download" / "hagrid-sample-30k-384p.zip"
PREFIX = "hagrid-sample-30k-384p/"

# HaGRID class -> list of (drone gesture, rotation applied to the crop)
MAPPING = {
    "like": [("up", None)],
    "dislike": [("down", None)],
    "palm": [("hover", None)],
    "fist": [("land", None)],
    "one": [("left", cv2.ROTATE_90_COUNTERCLOCKWISE), ("right", cv2.ROTATE_90_CLOCKWISE)],
}
VAL_PERCENT = 15
MIN_HAND_PIXELS = 28  # skip hands whose box is smaller than this (too blurry to learn from)


def download():
    if ZIP_PATH.exists() and zipfile.is_zipfile(ZIP_PATH):
        return
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ZIP_PATH.with_suffix(".part")
    print(f"Downloading {URL}\n  -> {ZIP_PATH}")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, tmp, reporthook=progress)
    print()
    tmp.replace(ZIP_PATH)


def is_val(user_id):
    return int(hashlib.sha1(user_id.encode()).hexdigest(), 16) % 100 < VAL_PERCENT


def main():
    download()
    if HAGRID_CROPS_DIR.exists():
        shutil.rmtree(HAGRID_CROPS_DIR)

    counts = Counter()
    skipped = Counter()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = set(zf.namelist())
        for hagrid_cls, targets in MAPPING.items():
            ann = json.loads(zf.read(f"{PREFIX}ann_train_val/{hagrid_cls}.json"))
            folder = f"{PREFIX}hagrid_30k/train_val_{hagrid_cls}/"
            images = sorted(n for n in names if n.startswith(folder) and n.lower().endswith(".jpg"))
            print(f"{hagrid_cls:8s} {len(images)} images")

            for name in images:
                image_id = name.rsplit("/", 1)[-1][:-4]
                record = ann.get(image_id)
                if record is None:
                    skipped["no annotation"] += 1
                    continue
                img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    skipped["unreadable"] += 1
                    continue
                h, w = img.shape[:2]
                split = "val" if is_val(record.get("user_id", image_id)) else "train"

                for i, (bbox, label) in enumerate(zip(record["bboxes"], record["labels"])):
                    if label != hagrid_cls:  # e.g. the other hand, labelled no_gesture
                        continue
                    x, y, bw, bh = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
                    if max(bw, bh) < MIN_HAND_PIXELS:
                        skipped["hand too small"] += 1
                        continue
                    crop = crop_square(img, square_box(x, y, x + bw, y + bh))
                    if crop is None:
                        skipped["empty crop"] += 1
                        continue
                    crop = cv2.resize(crop, (CROP_STORE_SIZE, CROP_STORE_SIZE), interpolation=cv2.INTER_AREA)

                    for gesture, rotation in targets:
                        out = crop if rotation is None else cv2.rotate(crop, rotation)
                        out_dir = HAGRID_CROPS_DIR / split / gesture
                        out_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_dir / f"{hagrid_cls}_{image_id}_{i}.jpg"), out,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        counts[(split, gesture)] += 1

    print(f"\nWrote crops to {HAGRID_CROPS_DIR}")
    for split in ("train", "val"):
        row = {g: counts[(split, g)] for g in ["down", "hover", "land", "left", "right", "up"]}
        print(f"  {split:5s} {row}")
    if skipped:
        print("  skipped:", dict(skipped))


if __name__ == "__main__":
    main()
