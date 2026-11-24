"""Compare the original grayscale CNN, the new RGB CNN and MobileNetV2.

All models are scored on the same held-out images:
  * your captures: the last 20% of each class in data/ (see train_cnn.py).
    NOTE: the original model was trained on *all* of data/, so these images
    are not unseen for it and its score here is optimistic.
  * HaGRID validation crops: people none of the models trained on.

Also reports parameters, file size and single-image CPU latency.

Usage:
  python compare_models.py
  python compare_models.py --legacy path/to/gesture_model_final.h5
Writes models/comparison.md.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from gesture_common import CLASS_NAMES, MODEL_DIR, ROOT, load_model_meta, model_paths
from train_cnn import hagrid_samples, make_dataset, own_samples

DEFAULT_LEGACY = ROOT / "models" / "legacy" / "gesture_model_final.h5"
LEGACY_SIZE = 224


def legacy_batch(samples):
    """Original preprocessing: grayscale, 224x224, scaled to 0-1, shape (N, 224, 224, 1)."""
    out = np.empty((len(samples), LEGACY_SIZE, LEGACY_SIZE, 1), np.float32)
    for i, (path, _, _) in enumerate(samples):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        out[i, ..., 0] = cv2.resize(gray, (LEGACY_SIZE, LEGACY_SIZE), interpolation=cv2.INTER_AREA) / 255.0
    return out


def predict(model, samples, kind, img_size):
    if not samples:
        return np.array([], int)
    if kind == "legacy":
        probs = model.predict(legacy_batch(samples), batch_size=32, verbose=0)
    else:
        probs = model.predict(make_dataset(samples, img_size, 32, training=False), verbose=0)
    return probs.argmax(1)


def latency_ms(model, shape, runs=50):
    x = np.random.rand(1, *shape).astype(np.float32) * (1 if shape[-1] == 1 else 255)
    for _ in range(5):
        model(x, training=False)
    t = time.perf_counter()
    for _ in range(runs):
        model(x, training=False)
    return (time.perf_counter() - t) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY,
                        help="original grayscale model (.h5)")
    args = parser.parse_args()

    from sklearn.metrics import f1_score
    from tensorflow import keras

    _, own_val = own_samples()
    _, hg_val = hagrid_samples()
    sets = {"Your captures": own_val, "HaGRID (unseen people)": hg_val, "All": own_val + hg_val}

    candidates = [("Original CNN (grayscale 224)", "legacy", args.legacy, LEGACY_SIZE)]
    for arch, label in [("cnn", "New CNN (RGB 128)"), ("mobilenetv2", "MobileNetV2 (RGB 128)")]:
        path, meta, _ = model_paths(arch)
        candidates.append((label, arch, path, load_model_meta(meta)[1]))

    rows = []
    for label, kind, path, size in candidates:
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue
        print(f"Evaluating {label} ...")
        model = keras.models.load_model(path, compile=False)
        row = {"id": kind, "model": label, "params": model.count_params(), "size_mb": path.stat().st_size / 1e6}
        for set_name, samples in sets.items():
            y_true = np.array([s[1] for s in samples])
            y_pred = predict(model, samples, kind, size)
            row[set_name] = float((y_pred == y_true).mean()) if len(samples) else float("nan")
            if set_name == "All":
                row["macro_f1"] = f1_score(y_true, y_pred, average="macro")
                row["per_class"] = {c: float((y_pred[y_true == i] == i).mean())
                                    for i, c in enumerate(CLASS_NAMES) if (y_true == i).any()}
        channels = 1 if kind == "legacy" else 3
        row["latency_ms"] = latency_ms(model, (size, size, channels))
        rows.append(row)

    lines = [
        f"Validation images: {len(own_val)} of your captures, {len(hg_val)} HaGRID crops of unseen people.",
        "",
        "| Model | Your captures | HaGRID (unseen people) | All | Macro F1 | Params | File | CPU latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['Your captures']:.1%} | {r['HaGRID (unseen people)']:.1%} | "
                     f"{r['All']:.1%} | {r['macro_f1']:.3f} | {r['params'] / 1e6:.2f}M | "
                     f"{r['size_mb']:.1f} MB | {r['latency_ms']:.1f} ms |")
    lines += ["", "Per-class accuracy (all validation images):", "",
              "| Model | " + " | ".join(CLASS_NAMES) + " |",
              "|---|" + "---|" * len(CLASS_NAMES)]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(f"{r['per_class'].get(c, float('nan')):.0%}"
                                                      for c in CLASS_NAMES) + " |")
    lines += ["", "The original CNN was trained on all of `data/`, including the images used here "
              "as \"your captures\", so that column flatters it."]
    text = "\n".join(lines)
    print("\n" + text)
    (MODEL_DIR / "comparison.md").write_text(text + "\n")
    # Read by export_web_models.py so the web app shows the same numbers.
    (MODEL_DIR / "comparison.json").write_text(json.dumps({
        r["id"]: {"own": r["Your captures"], "hagrid": r["HaGRID (unseen people)"], "all": r["All"],
                  "macro_f1": r["macro_f1"], "latency_ms": r["latency_ms"], "per_class": r["per_class"]}
        for r in rows}, indent=2))
    print(f"\nWrote {MODEL_DIR / 'comparison.md'} and comparison.json")


if __name__ == "__main__":
    main()
