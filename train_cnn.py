"""Train the RGB gesture classifier.

Two architectures (pick with --arch):
  cnn          small VGG-style CNN trained from scratch (default, ~1.3M params)
  mobilenetv2  MobileNetV2 pretrained on ImageNet (downloads ~9 MB of weights
               once). Trained in two phases: first only the new classifier head
               with the backbone frozen, then the top of the backbone is
               unfrozen and fine-tuned at a low learning rate.

Data sources (both optional, at least one needed):
  data/<class>/*.jpg                       your own webcam captures
  datasets/hagrid_crops/{train,val}/<class>  HaGRID hand crops (prepare_hagrid.py)

Your own captures are recorded as continuous bursts, so neighbouring frames are
near-duplicates. To keep validation honest the last VAL_FRACTION of each class
(by file number) is held out rather than a random sample. HaGRID crops are
already split by person in prepare_hagrid.py.

Augmentation happens on the fly (colour jitter, random grayscale, small
rotation/zoom/shift, and horizontal flips that swap the left/right labels), so
there is no separate augmentation step.

Usage:
  python train_cnn.py                       # CNN, own data + HaGRID crops
  python train_cnn.py --arch mobilenetv2    # pretrained MobileNetV2
  python train_cnn.py --no-hagrid           # own data only
  python train_cnn.py --epochs 40
"""
import argparse
import json
import re
from collections import Counter

import numpy as np

from gesture_common import (ARCHES, CLASS_NAMES, DEFAULT_ARCH, HAGRID_CROPS_DIR, IMG_SIZE, MODEL_DIR,
                            OWN_DATA_DIR, model_paths)

VAL_FRACTION = 0.2
SEED = 42


def own_samples():
    """(path, label, source) lists for train and val from data/<class>/*.jpg."""
    train, val = [], []
    for label, cls in enumerate(CLASS_NAMES):
        folder = OWN_DATA_DIR / cls
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.jpg"), key=lambda p: [int(t) if t.isdigit() else t
                                                            for t in re.split(r"(\d+)", p.name)])
        n_val = int(round(len(files) * VAL_FRACTION))
        cut = len(files) - n_val
        train += [(str(p), label, "own") for p in files[:cut]]
        val += [(str(p), label, "own") for p in files[cut:]]
    return train, val


def hagrid_samples():
    out = {}
    for split in ("train", "val"):
        items = []
        for label, cls in enumerate(CLASS_NAMES):
            folder = HAGRID_CROPS_DIR / split / cls
            if folder.is_dir():
                items += [(str(p), label, "hagrid") for p in sorted(folder.glob("*.jpg"))]
        out[split] = items
    return out["train"], out["val"]


def geometric_augmentation(x):
    """Random rotation/zoom/shift layers; Keras only applies them during training."""
    from tensorflow.keras import layers
    x = layers.RandomRotation(0.04, fill_mode="constant")(x)    # about +-15 degrees
    x = layers.RandomZoom((-0.15, 0.1), fill_mode="constant")(x)
    return layers.RandomTranslation(0.06, 0.06, fill_mode="constant")(x)


def build_cnn(num_classes, img_size):
    """Small VGG-style CNN trained from scratch. Returns (model, None)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def block(x, filters):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        return layers.MaxPooling2D()(x)

    inputs = keras.Input((img_size, img_size, 3))
    x = geometric_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    # Stride-2 stem: halves resolution straight away (128 -> 64). Running the
    # double-conv blocks at full resolution cost ~4x the compute for no real
    # gain on a hand crop.
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for filters in (32, 64, 128, 192):
        x = block(x, filters)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="gesture_cnn_rgb"), None


def build_mobilenetv2(num_classes, img_size):
    """MobileNetV2 (ImageNet weights) with a new classifier head.

    Returns (model, backbone); the backbone starts frozen.
    Input is still raw 0-255 RGB; the model rescales to MobileNetV2's [-1, 1] itself,
    so the simulator feeds both architectures the same way.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    backbone = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    backbone.trainable = False

    inputs = keras.Input((img_size, img_size, 3))
    x = geometric_augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)   # same as mobilenet_v2.preprocess_input
    # training=False keeps BatchNorm statistics frozen, even while fine-tuning.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="gesture_mobilenetv2"), backbone


BUILDERS = {"cnn": build_cnn, "mobilenetv2": build_mobilenetv2}


def make_dataset(samples, img_size, batch, training):
    import tensorflow as tf

    left, right = CLASS_NAMES.index("left"), CLASS_NAMES.index("right")
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)  # grayscale files -> 3 equal channels
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        # uint8 so the in-memory cache stays small (~50 KB per image).
        return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), label

    def to_float(img, label):
        return tf.cast(img, tf.float32), label

    def augment(img, label):
        # Mirror the image; a mirrored "left" is a "right" and vice versa.
        flip = tf.random.uniform(()) < 0.5
        img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
        label = tf.where(flip & tf.equal(label, left), right,
                         tf.where(flip & tf.equal(label, right), left, label))
        x = img / 255.0
        x = tf.image.random_brightness(x, 0.15)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        x = tf.image.random_saturation(x, 0.6, 1.4)
        x = tf.image.random_hue(x, 0.04)
        # Sometimes drop colour, so old grayscale captures and RGB look alike to the model.
        gray = tf.random.uniform(()) < 0.2
        x = tf.cond(gray, lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x)), lambda: x)
        return tf.clip_by_value(x, 0.0, 1.0) * 255.0, label

    # Decode + resize once, keep in RAM, then shuffle/augment fresh every epoch.
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(samples), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def report(model, samples, img_size, batch, title):
    from sklearn.metrics import classification_report, confusion_matrix

    if not samples:
        return None
    probs = model.predict(make_dataset(samples, img_size, batch, training=False), verbose=0)
    y_pred = probs.argmax(1)
    y_true = np.array([s[1] for s in samples])
    acc = float((y_pred == y_true).mean())
    present = sorted(set(y_true) | set(y_pred))
    names = [CLASS_NAMES[i] for i in present]
    print(f"\n=== {title}: {len(samples)} images, accuracy {acc:.3f}")
    print(classification_report(y_true, y_pred, labels=present, target_names=names, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):", names)
    print(confusion_matrix(y_true, y_pred, labels=present))
    return acc


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set(title="Accuracy", xlabel="epoch")
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set(title="Loss", xlabel="epoch")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHES), default=DEFAULT_ARCH)
    parser.add_argument("--epochs", type=int, default=30,
                        help="total epochs (for mobilenetv2: head epochs + fine-tune epochs)")
    parser.add_argument("--head-epochs", type=int, default=8,
                        help="mobilenetv2 only: epochs with the backbone frozen")
    parser.add_argument("--fine-tune-layers", type=int, default=40,
                        help="mobilenetv2 only: how many of the backbone's top layers to unfreeze")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--no-hagrid", action="store_true", help="train on your own captures only")
    parser.add_argument("--own-repeat", type=int, default=3,
                        help="repeat your own training images N times so HaGRID doesn't drown them out")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)

    own_train, own_val = own_samples()
    hg_train, hg_val = ([], []) if args.no_hagrid else hagrid_samples()
    train = own_train * max(1, args.own_repeat) + hg_train
    val = own_val + hg_val
    if not train or not val:
        raise SystemExit("No training data found. Put images in data/<class>/ or run prepare_hagrid.py.")

    print(f"Own data:  {len(own_train)} train / {len(own_val)} val")
    print(f"HaGRID:    {len(hg_train)} train / {len(hg_val)} val")
    counts = Counter(s[1] for s in train)
    print("Train images per class (after repeat):", {CLASS_NAMES[k]: counts[k] for k in sorted(counts)})

    # Balance classes in the loss.
    total = sum(counts.values())
    class_weight = {k: total / (len(counts) * v) for k, v in counts.items()}

    model_path, meta_path, curves_path = model_paths(args.arch)
    model, backbone = BUILDERS[args.arch](len(CLASS_NAMES), args.img_size)
    model.summary(line_length=90)

    MODEL_DIR.mkdir(exist_ok=True)
    train_ds = make_dataset(train, args.img_size, args.batch, training=True)
    val_ds = make_dataset(val, args.img_size, args.batch, training=False)
    class SaveBest(keras.callbacks.Callback):
        """Save the model whenever val accuracy beats every earlier epoch, across all phases.

        (keras.callbacks.ModelCheckpoint can't write .keras files in TF 2.13.)
        """
        best = -1.0

        def on_epoch_end(self, epoch, logs=None):
            acc = (logs or {}).get("val_accuracy", -1.0)
            if acc > self.best:
                print(f"\nEpoch {epoch + 1}: val_accuracy improved {max(self.best, 0):.4f} -> {acc:.4f}, "
                      f"saving {model_path.name}")
                self.best = acc
                self.model.save(model_path)

    checkpoint = SaveBest()

    def fit(lr, epochs, initial_epoch=0, min_lr=1e-5):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model.fit(
            train_ds, validation_data=val_ds, class_weight=class_weight,
            epochs=epochs, initial_epoch=initial_epoch,
            callbacks=[
                checkpoint,  # shared, so it only saves when val accuracy beats every earlier phase
                keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                              restore_best_weights=True, verbose=1),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                                  min_lr=min_lr, verbose=1),
            ],
        )

    if backbone is None:
        history = fit(1e-3, args.epochs)
    else:
        head_epochs = min(args.head_epochs, args.epochs)
        print(f"\nPhase 1: training the classifier head, backbone frozen ({head_epochs} epochs)")
        history = fit(1e-3, head_epochs)
        if args.epochs > head_epochs:
            backbone.trainable = True
            cutoff = len(backbone.layers) - args.fine_tune_layers
            for i, layer in enumerate(backbone.layers):
                # Keep the lower layers and every BatchNorm layer frozen.
                layer.trainable = i >= cutoff and not isinstance(layer, keras.layers.BatchNormalization)
            n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
            print(f"\nPhase 2: fine-tuning the top {args.fine_tune_layers} backbone layers "
                  f"({n_trainable:,} trainable params, {args.epochs - head_epochs} epochs)")
            fine = fit(1e-4, args.epochs, initial_epoch=len(history.history["loss"]), min_lr=1e-6)
            for key, values in fine.history.items():
                history.history.setdefault(key, []).extend(values)

    model = keras.models.load_model(model_path)  # best checkpoint
    overall = report(model, val, args.img_size, args.batch, "Validation (all)")
    own_acc = report(model, own_val, args.img_size, args.batch, "Validation (your captures)")
    hg_acc = report(model, hg_val, args.img_size, args.batch, "Validation (HaGRID, unseen people)")

    meta_path.write_text(json.dumps({
        "arch": args.arch,
        "class_names": CLASS_NAMES,
        "img_size": args.img_size,
        "params": int(model.count_params()),
        "val_accuracy": overall,
        "val_accuracy_own": own_acc,
        "val_accuracy_hagrid": hg_acc,
        "train_images": len(train),
        "val_images": len(val),
        "used_hagrid": not args.no_hagrid,
    }, indent=2))
    save_curves(history, curves_path)
    print(f"\nSaved {model_path.name}, {meta_path.name} and {curves_path.name} in {MODEL_DIR}")


if __name__ == "__main__":
    main()
