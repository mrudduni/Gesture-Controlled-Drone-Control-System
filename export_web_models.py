"""Export the trained models to ONNX for the browser app (web/).

Runs in a separate environment, because tf2onnx needs protobuf 3.20 and
MediaPipe needs protobuf 4.x:

  python -m venv .venv-export
  .venv-export\\Scripts\\pip install -r requirements-export.txt
  .venv-export\\Scripts\\python export_web_models.py

Writes web/models/*.onnx and web/models/manifest.json. Every exported model is
checked against the Keras original on random inputs.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent
WEB_MODELS = ROOT / "web" / "models"
CLASS_NAMES = ["down", "hover", "land", "left", "right", "up"]

MODELS = [
    {
        "id": "legacy",
        "name": "Original CNN",
        "description": "Your first model: grayscale 224x224, trained only on your 600 captures.",
        "source": ROOT / "models" / "legacy" / "gesture_model_final.h5",
        "meta": None,
        "file": "legacy_cnn.onnx",
        "input": {"size": 224, "channels": 1, "scale": "0-1", "crop": "legacy"},
        # 11M of its 11.2M weights are one Dense layer; storing those as uint8
        # shrinks the file ~4x with no visible accuracy change.
        "quantize": True,
    },
    {
        "id": "cnn",
        "name": "New CNN",
        "description": "VGG-style CNN, RGB 128x128, your captures + HaGRID.",
        "source": ROOT / "models" / "gesture_cnn_rgb.keras",
        "meta": ROOT / "models" / "gesture_cnn_rgb.json",
        "file": "cnn_rgb.onnx",
        "input": {"size": 128, "channels": 3, "scale": "0-255", "crop": "padded"},
        "quantize": False,
    },
    {
        "id": "mobilenetv2",
        "name": "MobileNetV2",
        "description": "Pretrained MobileNetV2 fine-tuned on your captures + HaGRID, RGB 128x128.",
        "source": ROOT / "models" / "gesture_mobilenetv2.keras",
        "meta": ROOT / "models" / "gesture_mobilenetv2.json",
        "file": "mobilenetv2.onnx",
        "input": {"size": 128, "channels": 3, "scale": "0-255", "crop": "padded"},
        "quantize": False,
    },
]


def export(entry):
    model = tf.keras.models.load_model(entry["source"], compile=False)
    size, ch = entry["input"]["size"], entry["input"]["channels"]
    spec = (tf.TensorSpec((None, size, size, ch), tf.float32, name="image"),)
    out = WEB_MODELS / entry["file"]
    tmp = out.with_suffix(".fp32.onnx") if entry["quantize"] else out
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(tmp))
    if entry["quantize"]:
        quantize_dynamic(str(tmp), str(out), op_types_to_quantize=["MatMul", "Gemm"],
                         weight_type=QuantType.QUInt8)
        tmp.unlink()

    # Check ONNX against Keras.
    hi = 1.0 if entry["input"]["scale"] == "0-1" else 255.0
    x = (np.random.default_rng(0).random((4, size, size, ch)) * hi).astype(np.float32)
    ref = model(x, training=False).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    diff = float(np.abs(ref - got).max())
    same_top = bool((ref.argmax(1) == got.argmax(1)).all())
    print(f"{entry['id']:12s} -> {out.name}  {out.stat().st_size / 1e6:6.1f} MB  "
          f"max prob diff {diff:.2e}  same top class: {same_top}")

    meta = json.loads(entry["meta"].read_text()) if entry["meta"] and entry["meta"].exists() else {}
    # Prefer compare_models.py's numbers: every model scored on the same images.
    comparison = ROOT / "models" / "comparison.json"
    scores = json.loads(comparison.read_text()).get(entry["id"], {}) if comparison.exists() else {}
    return {
        "id": entry["id"],
        "name": entry["name"],
        "description": entry["description"],
        "file": f"models/{entry['file']}",
        "input_name": sess.get_inputs()[0].name,
        "input": entry["input"],
        "classes": meta.get("class_names", CLASS_NAMES),
        "params": int(model.count_params()),
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "val_accuracy": scores.get("all", meta.get("val_accuracy")),
        "val_accuracy_hagrid": scores.get("hagrid", meta.get("val_accuracy_hagrid")),
        "val_accuracy_own": scores.get("own", meta.get("val_accuracy_own")),
    }


def main():
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in MODELS:
        if not entry["source"].exists():
            print(f"Skipping {entry['id']}: {entry['source']} not found")
            continue
        manifest.append(export(entry))
    (WEB_MODELS / "manifest.json").write_text(json.dumps({"models": manifest}, indent=2))
    print(f"Wrote {WEB_MODELS / 'manifest.json'}")


if __name__ == "__main__":
    main()
