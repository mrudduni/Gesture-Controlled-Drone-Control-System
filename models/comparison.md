Validation images: 120 of your captures, 1474 HaGRID crops of unseen people.

| Model | Your captures | HaGRID (unseen people) | All | Macro F1 | Params | File | CPU latency |
|---|---|---|---|---|---|---|---|
| Original CNN (grayscale 224) | 100.0% | 24.9% | 30.6% | 0.298 | 11.17M | 134.1 MB | 39.1 ms |
| New CNN (RGB 128) | 84.2% | 99.7% | 98.6% | 0.985 | 1.30M | 15.7 MB | 52.2 ms |
| MobileNetV2 (RGB 128) | 98.3% | 99.6% | 99.5% | 0.995 | 2.27M | 23.0 MB | 206.9 ms |

Per-class accuracy (all validation images):

| Model | down | hover | land | left | right | up |
|---|---|---|---|---|---|---|
| Original CNN (grayscale 224) | 43% | 26% | 11% | 43% | 34% | 25% |
| New CNN (RGB 128) | 100% | 100% | 92% | 100% | 99% | 100% |
| MobileNetV2 (RGB 128) | 100% | 100% | 100% | 100% | 100% | 98% |

The original CNN was trained on all of `data/`, including the images used here as "your captures", so that column flatters it.
