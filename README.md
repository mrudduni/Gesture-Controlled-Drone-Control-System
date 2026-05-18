# Gesture-Controlled-Drone-Control-System

This project uses computer vision and deep learning to recognize hand gestures and map them to drone control actions (up, down, left, right, hover, land). It includes data collection, augmentation, model training, and a simulation-based control interface.

## Features

- Hand detection and crop extraction using MediaPipe
- Grayscale gesture dataset collection from webcam frames
- Data augmentation pipeline with Augmentor
- CNN training pipeline with TensorFlow/Keras
- Gesture-based drone control simulation UI in OpenCV

## Gesture Classes

- `up`
- `down`
- `left`
- `right`
- `hover`
- `land`

## Project Files

- `Data_Collection.py`: Captures gesture images from webcam into `data/<gesture>/`
- `augmentation.py`: Applies augmentation and writes generated files to `data/<gesture>/output/`
- `cnn for grayscale.py`: Trains a CNN model from dataset folders and saves `gesture_model_final.h5`
- `drone simulation.py`: Loads the trained model and runs real-time gesture-based drone simulation
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Collect data:

```bash
python "Data_Collection.py"
```

2. Augment data:

```bash
python "augmentation.py"
```

3. Train model:

```bash
python "cnn for grayscale.py"
```

4. Run simulation:

```bash
python "drone simulation.py"
```

## Notes

- Trained model files (`.h5`) and generated augmentation output are excluded from git by default.
- Update hardcoded dataset paths in scripts if your local folder layout changes.
