"""Real-time gesture-controlled drone simulation.

Keys: T takeoff, L land, R reset position, Q quit.
Gestures (once flying): up / down / left / right move the drone while held,
hover holds position, land lands it.

Usage:
  python drone_simulation.py                      # fine-tuned MobileNetV2 (default)
  python drone_simulation.py --arch cnn           # small CNN
"""
import argparse
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from gesture_common import (ARCHES, DEFAULT_ARCH, hand_crop_from_landmarks, load_model_meta, model_paths,
                            to_model_input)

FRAME_W, FRAME_H = 1280, 720
CONFIDENCE_THRESHOLD = 0.70   # mean probability over the smoothing window
HOLD_TIME = 0.4               # seconds a gesture must be stable before it acts
SMOOTHING_WINDOW = 7          # frames used for majority-vote smoothing

COLORS = {
    "background": (30, 30, 30),
    "surface": (50, 50, 50),
    "primary": (88, 214, 141),
    "secondary": (71, 179, 255),
    "danger": (67, 67, 244),
    "success": (80, 175, 76),
    "text": (255, 255, 255),
    "text_dim": (150, 150, 150),
    "drone": (255, 200, 100),
}

ACTION_NAMES = {
    "up": "Move Up", "down": "Move Down", "left": "Move Left",
    "right": "Move Right", "hover": "Hover", "land": "Land",
}


class DroneSimulator:
    """2-D drone: x is left/right (-150..150), y is height (0..200)."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.is_flying = False
        self.battery = 100
        self.speed = 60.0  # units per second
        self.boundary_x = 150
        self.boundary_y = 200
        self.min_flying_height = 20

    def takeoff(self):
        if self.is_flying:
            return "Already flying"
        self.is_flying, self.y = True, 50.0
        return "Takeoff"

    def land(self):
        if not self.is_flying:
            return "Already landed"
        self.is_flying, self.y = False, 0.0
        return "Landed"

    def move(self, dx, dy, dt):
        if not self.is_flying:
            return
        self.x = float(np.clip(self.x + dx * self.speed * dt, -self.boundary_x, self.boundary_x))
        self.y = float(np.clip(self.y + dy * self.speed * dt, self.min_flying_height, self.boundary_y))

    def reset(self):
        self.x = 0.0
        self.y = 50.0 if self.is_flying else 0.0


MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def draw_gesture_icon(img, gesture, center, size, color, thickness=4):
    """Hershey fonts can't draw unicode arrows, so draw the icons as shapes."""
    cx, cy = center
    s = size // 2
    if gesture in MOVES:
        dx, dy = MOVES[gesture]
        start = (cx - dx * s, cy + dy * s)
        end = (cx + dx * s, cy - dy * s)
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.4)
    elif gesture == "hover":
        cv2.circle(img, (cx, cy), s // 2, color, -1)
        cv2.circle(img, (cx, cy), s, color, 2)
    elif gesture == "land":
        cv2.arrowedLine(img, (cx, cy - s), (cx, cy + s // 2), color, thickness, tipLength=0.4)
        cv2.line(img, (cx - s, cy + s), (cx + s, cy + s), color, thickness)


def draw_drone_view(drone, trail, width, height):
    vis = np.full((height, width, 3), COLORS["background"], dtype=np.uint8)
    for i in range(0, width, 40):
        cv2.line(vis, (i, 0), (i, height), (60, 60, 60), 1)
    for i in range(0, height, 40):
        cv2.line(vis, (0, i), (width, i), (60, 60, 60), 1)

    bx, by, bw, bh = 40, 20, width - 100, height - 40
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (80, 80, 80), 2)

    px = int(bx + bw / 2 + (drone.x / drone.boundary_x) * (bw / 2))
    py = int(by + bh - (drone.y / drone.boundary_y) * bh)
    trail.append((px, py))
    for i, (tx, ty) in enumerate(trail):
        a = (i + 1) / len(trail)
        cv2.circle(vis, (tx, ty), 2, tuple(int(c * a) for c in COLORS["primary"]), -1)

    r = 18
    if drone.is_flying:
        cv2.line(vis, (px - r, py - r), (px + r, py + r), COLORS["drone"], 2)
        cv2.line(vis, (px + r, py - r), (px - r, py + r), COLORS["drone"], 2)
        for dx, dy in [(-r, -r), (r, -r), (-r, r), (r, r)]:
            cv2.circle(vis, (px + dx, py + dy), 6, COLORS["primary"], -1)
        cv2.circle(vis, (px, py), 8, COLORS["drone"], -1)
    else:
        cv2.circle(vis, (px, py), 10, (100, 100, 100), -1)
        cv2.circle(vis, (px, py), 10, COLORS["text_dim"], 2)

    # Altitude bar
    ax, ay, ah = width - 30, 30, height - 60
    cv2.rectangle(vis, (ax - 8, ay), (ax + 8, ay + ah), (60, 60, 60), -1)
    fill = int((drone.y / drone.boundary_y) * ah)
    if fill > 0:
        cv2.rectangle(vis, (ax - 8, ay + ah - fill), (ax + 8, ay + ah), COLORS["primary"], -1)
    cv2.putText(vis, f"{drone.y:.0f}m", (ax - 30, ay + ah + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)
    return vis


def draw_ui(frame, drone, trail, gesture, confidence, probs, class_names, active):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.putText(frame, "DRONE GESTURE CONTROL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS["primary"], 2)
    cv2.putText(frame, "SIMULATION", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["secondary"], 1)

    status_color = COLORS["success"] if drone.is_flying else COLORS["text_dim"]
    cv2.circle(frame, (w - 200, 40), 12, status_color, -1)
    cv2.putText(frame, "FLYING" if drone.is_flying else "LANDED", (w - 180, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Battery: {drone.battery}%", (w - 200, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["success"], 1)

    # Right panel
    pw = 420
    px = w - pw
    cv2.rectangle(frame, (px, 90), (w, h), COLORS["background"], -1)
    cv2.line(frame, (px, 90), (px, h), COLORS["primary"], 2)
    cv2.putText(frame, "DRONE VIEW", (px + 20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_dim"], 1)
    frame[120:440, px:w] = draw_drone_view(drone, trail, pw, 320)

    cv2.putText(frame, f"X: {drone.x:+5.0f}   Y: {drone.y:5.0f}", (px + 20, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)

    # Current gesture card
    cy = 490
    cv2.rectangle(frame, (px + 10, cy), (w - 10, cy + 110), COLORS["surface"], -1)
    cv2.putText(frame, "CURRENT GESTURE" + ("  (ACTIVE)" if active else ""), (px + 25, cy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["primary"] if active else COLORS["text_dim"], 1)
    if gesture:
        conf_color = COLORS["success"] if confidence >= CONFIDENCE_THRESHOLD else COLORS["secondary"]
        draw_gesture_icon(frame, gesture, (px + 60, cy + 68), 50, COLORS["primary"])
        cv2.putText(frame, ACTION_NAMES.get(gesture, gesture).upper(), (px + 110, cy + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["text"], 2)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (px + 330, cy + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
    else:
        cv2.putText(frame, "NO HAND DETECTED", (px + 90, cy + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_dim"], 1)

    # Top-3 probabilities
    if probs is not None:
        top = np.argsort(probs)[::-1][:3]
        for i, idx in enumerate(top):
            y = cy + 130 + i * 25
            cv2.putText(frame, class_names[idx], (px + 20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1)
            cv2.rectangle(frame, (px + 100, y), (px + 300, y + 10), (60, 60, 60), -1)
            cv2.rectangle(frame, (px + 100, y), (px + 100 + int(200 * probs[idx]), y + 10),
                          COLORS["primary"] if i == 0 else COLORS["text_dim"], -1)
            cv2.putText(frame, f"{probs[idx] * 100:.0f}%", (px + 310, y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1)

    cv2.putText(frame, "T: Takeoff  |  L: Land  |  R: Reset  |  Q: Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text_dim"], 1)


def load_classifier(arch):
    """Load a trained model plus its class names and input size."""
    from tensorflow import keras  # imported here so --help stays fast

    model_path, meta_path, _ = model_paths(arch)
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path}. Run: python train_cnn.py --arch {arch}")
    class_names, img_size = load_model_meta(meta_path)
    return keras.models.load_model(model_path), class_names, img_size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHES), default=DEFAULT_ARCH)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    args = parser.parse_args()

    model, class_names, img_size = load_classifier(args.arch)
    print(f"Loaded {args.arch} model with classes {class_names}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    drone = DroneSimulator()
    trail = deque(maxlen=20)
    prob_window = deque(maxlen=SMOOTHING_WINDOW)
    stable_gesture, stable_since, land_fired = None, 0.0, False
    last_t = time.time()
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break
        # Fixed canvas size so the UI layout works with any webcam resolution.
        frame = cv2.resize(cv2.flip(frame, 1), (FRAME_W, FRAME_H))
        now = time.time()
        dt, last_t = now - last_t, now

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture, confidence, probs, active = None, 0.0, None, False

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            crop, box = hand_crop_from_landmarks(frame, landmarks)  # crop BEFORE drawing
            if crop is not None:
                probs = model.predict(to_model_input(crop, img_size), verbose=0)[0]
                prob_window.append(probs)
                votes = Counter(int(np.argmax(p)) for p in prob_window)
                idx = votes.most_common(1)[0][0]
                confidence = float(np.mean([p[idx] for p in prob_window]))
                gesture = class_names[idx]

                if gesture != stable_gesture:
                    stable_gesture, stable_since, land_fired = gesture, now, False
                active = confidence >= CONFIDENCE_THRESHOLD and now - stable_since >= HOLD_TIME

                if active and drone.is_flying:
                    if gesture in MOVES:
                        drone.move(*MOVES[gesture], dt)
                    elif gesture == "land" and not land_fired:
                        print(drone.land())
                        land_fired = True

                color = COLORS["success"] if confidence >= CONFIDENCE_THRESHOLD else COLORS["secondary"]
                cv2.rectangle(frame, box[:2], box[2:], color, 3)
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prob_window.clear()
            stable_gesture = None

        draw_ui(frame, drone, trail, gesture, confidence, probs, class_names, active)
        cv2.imshow("Drone Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            print(drone.takeoff())
        elif key == ord("l"):
            print(drone.land())
        elif key == ord("r"):
            drone.reset()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
