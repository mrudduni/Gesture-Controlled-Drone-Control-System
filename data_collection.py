"""Collect hand-gesture crops from the webcam into data/<gesture>/.

Keys: 0-5 pick a gesture, SPACE starts/stops recording, q quits.
Crops are saved in colour (RGB as JPEG) and *before* anything is drawn on the
frame, so the saved images contain only the hand, no landmarks or boxes.
Existing images are never overwritten: numbering continues from the highest
file already in each folder.
"""
import re
import time

import cv2
import mediapipe as mp

from gesture_common import CROP_STORE_SIZE, OWN_DATA_DIR, hand_crop_from_landmarks

FRAMES_PER_SESSION = 100      # how many new frames to record per gesture per run
SAVE_EVERY_SECONDS = 0.1      # spacing between saved frames, so they aren't near-duplicates

GESTURES = {
    "0": "up",
    "1": "down",
    "2": "left",
    "3": "right",
    "4": "hover",
    "5": "land",
}


def next_index(folder, gesture):
    """Index after the highest existing <gesture>_NNNN.jpg in folder."""
    pattern = re.compile(rf"^{gesture}_(\d+)\.jpg$")
    indices = [int(m.group(1)) for p in folder.glob("*.jpg") if (m := pattern.match(p.name))]
    return max(indices) + 1 if indices else 0


def draw_ui(frame, current, collecting, hand_detected, session_counts, totals):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "HAND GESTURE DATA COLLECTION", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "0-5: pick gesture | SPACE: start/stop | q: quit", (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, " | ".join(f"{k}:{g.upper()}" for k, g in GESTURES.items()), (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if current:
        color = (0, 255, 0) if collecting else (0, 200, 255)
        state = "RECORDING" if collecting else "READY"
        cv2.putText(frame, f"{current.upper()}  {state}  {session_counts[current]}/{FRAMES_PER_SESSION}"
                    f"  (total {totals[current]})", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(frame, "Select a gesture (0-5)", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    status = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
    cv2.putText(frame, status, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if hand_detected else (0, 0, 255), 2)


def main():
    folders = {g: OWN_DATA_DIR / g for g in GESTURES.values()}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    next_idx = {g: next_index(folders[g], g) for g in GESTURES.values()}
    totals = {g: len(list(folders[g].glob("*.jpg"))) for g in GESTURES.values()}
    session_counts = {g: 0 for g in GESTURES.values()}

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam 0.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current, collecting, last_save = None, False, 0.0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break
        frame = cv2.flip(frame, 1)  # mirror view (the simulation mirrors too)
        clean = frame.copy()        # untouched copy for saving
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand_detected = bool(results.multi_hand_landmarks)
        if hand_detected:
            landmarks = results.multi_hand_landmarks[0]
            crop, box = hand_crop_from_landmarks(clean, landmarks)

            now = time.time()
            if (collecting and current and crop is not None
                    and now - last_save >= SAVE_EVERY_SECONDS):
                crop = cv2.resize(crop, (CROP_STORE_SIZE, CROP_STORE_SIZE), interpolation=cv2.INTER_AREA)
                path = folders[current] / f"{current}_{next_idx[current]:04d}.jpg"
                cv2.imwrite(str(path), crop)
                next_idx[current] += 1
                session_counts[current] += 1
                totals[current] += 1
                last_save = now
                if session_counts[current] >= FRAMES_PER_SESSION:
                    collecting = False
                    print(f"Done: {current} ({session_counts[current]} new frames)")

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0) if collecting else (255, 255, 0), 2)
            if crop is not None:
                h, w = frame.shape[:2]
                preview = cv2.resize(crop, (150, 150))
                frame[140:290, w - 170:w - 20] = preview
                cv2.rectangle(frame, (w - 170, 140), (w - 20, 290), (255, 255, 255), 2)

        draw_ui(frame, current, collecting, hand_detected, session_counts, totals)
        cv2.imshow("Hand Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and current:
            if session_counts[current] < FRAMES_PER_SESSION:
                collecting = not collecting
                print(f"{'Recording' if collecting else 'Paused'}: {current}")
        elif key != 255 and chr(key) in GESTURES:
            new = GESTURES[chr(key)]
            if new != current:
                current, collecting = new, False
                print(f"Selected {current} ({totals[current]} images so far)")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\nSUMMARY (new this session / total in folder)")
    for g in GESTURES.values():
        print(f"  {g:6s} {session_counts[g]:4d} / {totals[g]}")
    print(f"Saved under {OWN_DATA_DIR}")


if __name__ == "__main__":
    main()
