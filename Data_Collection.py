import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
base_folder="data"
frames_per_gesture=100
img_size=224
gestures= {
    '0': 'up',
    '1': 'down',
    '2': 'left',
    '3': 'right',
    '4': 'hover',
    '5': 'land'
}

for gesture in gestures.values():
    gesture_folder = os.path.join(base_folder, gesture)
    if not os.path.exists(gesture_folder):
        os.makedirs(gesture_folder)

mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_gesture=None
frame_counts= {gesture: 0 for gesture in gestures.values()}
is_collecting= False
margin=40

def get_hand_bbox(hand_landmarks, frame_shape):

    h, w, _ = frame_shape
    # Get all landmark coordinates
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    x_min = int(min(x_coords)) - margin
    y_min = int(min(y_coords)) - margin
    x_max = int(max(x_coords)) + margin
    y_max = int(max(y_coords)) + margin
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    return x_min, y_min, x_max, y_max

def make_square_bbox(x_min, y_min, x_max, y_max, frame_shape):
    h,w, _ =frame_shape
    width = x_max - x_min
    height = y_max - y_min
    size= max(width, height)
    center_x = x_min + width // 2
    center_y = y_min + height // 2
    x_min_sq = center_x - size // 2
    y_min_sq = center_y - size // 2
    x_max_sq = center_x + size // 2
    y_max_sq = center_y + size // 2

    if x_min_sq < 0:
        x_max_sq -= x_min_sq
        x_min_sq = 0
    if y_min_sq < 0:
        y_max_sq -= y_min_sq
        y_min_sq = 0
    if x_max_sq > w:
        x_min_sq -= (x_max_sq - w)
        x_max_sq = w
    if y_max_sq > h:
        y_min_sq -= (y_max_sq - h)
        y_max_sq = h
    
    x_min_sq = max(0, x_min_sq)
    y_min_sq = max(0, y_min_sq)
    x_max_sq = min(w, x_max_sq)
    y_max_sq = min(h, y_max_sq)
    
    return x_min_sq, y_min_sq, x_max_sq, y_max_sq

def save_hand_image(frame, hand_landmarks, gesture_name, count):
    x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape)
    x_min, y_min, x_max, y_max = make_square_bbox(x_min, y_min, x_max, y_max, frame.shape)
    hand_crop = frame[y_min:y_max, x_min:x_max]
    
    if hand_crop.size == 0:
        return False
    hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    hand_resized = cv2.resize(hand_gray, (img_size, img_size))
    filename = os.path.join(base_folder, gesture_name, f"{gesture_name}_{count:04d}.jpg")
    cv2.imwrite(filename, hand_resized)
    
    return True

def draw_ui(frame, hand_detected=False):
    """
    Draw user interface with instructions and status.
    """
    h, w, _ = frame.shape
    
    # Create semi-transparent overlay for top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "HAND GESTURE DATA COLLECTION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Instructions
    instructions = [
        "Press 0-5 to select gesture | SPACE to start/stop | 'q' to quit",
        "0: UP | 1: DOWN | 2: RIGHT | 3: LEFT | 4: HOVER | 5: LAND"
    ]
    
    y_offset = 80
    for instruction in instructions:
        cv2.putText(frame, instruction, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 30
    
    # Current status
    if current_gesture:
        status_color = (0, 255, 0) if is_collecting else (0, 200, 255)
        status_text = "COLLECTING" if is_collecting else "READY"
        
        cv2.putText(frame, f"Gesture: {current_gesture.upper()}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Status: {status_text}", (350, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_counts[current_gesture]}/{frames_per_gesture}", 
                    (650, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    else:
        cv2.putText(frame, "Select a gesture (0-5)", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
    
    # Hand detection status
    hand_status = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, hand_status, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
    
    # Progress bars for all gestures at bottom
    bar_width = 200
    bar_height = 20
    x_start = 300
    y_start = h - 120
    
    cv2.putText(frame, "PROGRESS:", (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    for i, (key, gesture) in enumerate(gestures.items()):
        y_pos = y_start + (i * 30)
        
        # Draw background bar
        cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height),
                     (50, 50, 50), -1)
        
        # Draw progress bar
        progress = frame_counts[gesture] / frames_per_gesture
        progress_width = int(bar_width * progress)
        
        if progress_width > 0:
            color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (x_start, y_pos), 
                         (x_start + progress_width, y_pos + bar_height), color, -1)
        
        # Draw label and count
        label = f"{key}: {gesture.upper()}"
        cv2.putText(frame, label, (x_start - 100, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        count_text = f"{frame_counts[gesture]}/{frames_per_gesture}"
        cv2.putText(frame, count_text, (x_start + bar_width + 10, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

print("="*70)
print("HAND GESTURE DATA COLLECTION SYSTEM")
print("="*70)
print("\nINSTRUCTIONS:")
print("1. Press 0-5 to select which gesture you want to record")
print("2. Position your hand in front of the camera")
print("3. Press SPACE to start/stop collecting frames")
print("4. Hold the gesture steady while collecting")
print("5. The system will automatically detect and crop your hand")
print("6. Press 'q' to quit\n")
print("GESTURES:")
for key, gesture in gestures.items():
    print(f"  {key}: {gesture.upper()}")
print("="*70)
print("\nStarting webcam...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    
    hand_detected = False
    hand_landmarks = None
    
    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
        )
        
        # Draw bounding box
        if current_gesture:
            x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape)
            x_min, y_min, x_max, y_max = make_square_bbox(x_min, y_min, x_max, y_max, frame.shape)
            
            color = (0, 255, 0) if is_collecting else (255, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Show cropped preview
            if x_max > x_min and y_max > y_min:
                hand_crop = frame[y_min:y_max, x_min:x_max]
                if hand_crop.size > 0:
                    # Convert to grayscale and resize for preview
                    hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    hand_preview = cv2.resize(hand_gray, (150, 150))
                    hand_preview = cv2.cvtColor(hand_preview, cv2.COLOR_GRAY2BGR)
                    
                    # Place preview in top-right corner
                    frame[220:370, frame.shape[1]-170:frame.shape[1]-20] = hand_preview
                    cv2.rectangle(frame, (frame.shape[1]-170, 220), 
                                (frame.shape[1]-20, 370), (255, 255, 255), 2)
                    cv2.putText(frame, "Preview", (frame.shape[1]-160, 210),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Collect frame if collecting is active
        if is_collecting and current_gesture and hand_detected:
            if frame_counts[current_gesture] < frames_per_gesture:
                if save_hand_image(frame, hand_landmarks, current_gesture, frame_counts[current_gesture]):
                    frame_counts[current_gesture] += 1
                    print(f"✓ {current_gesture}: {frame_counts[current_gesture]}/{frames_per_gesture}")
                    
                    # Auto-stop when complete
                    if frame_counts[current_gesture] >= frames_per_gesture:
                        is_collecting = False
                        print(f"✅ Completed collecting {current_gesture}!")
    
    # Draw UI
    draw_ui(frame, hand_detected)
    
    # Display
    cv2.imshow("Hand Gesture Data Collection", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nQuitting...")
        break
    
    elif key == ord(' '):
        # Toggle collecting
        if current_gesture:
            if frame_counts[current_gesture] < frames_per_gesture:
                is_collecting = not is_collecting
                status = "started" if is_collecting else "stopped"
                print(f"\nCollection {status} for '{current_gesture}'")
                
                if not is_collecting:
                    print("Press SPACE to resume collecting")
            else:
                print(f"\nAlready collected all frames for '{current_gesture}'!")
        else:
            print("\nPlease select a gesture first (0-5)")
    
    elif chr(key) in gestures:
        # Select gesture
        new_gesture = gestures[chr(key)]
        
        if new_gesture != current_gesture:
            is_collecting = False
            current_gesture = new_gesture
            print(f"\n📌 Selected gesture: {current_gesture.upper()}")
            print(f"   Current progress: {frame_counts[current_gesture]}/{frames_per_gesture}")
            print("   Press SPACE to start collecting")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

# Final summary
print("DATA COLLECTION SUMMARY")

total_collected = 0
for gesture, count in frame_counts.items():
    status = "COMPLETE" if count >= frames_per_gesture else "⚠️  INCOMPLETE"
    print(f"{status} {gesture.upper():10s}: {count}/{frames_per_gesture} frames")
    total_collected += count

print("="*70)
print(f"Total frames collected: {total_collected}/{len(gestures) * frames_per_gesture}")
print(f"Data saved in '{base_folder}/' folder")
print("="*70)