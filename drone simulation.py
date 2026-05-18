import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
import time
from collections import deque

MODEL_PATH = "gesture_model_final.h5"
IMG_SIZE = 224
SIMULATION_MODE = True  

class_names = ['down', 'hover', 'land', 'left', 'right', 'up']
print(f"Loaded {len(class_names)} gesture classes: {class_names}")

try:
    model = load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except:
    print(f" Error: Could not load model from {MODEL_PATH}")
    exit(1)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Drone state
class DroneSimulator:
    def __init__(self):
        self.x = 0  # -100 to 100 (left/right)
        self.y = 0  # 0 to 200 (height)
        self.z = 0  # -100 to 100 (forward/back)
        self.is_flying = False
        self.battery = 100
        self.speed = 5  # pixels per frame
        self.boundary_x = 150
        self.boundary_y = 200
        self.boundary_z = 150
        
    def takeoff(self):
        if not self.is_flying:
            self.is_flying = True
            self.y = 50  # Start at safe height
            return "Takeoff successful"
        return "Already flying"
    
    def land(self):
        if self.is_flying:
            self.y = 0
            self.is_flying = False
            return "Landing..."
        return "Already landed"
    
    def move_up(self):
        if self.is_flying and self.y < self.boundary_y:
            self.y = min(self.y + self.speed, self.boundary_y)
    
    def move_down(self):
        if self.is_flying and self.y > 20:
            self.y = max(self.y - self.speed, 20)
    
    def move_left(self):
        if self.is_flying and self.x > -self.boundary_x:
            self.x = max(self.x - self.speed, -self.boundary_x)
    
    def move_right(self):
        if self.is_flying and self.x < self.boundary_x:
            self.x = min(self.x + self.speed, self.boundary_x)
    
    def hover(self):
        pass  # Do nothing, maintain position
    
    def get_position(self):
        return (self.x, self.y, self.z)
    
    def is_within_bounds(self):
        return (abs(self.x) <= self.boundary_x and 
                self.y <= self.boundary_y and 
                abs(self.z) <= self.boundary_z)


drone = DroneSimulator()
print("Drone simulator initialized")

# Gesture control parameters
margin = 40
prediction_buffer = deque(maxlen=7)
gesture_hold_time = 0.5  # Seconds to hold gesture before executing
last_gesture_time = time.time()
last_executed_gesture = None

# UI Colors (BGR)
COLORS = {
    'background': (30, 30, 30),
    'surface': (50, 50, 50),
    'primary': (88, 214, 141),
    'secondary': (255, 179, 71),
    'danger': (67, 67, 244),
    'success': (76, 175, 80),
    'text': (255, 255, 255),
    'text_dim': (150, 150, 150),
    'drone': (100, 200, 255),
}

# Gesture to action mapping
GESTURE_ACTIONS = {
    'up': ('Move Up', '↑'),
    'down': ('Move Down', '↓'),
    'left': ('Move Left', '←'),
    'right': ('Move Right', '→'),
    'hover': ('Hover', '●'),
    'land': ('Land', '⏷'),
}

def get_hand_bbox(hand_landmarks, frame_shape):
    """Extract bounding box from hand landmarks."""
    h, w, _ = frame_shape
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    
    x_min = int(min(x_coords)) - margin
    y_min = int(min(y_coords)) - margin
    x_max = int(max(x_coords)) + margin
    y_max = int(max(y_coords)) + margin
    
    return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)

def make_square_bbox(x_min, y_min, x_max, y_max, frame_shape):
    """Make bbox square."""
    h, w, _ = frame_shape
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min_sq = max(0, center_x - size // 2)
    y_min_sq = max(0, center_y - size // 2)
    x_max_sq = min(w, center_x + size // 2)
    y_max_sq = min(h, center_y + size // 2)
    
    return x_min_sq, y_min_sq, x_max_sq, y_max_sq

def preprocess_hand(frame, hand_landmarks):
    """Preprocess hand for model prediction."""
    x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape)
    x_min, y_min, x_max, y_max = make_square_bbox(x_min, y_min, x_max, y_max, frame.shape)
    
    hand_crop = frame[y_min:y_max, x_min:x_max]
    if hand_crop.size == 0:
        return None, None
    
    hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    hand_resized = cv2.resize(hand_gray, (IMG_SIZE, IMG_SIZE))
    hand_normalized = hand_resized / 255.0
    hand_input = np.expand_dims(hand_normalized, axis=(0, -1))
    
    return hand_input, (x_min, y_min, x_max, y_max)

def draw_drone_visualization(x=400, y=100, width=400, height=300):
    """Create 3D-style drone visualization."""
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    vis[:] = COLORS['background']
    
    # Draw grid
    grid_color = (60, 60, 60)
    for i in range(0, width, 40):
        cv2.line(vis, (i, 0), (i, height), grid_color, 1)
    for i in range(0, height, 40):
        cv2.line(vis, (0, i), (width, i), grid_color, 1)
    
    # Draw boundaries
    boundary_rect = (50, 20, width - 100, height - 40)
    cv2.rectangle(vis, (boundary_rect[0], boundary_rect[1]), 
                  (boundary_rect[0] + boundary_rect[2], boundary_rect[1] + boundary_rect[3]),
                  (80, 80, 80), 2)
    
    # Draw center crosshair
    center_x = width // 2
    center_y = height // 2
    cv2.line(vis, (center_x - 10, center_y), (center_x + 10, center_y), (100, 100, 100), 1)
    cv2.line(vis, (center_x, center_y - 10), (center_x, center_y + 10), (100, 100, 100), 1)
    
    # Calculate drone position on visualization
    # Map drone coordinates to visualization space
    drone_x = int(center_x + (drone.x / drone.boundary_x) * (boundary_rect[2] / 2))
    drone_y = int(boundary_rect[1] + boundary_rect[3] - (drone.y / drone.boundary_y) * boundary_rect[3])
    
    # Draw drone trail
    trail_length = 20
    if not hasattr(draw_drone_visualization, 'trail'):
        draw_drone_visualization.trail = deque(maxlen=trail_length)
    draw_drone_visualization.trail.append((drone_x, drone_y))
    
    for i, (tx, ty) in enumerate(draw_drone_visualization.trail):
        alpha = i / trail_length
        color = tuple(int(c * alpha) for c in COLORS['primary'])
        cv2.circle(vis, (tx, ty), 2, color, -1)
    
    # Draw drone (as a simple quadcopter shape)
    drone_size = 20
    if drone.is_flying:
        # Propeller arms
        cv2.line(vis, (drone_x - drone_size, drone_y - drone_size), 
                (drone_x + drone_size, drone_y + drone_size), COLORS['drone'], 2)
        cv2.line(vis, (drone_x + drone_size, drone_y - drone_size), 
                (drone_x - drone_size, drone_y + drone_size), COLORS['drone'], 2)
        
        # Propellers
        for dx, dy in [(-drone_size, -drone_size), (drone_size, -drone_size), 
                       (-drone_size, drone_size), (drone_size, drone_size)]:
            cv2.circle(vis, (drone_x + dx, drone_y + dy), 6, COLORS['primary'], -1)
        
        # Center body
        cv2.circle(vis, (drone_x, drone_y), 8, COLORS['drone'], -1)
        cv2.circle(vis, (drone_x, drone_y), 8, COLORS['text'], 2)
    else:
        # Landed drone (simpler icon)
        cv2.circle(vis, (drone_x, drone_y), 10, (100, 100, 100), -1)
        cv2.circle(vis, (drone_x, drone_y), 10, COLORS['text_dim'], 2)
    
    # Draw altitude indicator
    alt_x = width - 30
    alt_bar_height = height - 60
    alt_bar_y = 30
    
    # Background bar
    cv2.rectangle(vis, (alt_x - 10, alt_bar_y), (alt_x + 10, alt_bar_y + alt_bar_height),
                 (60, 60, 60), -1)
    
    # Current altitude
    if drone.y > 0:
        alt_fill = int((drone.y / drone.boundary_y) * alt_bar_height)
        cv2.rectangle(vis, (alt_x - 10, alt_bar_y + alt_bar_height - alt_fill),
                     (alt_x + 10, alt_bar_y + alt_bar_height), COLORS['primary'], -1)
    
    # Altitude labels
    cv2.putText(vis, f"{int(drone.y)}m", (alt_x - 35, alt_bar_y + alt_bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    return vis

def draw_ui(frame, predicted_gesture, confidence, all_probs, hand_detected):
    """Draw complete UI with drone visualization and gesture info."""
    h, w, _ = frame.shape
    
    # === TOP BAR ===
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
    
    # Title
    cv2.putText(frame, "DRONE GESTURE CONTROL", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['primary'], 2)
    
    # Mode indicator
    mode_text = "SIMULATION" if SIMULATION_MODE else "REAL DRONE"
    mode_color = COLORS['secondary'] if SIMULATION_MODE else COLORS['danger']
    cv2.putText(frame, mode_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)
    
    # Flying status
    status_text = "FLYING" if drone.is_flying else "LANDED"
    status_color = COLORS['success'] if drone.is_flying else COLORS['text_dim']
    cv2.circle(frame, (w - 200, 50), 15, status_color, -1)
    cv2.putText(frame, status_text, (w - 175, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Battery (simulation always shows 100%)
    battery = drone.battery if SIMULATION_MODE else drone.get_battery()
    battery_color = COLORS['success'] if battery > 20 else COLORS['danger']
    cv2.putText(frame, f"Battery: {battery}%", (w - 200, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, battery_color, 1)
    
    # === RIGHT PANEL - Drone Visualization ===
    panel_width = 420
    panel_x = w - panel_width
    panel_height = h - 100
    
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 100), (w, h), COLORS['background'], -1)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
    cv2.line(frame, (panel_x, 100), (panel_x, h), COLORS['primary'], 2)
    
    # Drone visualization
    drone_vis = draw_drone_visualization(x=panel_x, y=100, width=panel_width, height=350)
    frame[120:470, panel_x :panel_x + panel_width] = drone_vis
    
    cv2.putText(frame, "DRONE VIEW", (panel_x + 20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
    
    # Position info
    info_y = 490
    cv2.putText(frame, "POSITION", (panel_x + 20, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
    
    pos_data = [
        (f"X: {drone.x:+4.0f}", COLORS['text']),
        (f"Y: {drone.y:+4.0f}", COLORS['primary']),
        (f"Z: {drone.z:+4.0f}", COLORS['text']),
    ]
    
    for i, (text, color) in enumerate(pos_data):
        cv2.putText(frame, text, (panel_x + 20, info_y + 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # === CURRENT GESTURE DISPLAY ===
    gesture_y = 590
    
    # Gesture card
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x + 10, gesture_y), (w - 10, gesture_y + 120),
                 COLORS['surface'], -1)
    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)
    
    cv2.putText(frame, "CURRENT GESTURE", (panel_x + 25, gesture_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
    
    if hand_detected and predicted_gesture:
        # Gesture name and symbol
        action_name, symbol = GESTURE_ACTIONS.get(predicted_gesture, (predicted_gesture.upper(), '?'))
        
        # Large symbol
        cv2.putText(frame, symbol, (panel_x + 25, gesture_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLORS['primary'], 3)
        
        # Gesture name
        cv2.putText(frame, action_name.upper(), (panel_x + 100, gesture_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['text'], 2)
        
        # Confidence
        conf_color = COLORS['success'] if confidence > 0.8 else COLORS['secondary']
        cv2.putText(frame, f"{confidence*100:.0f}%", (panel_x + 330, gesture_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
    else:
        cv2.putText(frame, "NO HAND DETECTED", (panel_x + 80, gesture_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text_dim'], 1)
    
    # === GESTURE PROBABILITIES ===
    if all_probs is not None and hand_detected:
        prob_y = h - 100
        cv2.putText(frame, "ALL GESTURES", (panel_x + 20, prob_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dim'], 1)
        
        # Sort by probability
        sorted_indices = np.argsort(all_probs)[::-1]
        
        for i, idx in enumerate(sorted_indices[:3]):
            gesture_name = class_names[idx]
            prob = all_probs[idx]
            
            # Progress bar
            bar_width = 120
            bar_height = 8
            bar_x = panel_x + 120
            bar_y = prob_y + 10 + i * 25
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (60, 60, 60), -1)
            
            # Fill
            fill_width = int(bar_width * prob)
            if fill_width > 0:
                color = COLORS['primary'] if i == 0 else COLORS['text_dim']
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                             color, -1)
            
            # Label
            cv2.putText(frame, gesture_name[:5], (panel_x + 20, bar_y + 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
            
            # Percentage
            cv2.putText(frame, f"{prob*100:.0f}%", (bar_x + bar_width + 10, bar_y + 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    # === CONTROLS INFO ===
    controls_y = h - 30
    cv2.putText(frame, "T: Takeoff  |  L: Land  |  Q: Quit  |  R: Reset Position", 
                (20, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
    
    return frame

# Main loop
print("DRONE GESTURE CONTROL - STARTING")
print("\nControls:")
print("  T - Takeoff")
print("  L - Land")
print("  R - Reset drone position")
print("  Q - Quit")
print("\nGesture Commands:")
for gesture, (action, symbol) in GESTURE_ACTIONS.items():
    print(f"  {gesture.upper():8s} {symbol} - {action}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    predicted_gesture = None
    confidence = 0.0
    all_probs = None
    hand_detected = False
    
    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
        )
        
        # Get prediction
        hand_input, bbox = preprocess_hand(frame, hand_landmarks)
        
        if hand_input is not None:
            predictions = model.predict(hand_input, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            all_probs = predictions
            
            # Smooth predictions
            prediction_buffer.append(predicted_class)
            
            if len(prediction_buffer) >= 3:
                from collections import Counter
                counter = Counter(prediction_buffer)
                smoothed_class = counter.most_common(1)[0][0]
                predicted_gesture = class_names[smoothed_class]
                
                # Execute gesture command if held long enough
                current_time = time.time()
                if predicted_gesture != last_executed_gesture:
                    if current_time - last_gesture_time > gesture_hold_time and confidence > 0.7:
                        # Execute command
                        if drone.is_flying:
                            if predicted_gesture == 'up':
                                drone.move_up()
                                print(f"↑ Moving UP - Height: {drone.y}")
                            elif predicted_gesture == 'down':
                                drone.move_down()
                                print(f"↓ Moving DOWN - Height: {drone.y}")
                            elif predicted_gesture == 'left':
                                drone.move_left()
                                print(f"← Moving LEFT - X: {drone.x}")
                            elif predicted_gesture == 'right':
                                drone.move_right()
                                print(f"→ Moving RIGHT - X: {drone.x}")
                            elif predicted_gesture == 'hover':
                                drone.hover()
                                print(f"● HOVERING at ({drone.x}, {drone.y})")
                            elif predicted_gesture == 'land':
                                result = drone.land()
                                print(f"⏷ {result}")
                        
                        last_executed_gesture = predicted_gesture
                        last_gesture_time = current_time
                else:
                    last_gesture_time = current_time
            
            # Draw bounding box
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                color = COLORS['success'] if confidence > 0.8 else COLORS['secondary']
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
    
    # Draw UI
    frame = draw_ui(frame, predicted_gesture, confidence, all_probs, hand_detected)
    
    # Display
    cv2.imshow("Drone Gesture Control", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nQuitting...")
        if drone.is_flying:
            print("Landing drone before quit...")
            drone.land()
            time.sleep(2)
        break
    elif key == ord('t'):
        result = drone.takeoff()
        print(f"\n✈️  {result}")
    elif key == ord('l'):
        result = drone.land()
        print(f"\n⏷ {result}")
    elif key == ord('r'):
        drone.x = 0
        drone.y = 50 if drone.is_flying else 0
        drone.z = 0
        print(f"\n↺ Position reset to center")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

if not SIMULATION_MODE:
    if drone.is_flying:
        drone.land()
    drone.end()

print("\n✓ Drone control session ended")

