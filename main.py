import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import shutil
import time

# Initialize the mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set screen parameters and remove pyautogui delay
screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Variables for smooth cursor movement and control
smoothening = 3
prev_x, prev_y = 0, 0

# Variables for drag-and-drop and click handling
dragging_file = False
start_drag = None
gesture_start_time = None
gesture_threshold = 0.5
last_click_time = 0
double_click_threshold = 0.3  # Max time between taps for double-click

# Define paths for file drag-and-drop
source_path = "/path/to/source/file.txt"  # Replace with your file path
destination_path = "/path/to/destination/folder"  # Replace with your destination folder

zoom_threshold = 50  # Distance between thumb and index finger to trigger zoom
scroll_threshold = 100  # Vertical movement threshold to trigger scrolling

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Control cursor with index finger
            index_finger = landmarks[8]
            x = np.interp(index_finger[0], (75, w - 75), (0, screen_width))
            y = np.interp(index_finger[1], (75, h - 75), (0, screen_height))

            curr_x = prev_x + (x - prev_x) / smoothening
            curr_y = prev_y + (y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Calculate distance between index finger and thumb
            thumb_finger = landmarks[4]
            distance = np.linalg.norm(np.array(index_finger) - np.array(thumb_finger))

            # Check if all five fingers are joined for drag-and-drop gesture
            threshold = 40  # Adjust as needed
            fingers_joined = all(
                np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[i])) < threshold
                for i in [8, 12, 16, 20]
            )

            if fingers_joined and not dragging_file:
                # Start dragging the file
                dragging_file = True
                start_drag = landmarks[8]  # Starting point of drag
                print("Drag started")

            elif dragging_file and not fingers_joined:
                # Drop the file at destination
                dragging_file = False
                print("File dropped")

                try:
                    shutil.move(source_path, destination_path)
                    print("File moved to:", destination_path)
                except Exception as e:
                    print("Error moving file:", e)

            # Scroll when index finger points up and moves significantly
            if index_finger[1] < landmarks[6][1]:  # MCP of index finger
                if curr_y - prev_y > scroll_threshold:
                    pyautogui.scroll(-50)  # Scroll down
                elif prev_y - curr_y > scroll_threshold:
                    pyautogui.scroll(50)  # Scroll up

            # Zoom in/out based on thumb and index finger distance
            if distance < zoom_threshold:
                if gesture_start_time is None:
                    gesture_start_time = cv2.getTickCount()
                else:
                    time_elapsed = (cv2.getTickCount() - gesture_start_time) / cv2.getTickFrequency()
                    if time_elapsed > gesture_threshold:
                        pyautogui.hotkey('ctrl', '+')  # Zoom in
                        gesture_start_time = None
            else:
                gesture_start_time = None

            if distance > zoom_threshold + 20:  # Threshold for zooming out
                pyautogui.hotkey('ctrl', '-')  # Zoom out

            # Click or double-click based on thumb and index finger proximity
            if distance < 20:
                current_time = time.time()
                if current_time - last_click_time < double_click_threshold:
                    pyautogui.doubleClick()
                    print("Double-click")
                    last_click_time = 0  # Reset to avoid repeated double-clicks
                else:
                    pyautogui.click()
                    print("Single-click")
                    last_click_time = current_time
                time.sleep(0.2)  # Prevents rapid firing of clicks

    cv2.imshow("Air Canvas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
