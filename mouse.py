import cv2
import mediapipe as mp
import pyautogui
import math

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to smooth the movement of the cursor
def smooth_movement(current, previous, alpha=0.5):
    return (1 - alpha) * previous + alpha * current

# Function to detect hand and control mouse based on hand gestures
def detect_hand():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)  # Initialize hand detection with a maximum of 1 hand for faster processing
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # Open the default camera
    screen_width, screen_height = pyautogui.size()  # Get screen size

    alpha = 0.2  # Smoothing factor
    smoothed_x, smoothed_y = 0, 0  # Initial smoothed coordinates
    left_click_pressed = False  # Flag to track left click state
    right_click_pressed = False  # Flag to track right click state
    scrolling_enabled = True  # Flag to enable/disable scrolling

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

        # Process the frame to detect hand landmarks
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand's landmarks

            # Get landmarks for fingertips
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Smooth the movement of the cursor
            smoothed_x = smooth_movement(index_finger_tip.x * screen_width, smoothed_x, alpha)
            smoothed_y = smooth_movement(index_finger_tip.y * screen_height, smoothed_y, alpha)

            # Move the cursor to the smoothed position
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Calculate distance between index finger and middle finger for left-click
            distance_index_middle = calculate_distance(
                (index_finger_tip.x * screen_width, index_finger_tip.y * screen_height),
                (middle_finger_tip.x * screen_width, middle_finger_tip.y * screen_height)
            )

            # Perform left-click if index finger and middle finger are touching
            if distance_index_middle < 40:
                if not left_click_pressed:
                    pyautogui.mouseDown()
                    left_click_pressed = True
            else:
                if left_click_pressed:
                    pyautogui.mouseUp()
                    left_click_pressed = False

            # Calculate distance between thumb and index finger for right-click
            distance_thumb_index = calculate_distance(
                (thumb_tip.x * screen_width, thumb_tip.y * screen_height),
                (index_finger_tip.x * screen_width, index_finger_tip.y * screen_height)
            )

            # Perform right-click if thumb and index finger are touching
            if distance_thumb_index < 40:
                if not right_click_pressed:
                    pyautogui.rightClick()
                    right_click_pressed = True
            else:
                if right_click_pressed:
                    pyautogui.rightClick()
                    right_click_pressed = False

            # Check for scrolling gestures
            if scrolling_enabled:
                if thumb_tip.y < middle_finger_tip.y:
                    pyautogui.scroll(50)  # Scroll up if thumb is above middle finger
                elif thumb_tip.y < ring_finger_tip.y:
                    pyautogui.scroll(-50)  # Scroll down if thumb is above ring finger

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Hand Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the hand detection function if this script is executed directly
if __name__ == "__main__":
    detect_hand()
