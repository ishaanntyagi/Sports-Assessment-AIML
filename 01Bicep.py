import cv2
import mediapipe as mp
import time

# Placeholder for common utilities (like speak and calculate_angle)
# In a real project, you'd import these from a separate file.
def speak(text):
    pass # Placeholder for a text-to-speech function

def calculate_angle(a, b, c):
    # Placeholder
    return 0

# Bicep Curl state variables
counter = 0
stage = None
last_bad_form_time = 0

def process_frame(image, landmarks):
    """
    Processes a single frame for bicep curl detection and updates state.
    Returns the angle for display.
    """
    global counter, stage, last_bad_form_time

    shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
    
    angle = calculate_angle(shoulder, elbow, wrist)

    # Bicep Curl rep counting logic
    if angle > 160:
        stage = "down"
    if angle < 30 and stage == "down":
        stage = "up"
        counter += 1
        print(f"Bicep Curl Reps: {counter}")
        speak(f"Bicep Curl rep number {counter}")
    
    # Bad form detection: elbow angle is too open during curl
    if angle > 90 and angle < 160 and time.time() - last_bad_form_time > 5:
        speak("Warning! Bad form detected. Keep your elbows in!")
        print("Bad form detected in Bicep Curl.")
        last_bad_form_time = time.time()
        
    return angle