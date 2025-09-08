# Personal Trainer Application: Bicep Curl Counter
# This script is a dedicated counter for Bicep Curls using MediaPipe and OpenCV.

import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def speak(text):
    """Speak a warning message."""
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def main():
    cap = cv2.VideoCapture(0)

    # Bicep Curl state variables
    counter = 0
    stage = None
    last_bad_form_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        start_time = time.time()
        prompt_shown = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not prompt_shown:
                cv2.putText(
                    image,
                    "Starting Bicep Curl counter...",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if time.time() - start_time > 3:
                    speak("Starting bicep curl counter.")
                    prompt_shown = True

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for bicep curl
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
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
                
                # Render rep counter and stage
                cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
                
                cv2.putText(image, "REPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, "STAGE", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, (stage if stage else ""), (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, "Exercise: Bicep Curl", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw landmarks and angle
                image_height, image_width, _ = image.shape
                elbow_pixel = tuple(np.multiply(elbow, [image_width, image_height]).astype(int))
                cv2.putText(image, str(int(angle)), elbow_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            except AttributeError:
                pass

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )
            
            cv2.imshow("Bicep Curl Counter", image)

            key = cv2.waitKey(10) & 0xFF
            
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
