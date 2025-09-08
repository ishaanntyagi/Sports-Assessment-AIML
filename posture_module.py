import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Import drawing utilities
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculates angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_and_draw_posture(image):
    """
    Analyzes a single image for posture and draws the results onto it.
    Returns the modified image.
    """
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- FIX #1: DRAW THE LANDMARKS ---
        # We add this block to draw the skeleton on the image
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        )

        try:
            # Get coordinates and calculate angles
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            avg_back_angle = (calculate_angle(left_shoulder, left_hip, left_knee) + calculate_angle(right_shoulder, right_hip, right_knee)) / 2
            avg_neck_angle = (calculate_angle(left_hip, left_shoulder, left_ear) + calculate_angle(right_hip, right_shoulder, right_ear)) / 2
            
            # Determine posture and color
            if avg_back_angle > 165 and avg_neck_angle > 160:
                posture = "Good Posture"
                color = (0, 255, 0)
            else:
                posture = "Bad Posture"
                color = (0, 0, 255)

            # --- FIX #2: DISPLAY THE LIVE ANGLES ---
            # Draw the status text
            cv2.putText(image, posture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # Draw the angle values
            cv2.putText(image, f"Back Angle: {int(avg_back_angle)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Neck Angle: {int(avg_neck_angle)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass
            
    return image