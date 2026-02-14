from pythonosc import udp_client
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# OSC setup
client = udp_client.SimpleUDPClient("127.0.0.1", 10000)

# Load model
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# Face model setup
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Pose model setup
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

# Define hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (5,9),(9,10),(10,11),(11,12),     # Middle
    (9,13),(13,14),(14,15),(15,16),   # Ring
    (13,17),(17,18),(18,19),(19,20),  # Pinky
    (0,17)                            # Palm base
]

# Simple Face connections (basic outline + eyes + mouth)
FACE_CONNECTIONS = [
    # Jaw outline (approx indices)
    (10, 338),(338, 297),(297, 332),(332, 284),(284, 251),(251, 389),(389, 356),(356, 454),(454, 323),(323, 361),(361, 288),(288, 397),(397, 365),(365, 379),(379, 378),(378, 400),(400, 377),(377, 152),(152, 148),(148, 176),(176, 149),(149, 150),(150, 136),(136, 172),(172, 58),(58, 132),(132, 93),(93, 234),(234, 127),(127, 162),(162, 21),(21, 54),(54, 103),(103, 67),(67, 109),(109, 10),
    # Left eye
    (33,160),(160,158),(158,133),(133,153),(153,144),(144,33),
    # Right eye
    (362,385),(385,387),(387,263),(263,373),(373,380),(380,362),
    # Mouth
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),(291,61)
]

# Pose connections (basic skeleton)
POSE_CONNECTIONS = [
    (11,12),  # shoulders
    (11,13),(13,15),  # left arm
    (12,14),(14,16),  # right arm
    (11,23),(12,24),  # torso upper
    (23,24),          # hips
    (23,25),(25,27),  # left leg
    (24,26),(26,28)   # right leg
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
frame_timestamp_ms = 0
pose_frame_skip = 0
POSE_SKIP_RATE = 2  # Run pose every 2 frames for smoother tracking
last_pose_result = None  # Cache last valid pose result

prev_gray = None  # For motion energy calculation

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 360))  # Lower processing resolution
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    hand_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # Run pose detection every 3 frames to reduce latency
    if pose_frame_skip % POSE_SKIP_RATE == 0:
        last_pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    pose_frame_skip += 1

    frame_timestamp_ms += 1

    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:

            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            # Draw connections (lines between landmarks)
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                x0 = int(hand_landmarks[start_idx].x * frame.shape[1])
                y0 = int(hand_landmarks[start_idx].y * frame.shape[0])
                x1 = int(hand_landmarks[end_idx].x * frame.shape[1])
                y1 = int(hand_landmarks[end_idx].y * frame.shape[0])

                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # Wrist position
            wrist = hand_landmarks[0]
            client.send_message("/hand_x", wrist.x)
            client.send_message("/hand_y", wrist.y)

            # Pinch distance
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            pinch_dist = math.sqrt(
                (thumb.x - index.x)**2 +
                (thumb.y - index.y)**2
            )

            # Strong exponential scaling for much larger expressive range
            normalized = np.clip(pinch_dist, 0.0, 0.2)
            scaled_pinch = (normalized ** 2.5) * 50.0
            client.send_message("/pinch", scaled_pinch)

    # Face debug drawing
    if face_result.face_landmarks:
        # Draw face connections (thin red lines)
        for connection in FACE_CONNECTIONS:
            start_idx, end_idx = connection
            x0 = int(face_result.face_landmarks[0][start_idx].x * frame.shape[1])
            y0 = int(face_result.face_landmarks[0][start_idx].y * frame.shape[0])
            x1 = int(face_result.face_landmarks[0][end_idx].x * frame.shape[1])
            y1 = int(face_result.face_landmarks[0][end_idx].y * frame.shape[0])
            cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)

    # Pose debug drawing
    if last_pose_result is not None and last_pose_result.pose_landmarks:
        # Body center (midpoint of hips)
        left_hip = last_pose_result.pose_landmarks[0][23]
        right_hip = last_pose_result.pose_landmarks[0][24]
        body_center_x = (left_hip.x + right_hip.x) / 2
        body_center_y = (left_hip.y + right_hip.y) / 2

        client.send_message("/body_center_x", body_center_x)
        client.send_message("/body_center_y", body_center_y)

        for landmark in last_pose_result.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)
        # Draw pose connections (thin yellow lines)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            x0 = int(last_pose_result.pose_landmarks[0][start_idx].x * frame.shape[1])
            y0 = int(last_pose_result.pose_landmarks[0][start_idx].y * frame.shape[0])
            x1 = int(last_pose_result.pose_landmarks[0][end_idx].x * frame.shape[1])
            y1 = int(last_pose_result.pose_landmarks[0][end_idx].y * frame.shape[0])
            cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 1)

    # ---- Motion Energy ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion_energy = float(np.mean(diff))
    else:
        motion_energy = 0.0

    prev_gray = gray

    client.send_message("/motion", motion_energy)

    cv2.putText(frame, "Press 'r' to rescan | ESC to quit", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press ESC to quit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()