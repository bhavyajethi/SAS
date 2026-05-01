# import cv2
# import face_recognition
# import mediapipe as mp
# import pickle
# import time
# import os
# import csv
# import math
# from datetime import datetime

# # --- Configuration & Paths ---
# ENCODINGS_FILE = "data\\encodings.pkl"
# ATTENDANCE_FILE = "data\\daily_attendance.csv"
# EAR_THRESHOLD = 0.22  # If Eye Aspect Ratio drops below this, it's a blink
# COOLDOWN_SECONDS = 60 * 60  # Only mark once per hour per student

# # --- Load Database ---
# print("[INFO] Loading Face Database...")
# with open(ENCODINGS_FILE, "rb") as f:
#     data = pickle.load(f)
# known_encodings = data["encodings"]
# known_names = data["names"]

# # --- Initialize MediaPipe Face Mesh (For Liveness) ---
# mp_face_mesh = mp.solutions.face_mesh  # <-- Add this line back
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=3, 
#     refine_landmarks=True, 
#     min_detection_confidence=0.5, 
#     min_tracking_confidence=0.5
# )

# # --- State Management ---
# # Keeps track of who has been marked recently to prevent spamming the CSV
# marked_students = {}  # Format: {"Bhavya_img": timestamp}
# # Keeps track of blink states per frame
# blinked_in_frame = False 

# def calculate_ear(eye_landmarks):
#     """Calculates the Eye Aspect Ratio (EAR) from 6 landmarks."""
#     # Vertical distances
#     v1 = math.hypot(eye_landmarks[1].x - eye_landmarks[5].x, eye_landmarks[1].y - eye_landmarks[5].y)
#     v2 = math.hypot(eye_landmarks[2].x - eye_landmarks[4].x, eye_landmarks[2].y - eye_landmarks[4].y)
#     # Horizontal distance
#     h = math.hypot(eye_landmarks[0].x - eye_landmarks[3].x, eye_landmarks[0].y - eye_landmarks[3].y)
    
#     ear = (v1 + v2) / (2.0 * h)
#     return ear

# def mark_attendance(name):
#     """Logs the attendance to a CSV file if not in cooldown."""
#     current_time = time.time()
    
#     # Check Cooldown
#     if name in marked_students:
#         time_since_last_mark = current_time - marked_students[name]
#         if time_since_last_mark < COOLDOWN_SECONDS:
#             return False # Still in cooldown, don't mark again

#     # Log to CSV
#     file_exists = os.path.isfile(ATTENDANCE_FILE)
#     now = datetime.now()
#     date_str = now.strftime("%Y-%m-%d")
#     time_str = now.strftime("%H:%M:%S")

#     with open(ATTENDANCE_FILE, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["Name", "Date", "Time", "Status", "Liveness"])
#         writer.writerow([name, date_str, time_str, "Present", "Verified (Blink)"])
    
#     # Update state
#     marked_students[name] = current_time
#     print(f"[SUCCESS] Attendance logged for: {name}")
#     return True

# # --- Main Video Loop ---
# print("[INFO] Starting Video Stream...")
# cap = cv2.VideoCapture(0)

# # Optimization: Only run heavy face recognition every N frames
# process_this_frame = True

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 1. Resize frame to 1/4 size for faster processing on i3 CPU
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
#     # Need full size RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # 2. Run Liveness Detection (MediaPipe is extremely fast, run every frame)
#     results = face_mesh.process(rgb_frame)
#     blinked_in_frame = False
    
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # MediaPipe Left Eye indices: 33, 160, 158, 133, 153, 144
#             left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
#             # MediaPipe Right Eye indices: 362, 385, 387, 263, 373, 380
#             right_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]

#             left_ear = calculate_ear(left_eye)
#             right_ear = calculate_ear(right_eye)
#             avg_ear = (left_ear + right_ear) / 2.0

#             if avg_ear < EAR_THRESHOLD:
#                 blinked_in_frame = True
#                 cv2.putText(frame, "BLINK DETECTED - LIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # 3. Run Face Recognition (Heavy task, run alternate frames)
#     if process_this_frame:
#         face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
#             name = "Unknown"
            
#             # Find the best match distance
#             face_distances = face_recognition.face_distance(known_encodings, face_encoding)
#             if len(face_distances) > 0:
#                 best_match_index = int(face_distances.argmin())
#                 if matches[best_match_index]:
#                     name = known_names[best_match_index]

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # 4. Draw UI and Process Attendance
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up to original size since we found boxes on the 1/4 size image
#         top *= 4; right *= 4; bottom *= 4; left *= 4

#         # Box Color Logic: Red = Fake/Unknown, Green = Verified Alive
#         color = (0, 0, 255) # Default Red
#         label = f"{name} (Checking Liveness...)"

#         if name != "Unknown":
#             if blinked_in_frame:
#                 color = (0, 255, 0) # Green
#                 label = f"{name} - VERIFIED"
#                 # Trigger Passive Marking
#                 mark_attendance(name)
#             elif name in marked_students and (time.time() - marked_students[name]) < COOLDOWN_SECONDS:
#                  color = (255, 255, 0) # Cyan/Yellow if already marked today
#                  label = f"{name} - ALREADY MARKED"
        
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Display the resulting image
#     cv2.imshow('Liveness-Guard Video Feed', frame)

#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# cap.release()
# cv2.destroyAllWindows()







import cv2
import face_recognition
import mediapipe as mp
import pickle
import time
import os
import csv
import math
from datetime import datetime

ENCODINGS_FILE = "data/encodings.pkl"
ATTENDANCE_FILE = "data/daily_attendance.csv"
EAR_THRESH = 0.22  
COOLDOWN_SEC = 3600  

print("[INFO] Loading DB...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
encodings = data["encodings"]
names = data["names"]

mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True)

marked = {}  
blink_state = False 

def get_ear(landmarks):
    v1 = math.hypot(landmarks[1].x - landmarks[5].x, landmarks[1].y - landmarks[5].y)
    v2 = math.hypot(landmarks[2].x - landmarks[4].x, landmarks[2].y - landmarks[4].y)
    h = math.hypot(landmarks[0].x - landmarks[3].x, landmarks[0].y - landmarks[3].y)
    return (v1 + v2) / (2.0 * h)

def log_attendance(name):
    now_ts = time.time()
    
    if name in marked and (now_ts - marked[name]) < COOLDOWN_SEC:
        return False 

    # --- BULLETPROOF HEADER CHECK ---
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    is_empty = not file_exists or os.path.getsize(ATTENDANCE_FILE) == 0
    now = datetime.now()
    
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(["Name", "Date", "Time", "Status", "Liveness"])
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present", "Verified"])
    
    marked[name] = now_ts
    print(f"[+] Logged: {name}")
    return True

print("[INFO] Starting Stream...")
cap = cv2.VideoCapture(0)
process_frame = True

# Initialize state variables to prevent errors on alternate frames
boxes = []
curr_names = []
curr_confidences = []

while True:
    # 1. Start System Diagnostic Timer
    start_time = time.time()

    ret, frame = cap.read()
    if not ret: break

    # --- CAM FIX: Auto-Brightness & Contrast Boost ---
    frame = cv2.convertScaleAbs(frame, alpha=2.5, beta=80) 

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mesh.process(rgb_frame)
    blink_state = False
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            l_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            r_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]

            avg_ear = (get_ear(l_eye) + get_ear(r_eye)) / 2.0

            if avg_ear < EAR_THRESH:
                blink_state = True
                cv2.putText(frame, "BLINK DETECTED", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if process_frame:
        boxes = face_recognition.face_locations(rgb_small, model="hog")
        encs = face_recognition.face_encodings(rgb_small, boxes)

        curr_names = []
        curr_confidences = []
        
        for enc in encs:
            name = "Unknown"
            conf = 0.0
            distances = face_recognition.face_distance(encodings, enc)
            
            if len(distances) > 0:
                best_idx = int(distances.argmin())
                if distances[best_idx] < 0.5:  # Tolerance check
                    name = names[best_idx]
                    # Convert distance to a clean percentage for the UI
                    conf = (1 - distances[best_idx]) * 100 
            
            curr_names.append(name)
            curr_confidences.append(conf)

    process_frame = not process_frame

    # 2. Draw Face Bounding Boxes & Confidence UI
    for (top, right, bottom, left), name, conf in zip(boxes, curr_names, curr_confidences):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        color = (0, 0, 255) 
        
        # Base label logic
        if name == "Unknown":
            label = "Unknown Face"
        else:
            label = f"{name} ({conf:.1f}%) - Checking..."

            if blink_state:
                color = (0, 255, 0)
                label = f"{name} ({conf:.1f}%) - VERIFIED"
                log_attendance(name)
            elif name in marked and (time.time() - marked[name]) < COOLDOWN_SEC:
                 color = (255, 255, 0)
                 label = f"{name} ({conf:.1f}%) - MARKED"
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. Draw System Health Diagnostics (Top Left)
    process_time = time.time() - start_time
    latency = process_time * 1000
    fps = 1.0 / process_time if process_time > 0 else 30.0

    # Draw a semi-transparent black box behind the text so it's always readable
    cv2.rectangle(frame, (5, 5), (320, 45), (0, 0, 0), -1)
    cv2.putText(frame, f"Latency: {latency:.0f}ms | FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Edge Mode: Active | Hardware: i3 CPU", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Liveness-Guard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()