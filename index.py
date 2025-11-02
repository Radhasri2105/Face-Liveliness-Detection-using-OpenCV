"""
Liveliness detection using OpenCV + MediaPipe (no dlib).
Combines blink detection, mouth movement and small head motion to decide live vs spoof.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# --------- Parameters (tweak if needed) ----------
EYE_AR_THRESH = 0.22        # EAR threshold to consider eyes closed
EYE_AR_CONSEC_FRAMES = 3   # frames required for a valid blink
MOUTH_AR_THRESH = 0.5      # mouth open threshold (normalized)
MOTION_THRESHOLD = 2.0     # minimal centroid movement in pixels per frame to indicate motion
WINDOW_SECONDS = 4         # evaluation window length in seconds
REQUIRED_LIVE_SIGNALS = 2  # how many types of live signals required in window to mark live

# -------------------------------------------------

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# MediaPipe face-mesh landmark indices for eyes and mouth (468-landmark model)
# left eye outer/inner approximate indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # corresponds to eye contour points
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# mouth indices (upper inner lip, lower inner lip, left, right)
MOUTH_IDX = [13, 14, 78, 308]  # 13 upper inner, 14 lower inner, 78 left, 308 right (approx)

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(landmarks, idxs):
    # landmarks: list of (x,y) in image coords
    # idxs: six indices (p0..p5) approximating the eye
    p = [landmarks[i] for i in idxs]
    # vertical distances
    A = euclid(p[1], p[5])
    B = euclid(p[2], p[4])
    # horizontal distance
    C = euclid(p[0], p[3]) + 1e-6
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, idxs):
    # using inner lips (13 upper, 14 lower) and left/right mouth corners
    up = landmarks[idxs[0]]
    down = landmarks[idxs[1]]
    left = landmarks[idxs[2]]
    right = landmarks[idxs[3]]
    vertical = euclid(up, down)
    horizontal = euclid(left, right) + 1e-6
    mar = vertical / horizontal
    return mar

def landmarks_to_pixel_list(landmark_list, frame_w, frame_h):
    pts = []
    for lm in landmark_list:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        pts.append((x, y))
    return pts

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    blink_counter = 0
    blink_total = 0
    ear_closed_frames = 0

    # store events (timestamped) for windowed decision
    recent_signals = deque()  # items: (timestamp, "blink"/"mouth"/"motion")

    prev_centroid = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()

        is_face_present = False
        frame_signals = set()

        if results.multi_face_landmarks:
            is_face_present = True
            face_landmarks = results.multi_face_landmarks[0].landmark
            pts = landmarks_to_pixel_list(face_landmarks, frame_w, frame_h)

            # Draw face mesh (optional)
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0],
                                      mp_face.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))

            # EAR for both eyes
            leftEAR = eye_aspect_ratio(pts, LEFT_EYE_IDX)
            rightEAR = eye_aspect_ratio(pts, RIGHT_EYE_IDX)
            ear = (leftEAR + rightEAR) / 2.0

            # Mouth aspect ratio
            mar = mouth_aspect_ratio(pts, MOUTH_IDX)

            # Blink detection logic
            if ear < EYE_AR_THRESH:
                ear_closed_frames += 1
            else:
                if ear_closed_frames >= EYE_AR_CONSEC_FRAMES:
                    blink_total += 1
                    recent_signals.append((now, "blink"))
                    frame_signals.add("blink")
                ear_closed_frames = 0

            # Mouth detection
            if mar > MOUTH_AR_THRESH:
                # mouth open -> register signal
                recent_signals.append((now, "mouth"))
                frame_signals.add("mouth")

            # Head motion: compare centroid across frames
            # compute centroid of key face points (use nose tip ~ index 1 or average of landmarks)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            centroid = (np.mean(xs), np.mean(ys))

            if prev_centroid is not None:
                mv = euclid(centroid, prev_centroid)
                if mv > MOTION_THRESHOLD:
                    recent_signals.append((now, "motion"))
                    frame_signals.add("motion")
            prev_centroid = centroid

            # overlay metrics
            cv2.putText(frame, f"EAR:{ear:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"MAR:{mar:.3f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"BlinkCnt:{blink_total}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        else:
            prev_centroid = None

        # remove old signals outside window
        cutoff = now - WINDOW_SECONDS
        while recent_signals and recent_signals[0][0] < cutoff:
            recent_signals.popleft()

        # count unique types of live signals in window
        types_present = {t for (_, t) in recent_signals}
        live_signal_count = len(types_present)

        # decision logic: if enough signals in recent window -> live
        if is_face_present and live_signal_count >= REQUIRED_LIVE_SIGNALS:
            label = "LIVE"
            color = (0, 255, 0)
        elif is_face_present and live_signal_count > 0:
            label = "POSSIBLE LIVE"
            color = (0, 200, 200)
        elif is_face_present:
            label = "POSSIBLE SPOOF"
            color = (0, 165, 255)
        else:
            label = "NO FACE"
            color = (0, 0, 255)

        # Show which signals are present
        y = 100
        cv2.putText(frame, f"Signals: {', '.join(sorted(types_present)) if types_present else 'none'}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y += 25
        cv2.putText(frame, f"Decision: {label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        cv2.imshow("Liveliness Detection (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
