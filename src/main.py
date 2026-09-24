import cv2
import numpy as np
import os
import csv
import onnxruntime as ort
from collections import deque
from datetime import datetime
import time


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DETECTOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2026may.onnx"
)

RECOGNITION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_recognition_sface_2021dec.onnx"
)

LIVENESS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "minifasnet_v2.onnx"
)

FACES_DIR = os.path.join(
    BASE_DIR,
    "faces"
)

LOGS_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

LOG_FILE = os.path.join(
    LOGS_DIR,
    "authentication_log.csv"
)


# ============================================================
# CREATE LOG DIRECTORY
# ============================================================

os.makedirs(
    LOGS_DIR,
    exist_ok=True
)


# ============================================================
# CREATE CSV LOG FILE
# ============================================================

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Liveness",
            "Identity",
            "Match Score",
            "Status"
        ])


# ============================================================
# LOGGING FUNCTION
# ============================================================

def log_event(
    liveness,
    identity,
    score,
    status
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if score is None:

        score_value = "N/A"

    else:

        score_value = f"{score:.4f}"


    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            liveness,
            identity,
            score_value,
            status
        ])


# ============================================================
# LOAD MODELS
# ============================================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_PATH,
    ""
)

liveness_session = ort.InferenceSession(
    LIVENESS_PATH,
    providers=["CPUExecutionProvider"]
)

liveness_input_name = (
    liveness_session.get_inputs()[0].name
)


# ============================================================
# LOAD REGISTERED FACES
# ============================================================

known_faces = {}

for filename in os.listdir(FACES_DIR):

    if filename.endswith(".npy"):

        name = os.path.splitext(filename)[0]

        file_path = os.path.join(
            FACES_DIR,
            filename
        )

        feature = np.load(file_path)

        known_faces[name] = feature


print("Known faces:")

for name in known_faces:
    print("-", name)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera")
    exit()


print("\nFace Authentication System started.")
print("Press Q to quit.")
print(
    f"Logging authentication events to:\n{LOG_FILE}"
)


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

HISTORY_SIZE = 8

liveness_history = deque(
    maxlen=HISTORY_SIZE
)

recognition_history = deque(
    maxlen=HISTORY_SIZE
)


# ============================================================
# LIVENESS STATE
# ============================================================

liveness_state = "CHECKING"


# ============================================================
# LOGGING TIMER
# ============================================================

last_log_time = 0


# ============================================================
# LIVENESS CROP
# ============================================================

def get_liveness_crop(frame, face):

    x, y, w, h = face[:4].astype(int)

    size = int(
        max(w, h) * 2.7
    )

    center_x = x + w // 2
    center_y = y + h // 2

    left = center_x - size // 2
    top = center_y - size // 2

    right = left + size
    bottom = top + size

    left = max(0, left)
    top = max(0, top)

    right = min(
        frame.shape[1],
        right
    )

    bottom = min(
        frame.shape[0],
        bottom
    )

    face_crop = frame[
        top:bottom,
        left:right
    ]

    if face_crop.size == 0:

        return None

    face_crop = cv2.resize(
        face_crop,
        (80, 80)
    )

    return face_crop


# ============================================================
# LIVENESS PREDICTION
# ============================================================

def predict_liveness(face_crop):

    face_crop = face_crop.astype(
        np.float32
    )

    face_crop = np.transpose(
        face_crop,
        (2, 0, 1)
    )

    face_crop = np.expand_dims(
        face_crop,
        axis=0
    )

    output = liveness_session.run(
        None,
        {
            liveness_input_name:
            face_crop
        }
    )

    logits = output[0][0]

    exp_values = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        exp_values /
        np.sum(exp_values)
    )

    # Model mapping:
    #
    # Class 0 = SPOOF
    # Class 1 = REAL
    # Class 2 = SPOOF

    spoof_0 = probabilities[0]
    real = probabilities[1]
    spoof_2 = probabilities[2]

    spoof = max(
        spoof_0,
        spoof_2
    )

    return (
        float(real),
        float(spoof)
    )


# ============================================================
# DRAW UI PANEL
# ============================================================

def draw_panel(
    frame,
    liveness,
    identity,
    score,
    status
):

    panel_x = 20
    panel_y = 20

    panel_width = 390
    panel_height = 205

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0,
        frame
    )

    cv2.putText(
        frame,
        "FACE AUTHENTICATION",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (40, 68),
        (390, 68),
        (120, 120, 120),
        1
    )

    cv2.putText(
        frame,
        f"Liveness     : {liveness}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Identity     : {identity}",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    if score is None:

        score_text = "N/A"

    else:

        score_text = f"{score:.2f}"

    cv2.putText(
        frame,
        f"Match Score  : {score_text}",
        (40, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Status       : {status}",
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read frame")
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)


    # ========================================================
    # NO FACE
    # ========================================================

    if faces is None or len(faces) == 0:

        liveness_history.clear()
        recognition_history.clear()

        liveness_state = "CHECKING"

        draw_panel(
            frame,
            "N/A",
            "N/A",
            None,
            "NO FACE"
        )

        cv2.imshow(
            "Face Authentication System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        continue


    # ========================================================
    # PROCESS DETECTED FACES
    # ========================================================

    for face in faces:

        x, y, w, h = face[:4].astype(int)


        # ====================================================
        # LIVENESS CROP
        # ====================================================

        face_crop = get_liveness_crop(
            frame,
            face
        )

        if face_crop is None:
            continue


        # ====================================================
        # LIVENESS PREDICTION
        # ====================================================

        real_score, spoof_score = (
            predict_liveness(face_crop)
        )


        current_is_real = (
            real_score >= 0.50
        )


        liveness_history.append(
            current_is_real
        )


        real_frames = sum(
            liveness_history
        )

        total_frames = len(
            liveness_history
        )

        spoof_frames = (
            total_frames -
            real_frames
        )


        # ====================================================
        # LIVENESS STATE
        # ====================================================

        if total_frames < HISTORY_SIZE:

            liveness_state = "CHECKING"


        else:

            if liveness_state == "CHECKING":

                if real_frames >= 5:

                    liveness_state = "REAL"

                elif spoof_frames >= 7:

                    liveness_state = "SPOOF"


            elif liveness_state == "REAL":

                if spoof_frames >= 7:

                    liveness_state = "SPOOF"


            elif liveness_state == "SPOOF":

                if real_frames >= 5:

                    liveness_state = "REAL"


        # ====================================================
        # FACE BOX
        # ====================================================

        if liveness_state == "REAL":

            box_color = (
                0,
                255,
                0
            )

        elif liveness_state == "SPOOF":

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                0,
                255,
                255
            )


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )


        # ====================================================
        # SPOOF
        # ====================================================

        if liveness_state == "SPOOF":

            recognition_history.clear()

            identity = "Blocked"
            match_score = None
            final_status = "SPOOF DETECTED"

            draw_panel(
                frame,
                "SPOOF",
                identity,
                match_score,
                final_status
            )


            # ------------------------------------------------
            # LOG
            # ------------------------------------------------

            current_time = time.time()

            if current_time - last_log_time >= 1:

                log_event(
                    "SPOOF",
                    "Blocked",
                    None,
                    "SPOOF DETECTED"
                )

                last_log_time = current_time


            continue


        # ====================================================
        # CHECKING
        # ====================================================

        if liveness_state == "CHECKING":

            draw_panel(
                frame,
                "CHECKING",
                "N/A",
                None,
                "VERIFYING"
            )

            continue


        # ====================================================
        # REAL → FACE RECOGNITION
        # ====================================================

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )


        best_name = "Unknown"
        best_score = -1


        # ====================================================
        # COMPARE WITH REGISTERED FACES
        # ====================================================

        for name, known_feature in known_faces.items():

            score = recognizer.match(
                feature,
                known_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if score > best_score:

                best_score = score
                best_name = name


        # ====================================================
        # RECOGNITION THRESHOLD
        # ====================================================

        if best_score < 0.363:

            best_name = "Unknown"


        # ====================================================
        # RECOGNITION HISTORY
        # ====================================================

        recognition_history.append(
            (
                best_name,
                best_score
            )
        )


        # ====================================================
        # STABILIZE RECOGNITION
        # ====================================================

        names = [
            item[0]
            for item in recognition_history
        ]


        counts = {}

        for name in names:

            counts[name] = (
                counts.get(name, 0) + 1
            )


        stable_name = max(
            counts,
            key=counts.get
        )


        stable_scores = [
            score
            for name, score
            in recognition_history
            if name == stable_name
        ]


        stable_score = float(
            np.mean(stable_scores)
        )


        # ====================================================
        # FINAL STATUS
        # ====================================================

        if stable_name == "Unknown":

            final_status = "UNKNOWN"

        else:

            final_status = "VERIFIED"


        # ====================================================
        # DISPLAY
        # ====================================================

        draw_panel(
            frame,
            "REAL",
            stable_name,
            stable_score,
            final_status
        )


        # ====================================================
        # LOG REAL / UNKNOWN / VERIFIED
        # ====================================================

        current_time = time.time()

        if current_time - last_log_time >= 1:

            log_event(
                "REAL",
                stable_name,
                stable_score,
                final_status
            )

            last_log_time = current_time


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Face Authentication System",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print(
    f"\nAuthentication log saved to:\n{LOG_FILE}"
)