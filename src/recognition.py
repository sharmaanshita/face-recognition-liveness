import cv2
import numpy as np
import os

# -----------------------------
# Project paths
# -----------------------------

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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_recognition_sface_2021dec.onnx"
)

FACES_DIR = os.path.join(
    BASE_DIR,
    "faces"
)


# -----------------------------
# Create models
# -----------------------------

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)


# -----------------------------
# Load known faces
# -----------------------------

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


# -----------------------------
# Open webcam
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

print("\nFace recognition started.")
print("Press Q to quit.")


# -----------------------------
# Recognition loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # -----------------------------
            # Align face
            # -----------------------------

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            # -----------------------------
            # Generate embedding
            # -----------------------------

            feature = recognizer.feature(
                aligned_face
            )

            best_name = "Unknown"
            best_score = -1

            # -----------------------------
            # Compare with known faces
            # -----------------------------

            for name, known_feature in known_faces.items():

                score = recognizer.match(
                    feature,
                    known_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:

                    best_score = score
                    best_name = name

            # -----------------------------
            # Recognition threshold
            # -----------------------------

            if best_score < 0.363:
                best_name = "Unknown"

            # -----------------------------
            # Bounding box
            # -----------------------------

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Display name + score
            # -----------------------------

            text = f"{best_name} ({best_score:.2f})"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Show frame
    cv2.imshow(
        "Face Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()
cv2.destroyAllWindows()