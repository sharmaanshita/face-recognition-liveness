import cv2
import numpy as np
import os

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
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

# Faces folder
FACES_DIR = os.path.join(BASE_DIR, "faces")

# Create faces folder if it doesn't exist
os.makedirs(FACES_DIR, exist_ok=True)


# Create YuNet detector
detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# Create SFace recognizer
recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)


# Ask for name
name = input("Enter person's name: ").strip()

if not name:
    print("Name cannot be empty.")
    exit()


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

print("\nFace registration started.")
print("Look at the camera.")
print("Press SPACE to capture.")
print("Press Q to quit.")


while True:

    # Read frame
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Get dimensions
    height, width = frame.shape[:2]

    # Tell YuNet the frame size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # Bounding box
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Show camera
    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE → capture
    if key == ord(" "):

        if faces is None or len(faces) == 0:
            print("No face detected. Try again.")
            continue

        if len(faces) > 1:
            print("Multiple faces detected.")
            print("Please keep only one face in the frame.")
            continue

        # Take detected face
        face = faces[0]

        # Align face
        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        # Generate embedding
        feature = recognizer.feature(
            aligned_face
        )

        # Save embedding
        file_path = os.path.join(
            FACES_DIR,
            name + ".npy"
        )

        np.save(file_path, feature)

        print("\nFace registered successfully!")
        print("Name:", name)
        print("Embedding shape:", feature.shape)
        print("Saved to:", file_path)

        break

    # Q → quit
    elif key == ord("q"):
        break


# Cleanup
camera.release()
cv2.destroyAllWindows()