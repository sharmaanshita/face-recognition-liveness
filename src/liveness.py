import cv2
import numpy as np
import os
import onnxruntime as ort


# ==========================================
# Project paths
# ==========================================

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

LIVENESS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "minifasnet_v2.onnx"
)


# ==========================================
# Load YuNet
# ==========================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# ==========================================
# Load MiniFASNet
# ==========================================

session = ort.InferenceSession(
    LIVENESS_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Liveness model loaded.")
print("Input shape:", session.get_inputs()[0].shape)


# ==========================================
# Softmax
# ==========================================

def softmax(values):

    values = values - np.max(values)

    probabilities = np.exp(values)

    return probabilities / np.sum(probabilities)


# ==========================================
# Create 2.7x face crop
# ==========================================

def get_face_crop(frame, face):

    height, width = frame.shape[:2]

    x, y, w, h = face[:4]

    center_x = x + w / 2
    center_y = y + h / 2

    side = max(w, h) * 2.7

    x1 = max(
        0,
        int(center_x - side / 2)
    )

    y1 = max(
        0,
        int(center_y - side / 2)
    )

    x2 = min(
        width,
        int(center_x + side / 2)
    )

    y2 = min(
        height,
        int(center_y + side / 2)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (80, 80)
    )


# ==========================================
# Open webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera")
    exit()


print()
print("Liveness detection started.")
print("Press Q to quit.")


# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # ======================================
    # Detect face
    # ======================================

    _, faces = detector.detect(frame)


    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # ----------------------------------
            # Crop face
            # ----------------------------------

            face_crop = get_face_crop(
                frame,
                face
            )

            if face_crop is None:
                continue


            # ----------------------------------
            # Prepare input
            #
            # Empirically validated for the
            # exact ONNX file being used.
            # ----------------------------------

            face_crop = face_crop.astype(
                np.float32
            )

            # HWC -> CHW
            face_crop = np.transpose(
                face_crop,
                (2, 0, 1)
            )

            # Add batch dimension
            face_crop = np.expand_dims(
                face_crop,
                axis=0
            )


            # ----------------------------------
            # Run model
            # ----------------------------------

            output = session.run(
                None,
                {
                    input_name: face_crop
                }
            )


            # ----------------------------------
            # Probabilities
            # ----------------------------------

            probabilities = softmax(
                output[0][0]
            )


            # For this MiniFASNet V2 model:
            #
            # 0 = spoof
            # 1 = real
            # 2 = spoof

            spoof_0 = probabilities[0]
            real = probabilities[1]
            spoof_2 = probabilities[2]


            # ----------------------------------
            # Decide REAL / SPOOF
            # ----------------------------------

            if real >= 0.50:

                label = "REAL"

            else:

                label = "SPOOF"


            # ----------------------------------
            # Draw bounding box
            # ----------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # Display
            # ----------------------------------

            text = (
                f"{label} | "
                f"Real:{real:.2f} "
                f"Spoof:{max(spoof_0, spoof_2):.2f}"
            )

            cv2.putText(
                frame,
                text,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


    # ======================================
    # Display camera
    # ======================================

    cv2.imshow(
        "Liveness Detection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

camera.release()

cv2.destroyAllWindows()