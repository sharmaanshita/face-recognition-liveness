import cv2

MODEL_PATH = "models/face_detection_yunet_2026may.onnx"

# Create YuNet face detector
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.6,    # confidence threshold
    0.3,    # NMS threshold
    5000
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

print("Face detection started.")
print("Click the camera window and press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Get current frame dimensions
    height, width = frame.shape[:2]

    # Tell YuNet the size of the current frame
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # First 4 values = x, y, width, height
            x, y, w, h = face[:4].astype(int)

            # Last value = confidence
            confidence = face[-1]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display confidence
            text = f"Face: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 5 facial landmarks
            landmarks = face[4:14].reshape((5, 2)).astype(int)

            for landmark_x, landmark_y in landmarks:
                cv2.circle(
                    frame,
                    (landmark_x, landmark_y),
                    3,
                    (0, 0, 255),
                    -1
                )

    cv2.imshow("YuNet Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()