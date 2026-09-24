# Face Recognition & Liveness Detection

An AI-based real-time face recognition system that combines face detection, face recognition, and liveness detection to identify registered individuals and help prevent spoofing attempts.

## Overview

This project uses computer vision and deep learning models with OpenCV and ONNX Runtime to perform real-time face detection, face recognition, and liveness detection through a webcam.

The system detects a face, verifies whether it is a live person, and compares the detected face with registered face data to identify the person.

## Key Features

- Real-time face detection
- Face recognition using facial embeddings
- Face liveness / anti-spoofing detection
- Face registration
- Webcam-based processing
- ONNX model inference
- Modular Python implementation

## Technologies Used

- Python
- OpenCV
- NumPy
- ONNX Runtime
- YuNet
- SFace
- MiniFASNet
- ONNX

## AI Models

### YuNet
Used for face detection and locating faces in the camera frame.

### SFace
Used for face recognition and generating facial feature representations for identity matching.

### MiniFASNet
Used for liveness detection to help distinguish a real face from spoofing attempts such as photographs or screens.

## Project Structure

```text
face-recognition-liveness/
│
├── models/
│   ├── face_detection_yunet_2026may.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── minifasnet_v2.onnx
│
├── src/
│   ├── detection.py
│   ├── liveness.py
│   ├── main.py
│   ├── recognition.py
│   └── register_face.py
│
├── .gitignore
├── README.md
└── requirements.txt