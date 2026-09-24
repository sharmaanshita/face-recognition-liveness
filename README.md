# Face Recognition & Liveness Detection System

A computer vision-based face authentication system that combines **face detection, face recognition, and liveness detection** to verify whether the person in front of the camera is a registered user and whether the input is a real face rather than a spoof attempt.

The system uses OpenCV-based face detection and recognition models together with an ONNX-based liveness model. Authentication events are also recorded in a CSV log for evaluation and auditing.

---

## Features

- Real-time face detection using YuNet
- Face registration using face embeddings
- Face recognition using SFace
- Cosine similarity-based identity matching
- ML-based liveness detection using MiniFASNet
- Protection against basic photo-based spoofing attempts
- Temporal smoothing for more stable liveness decisions
- Hysteresis-based liveness state transitions
- Unknown-person rejection
- Authentication status displayed in real time
- Authentication events stored in CSV format

---

## System Architecture

```text
                    Camera Input
                         |
                         v
                +------------------+
                |  Face Detection  |
                |      YuNet       |
                +------------------+
                         |
                         v
                  Detected Face
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Liveness Detection       Face Recognition
         MiniFASNet                 SFace
             |                       |
             v                       v
       REAL / SPOOF            Face Embedding
                                     |
                                     v
                            Cosine Similarity
                                     |
                                     v
                              Known / Unknown
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
                 VERIFIED                         UNKNOWN
                     |
                     v
              Authentication Log