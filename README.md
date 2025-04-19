# Hand Gesture Volume Control

A Python application that uses computer vision to control 
your device's volume with hand gestures.

## Overview

This project uses a webcam to detect hand movements and allows you to control your system's volume by adjusting the distance between your thumb and index finger. A smaller distance lowers the volume, while a greater distance increases it.

### Features

- Real-time hand tracking using MediaPipe
- Intuitive volume control using finger pinch gestures
- Visual feedback with volume percentage and level bar
- Smooth volume transitions with gesture dampening

### Requirements

- Python 3.7+
- Webcam
- Windows OS (for pycaw volume control)

### Usage

- Run the script:
  python gesture_volume_control.py

- Position your hand in front of the webcam with your palm facing the camera.
Control the volume by changing the distance between your thumb and index finger:

- Bringing them closer together decreases volume
Moving them further apart increases volume


- Press 'q' to exit the application.

## How It Works
The application uses:

OpenCV for capturing and processing webcam video
MediaPipe for hand landmark detection
pycaw for controlling the system volume on Windows

The program detects the positions of your thumb and index finger tips and calculates the Euclidean distance between them. This distance is mapped to your system's volume range.
Customization
You can adjust the following parameters in the code:

- min_detection_confidence: Adjust the hand detection sensitivity
- min_distance and max_distance: Calibrate the finger distance range
- smoothing_factor: Change how quickly volume adjusts to gestures

### Adapting for Other Operating Systems
This project uses pycaw for Windows volume control. For other operating systems:

macOS: Replace with osascript commands
Linux: Use packages like alsaaudio or pulseaudio

## Acknowledgments

- MediaPipe for the hand tracking solution
- pycaw for Windows audio control
