import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the volume control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

# Initialize webcam
cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Variables for smoothing
prev_volume_level = 0
smoothing_factor = 0.3

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image from camera")
        continue

    # Flip the image horizontally for a more intuitive mirror view
    image = cv2.flip(image, 1)
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmarks of the thumb and index finger
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Convert normalized coordinates to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
            
            # Draw circles on the fingertips
            cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            
            # Draw a line between the fingertips
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
            
            # Calculate the distance between fingertips
            distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))
            
            # Map distance to volume range 
            min_distance, max_distance = 20, 100
            
            # Constrain the distance to the range
            distance = np.clip(distance, min_distance, max_distance)
            
            # Map the distance to volume level
            volume_level = np.interp(distance, [min_distance, max_distance], [min_volume, max_volume])
            
            # Apply smoothing
            volume_level = prev_volume_level * (1 - smoothing_factor) + volume_level * smoothing_factor
            prev_volume_level = volume_level
            
            # Set the volume
            volume.SetMasterVolumeLevel(volume_level, None)
            
            # Calculate volume percentage for display
            volume_percentage = int(np.interp(volume_level, [min_volume, max_volume], [0, 100]))
            
            # Display volume percentage
            cv2.putText(image, f"Volume: {volume_percentage}%", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display distance
            cv2.putText(image, f"Distance: {int(distance)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw volume bar
            bar_width = 30
            bar_height = 150
            bar_x = width - 50
            bar_y = int((height - bar_height) / 2)
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                          (0, 0, 255), 3)
            
            filled_height = int(np.interp(volume_level, [min_volume, max_volume], [0, bar_height]))
            cv2.rectangle(image, (bar_x, bar_y + bar_height - filled_height), 
                          (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), cv2.FILLED)
    
    # Display instructions
    cv2.putText(image, "Pinch to control volume", (width - 280, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Hand Volume Control', image)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()