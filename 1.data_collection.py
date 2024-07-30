import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Initialize the MediaPipe FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

# Create directories to save the extracted eye images
base_dir = "new_data"
open_eyes = os.path.join(base_dir, "open eyes")
close_eyes = os.path.join(base_dir, "close eyes")

os.makedirs(open_eyes, exist_ok=True)
os.makedirs(close_eyes, exist_ok=True)

eye_counter = 0
collect_open_eyes = True  # Flag to determine which images to collect

def get_eye_dimensions(landmarks, eye_landmarks, frame):
    # Get coordinates for the eye landmarks
    coords = [(int(landmarks[id].x * frame.shape[1]), int(landmarks[id].y * frame.shape[0])) for id in eye_landmarks]

    # Calculate width and height
    width = max([point[0] for point in coords]) - min([point[0] for point in coords])
    height = max([point[1] for point in coords]) - min([point[1] for point in coords])

    return coords, width, height

def display_message(window_name, message, duration=3):
    # Display a message on the screen for a specified duration
    window_height = 700  # Increased height
    window_width = 1400  # Increased width
    message_frame = 255 * np.ones((window_height, window_width, 3), dtype=np.uint8)
    font_scale = 1.5
    thickness = 2
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (message_frame.shape[1] - text_size[0]) // 2
    text_y = (message_frame.shape[0] + text_size[1]) // 2
    cv2.putText(message_frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.imshow(window_name, message_frame)
    cv2.waitKey(duration * 1000)


# Display initial message to open eyes
display_message("Instruction", "Please open your eyes and rotate in all directions.", duration=5)

while eye_counter < 6000:
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the face landmarks
    result = face_mesh.process(rgb_frame)

    # Draw face landmarks and capture images
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Define eye landmarks for left and right eyes
            left_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]

            # Calculate dimensions for left and right eyes
            left_eye_coords, left_eye_width, left_eye_height = get_eye_dimensions(face_landmarks.landmark, left_eye_landmarks, frame)
            right_eye_coords, right_eye_width, right_eye_height = get_eye_dimensions(face_landmarks.landmark, right_eye_landmarks, frame)

            # Draw bounding boxes around eyes with a margin of 10 pixels
            left_eye_x_min = min([point[0] for point in left_eye_coords]) - 15
            left_eye_x_max = max([point[0] for point in left_eye_coords]) + 10
            left_eye_y_min = min([point[1] for point in left_eye_coords]) - 30
            left_eye_y_max = max([point[1] for point in left_eye_coords]) + 20

            right_eye_x_min = min([point[0] for point in right_eye_coords]) - 15
            right_eye_x_max = max([point[0] for point in right_eye_coords]) + 10
            right_eye_y_min = min([point[1] for point in right_eye_coords]) - 30
            right_eye_y_max = max([point[1] for point in right_eye_coords]) + 20

            # Determine folder to save images
            folder = open_eyes if eye_counter < 3000 else close_eyes

            # Extract and save left eye region from the original frame
            left_eye_region = frame[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max]
            cv2.imwrite(os.path.join(folder, f"left_eye_{eye_counter}.png"), left_eye_region)

            # Extract and save right eye region from the original frame
            right_eye_region = frame[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max]
            cv2.imwrite(os.path.join(folder, f"right_eye_{eye_counter}.png"), right_eye_region)

            eye_counter += 1

            # Draw face mesh landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

            # Draw left eye landmarks with red color
            for (x, y) in left_eye_coords:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red color

            # Draw right eye landmarks with blue color
            for (x, y) in right_eye_coords:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue color

            # Draw height points
            left_eye_top = left_eye_coords[1]  # Top point of left eye
            left_eye_bottom = left_eye_coords[5]  # Bottom point of left eye
            right_eye_top = right_eye_coords[1]  # Top point of right eye
            right_eye_bottom = right_eye_coords[5]  # Bottom point of right eye

            # Draw circles on height points
            cv2.circle(frame, left_eye_top, 2, (0, 255, 255), -1)  # Yellow color for left eye top
            cv2.circle(frame, left_eye_bottom, 2, (0, 255, 255), -1)  # Yellow color for left eye bottom
            cv2.circle(frame, right_eye_top, 2, (0, 255, 255), -1)  # Yellow color for right eye top
            cv2.circle(frame, right_eye_bottom, 2, (0, 255, 255), -1)  # Yellow color for right eye bottom

            # Draw bounding boxes around eyes
            cv2.rectangle(frame, (left_eye_x_min, left_eye_y_min), (left_eye_x_max, left_eye_y_max), (0, 255, 0), 2)  # Green box for left eye
            cv2.rectangle(frame, (right_eye_x_min, right_eye_y_min), (right_eye_x_max, right_eye_y_max), (0, 255, 0), 2)  # Green box for right eye

            # Display dimensions on the frame
            cv2.putText(frame, f'Left Eye: W={left_eye_width}, H={left_eye_height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Right Eye: W={right_eye_width}, H={right_eye_height}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check if we need to switch to collecting closed eye images
    if eye_counter == 3000:
        display_message("Instruction", "Please close your eyes.", duration=5)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
