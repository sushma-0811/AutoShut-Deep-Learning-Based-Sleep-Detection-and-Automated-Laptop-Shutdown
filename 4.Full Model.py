import cv2
import mediapipe as mp
import numpy as np
from keras.api.models import load_model
import time
import os

# Load the trained model
model = load_model('model.h5')

# Initialize the MediaPipe FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)



if not cap.isOpened():
    raise RuntimeError("Could not open video capture")

# Initialize counters
open_eye_count = 0
closed_eye_count = 0
total_predictions = 0
eye_counter = 0

# Set the time interval for checking counts (in seconds)
time_interval = 60  # 1 minute
start_time = time.time()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Ensure it's grayscale
    image = cv2.resize(image, (64, 64))  # Resize to (64, 64)
    image = image.astype(np.float32)  # Ensure the data type is float
    image = image.reshape(64, 64, 1)  # Reshape to (64, 64, 1)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

def display_result(frame, prediction, position):
    label = "Closed" if prediction == 0 else "Open"
    color = (0, 0, 255) if prediction == 0 else (0, 255, 0)
    cv2.putText(frame, f'{label}', position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def get_eye_dimensions(landmarks, eye_landmarks, frame):
    coords = [(int(landmarks[id].x * frame.shape[1]), int(landmarks[id].y * frame.shape[0])) for id in eye_landmarks]
    width = max([point[0] for point in coords]) - min([point[0] for point in coords])
    height = max([point[1] for point in coords]) - min([point[1] for point in coords])
    return coords, width, height

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the face landmarks
    result = face_mesh.process(rgb_frame)

    # Draw face landmarks
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]

            left_eye_coords, left_eye_width, left_eye_height = get_eye_dimensions(face_landmarks.landmark, left_eye_landmarks, frame)
            right_eye_coords, right_eye_width, right_eye_height = get_eye_dimensions(face_landmarks.landmark, right_eye_landmarks, frame)

            # Draw bounding boxes around eyes with a margin
            left_eye_x_min = max(min([point[0] for point in left_eye_coords]) - 15, 0)
            left_eye_x_max = min(max([point[0] for point in left_eye_coords]) + 10, frame.shape[1])
            left_eye_y_min = max(min([point[1] for point in left_eye_coords]) - 30, 0)
            left_eye_y_max = min(max([point[1] for point in left_eye_coords]) + 20, frame.shape[0])

            right_eye_x_min = max(min([point[0] for point in right_eye_coords]) - 15, 0)
            right_eye_x_max = min(max([point[0] for point in right_eye_coords]) + 10, frame.shape[1])
            right_eye_y_min = max(min([point[1] for point in right_eye_coords]) - 30, 0)
            right_eye_y_max = min(max([point[1] for point in right_eye_coords]) + 20, frame.shape[0])

            # Extract and preprocess eye regions
            left_eye_region = frame[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max]
            right_eye_region = frame[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max]

            left_eye_region_preprocessed = preprocess_image(left_eye_region)
            right_eye_region_preprocessed = preprocess_image(right_eye_region)

            # Predict eye state
            left_eye_pred_prob = model.predict(left_eye_region_preprocessed)
            right_eye_pred_prob = model.predict(right_eye_region_preprocessed)

            left_eye_pred = (left_eye_pred_prob > 0.5).astype(np.int32).flatten()
            right_eye_pred = (right_eye_pred_prob > 0.5).astype(np.int32).flatten()

            # Increment counters
            if left_eye_pred[0] == 1:
                open_eye_count += 1
            else:
                closed_eye_count += 1

            if right_eye_pred[0] == 1:
                open_eye_count += 1
            else:
                closed_eye_count += 1

            # Increment total predictions
            total_predictions += 2

            # Draw eye status on the main frame
            display_result(frame, left_eye_pred[0], (left_eye_x_min, left_eye_y_min - 10))  # Display for left eye
            display_result(frame, right_eye_pred[0], (right_eye_x_min, right_eye_y_min - 10))  # Display for right eye

            eye_counter += 1

            # Draw face mesh landmarks
            #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                      #mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                      #mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

            # Draw bounding boxes around eyes
            cv2.rectangle(frame, (left_eye_x_min, left_eye_y_min), (left_eye_x_max, left_eye_y_max), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye_x_min, right_eye_y_min), (right_eye_x_max, right_eye_y_max), (0, 255, 0), 2)

            # Display dimensions on the frame
            cv2.putText(frame, f'Left Eye: W={left_eye_width}, H={left_eye_height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Right Eye: W={right_eye_width}, H={right_eye_height}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check if 1 minute has passed
    if time.time() - start_time > time_interval:
        # Calculate probabilities
        open_eye_prob = open_eye_count / total_predictions
        closed_eye_prob = closed_eye_count / total_predictions

        # Print the counts and probabilities
        print(f"Open Eye Count: {open_eye_count}, Probability: {open_eye_prob:.2f}")
        print(f"Closed Eye Count: {closed_eye_count}, Probability: {closed_eye_prob:.2f}")

        if closed_eye_prob > 0.6 : 
            print("Closed eye probability exceeds 0.6. Shutting down the system.")
            os.system('shutdown /s /t 1')  # Shutdown command for Windows


        # Reset the counts
        open_eye_count = 0
        closed_eye_count = 0
        total_predictions = 0

        # Reset the start time
        start_time = time.time()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
