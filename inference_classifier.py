
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Check if model.p exists
if not os.path.exists('./model.p'):
    print("Error: model.p not found! Train and save the model first.")
    exit()

# Load model & label encoder
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
label_encoder = model_dict['label_encoder']  # Ensure labels are mapped properly

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Frame rate calculation
prev_time = 0

while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame from webcam.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmark positions
            min_x, min_y = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Ensure `data_aux` has exactly 84 features (42 points × 2 coordinates)
            while len(data_aux) < 84:
                data_aux.append(0.0)  # Padding

            data_aux = data_aux[:84]  # Truncate extra features

            # Bounding box
            x1 = max(1, int(min(x_) * W) - 10)
            y1 = max(1, int(min(y_) * H) - 10)
            x2 = min(W - 1, int(max(x_) * W) + 10)
            y2 = min(H - 1, int(max(y_) * H) + 10)

            # 🔥 FIXED: Predict & decode label correctly
            prediction = model.predict(np.array([data_aux]))  # Ensure input is a 2D array
            predicted_label = label_encoder.inverse_transform([prediction[0]])[0]  # Convert back to original label

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detector', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

