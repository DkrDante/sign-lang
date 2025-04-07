
import os
import pickle
import cv2
import mediapipe as mp
import string

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define dataset directory
DATA_DIR = './data'

# Define categories (Only Uppercase Alphabets A-Z)
categories = list(string.ascii_uppercase)  # A-Z

data = []
labels = []

# Iterate through each category in the dataset directory
for category in categories:
    class_path = os.path.join(DATA_DIR, category)
    
    # Check if the directory exists
    if not os.path.exists(class_path):
        print(f"Warning: Directory {class_path} does not exist. Skipping.")
        continue

    print(f"\nProcessing category: {category}")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Read and process image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue  # Skip if OpenCV fails to load image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Extract hand landmarks
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize landmarks relative to the smallest x and y
                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            data.append(data_aux)
            labels.append(category)

# Save dataset as a pickle file
dataset = {'data': data, 'labels': labels}
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("\n✅ Dataset successfully processed and saved as 'dataset.pickle'.")
print(f"Total samples collected: {len(data)}")

