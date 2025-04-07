
import os
import cv2
import string

# Define dataset storage path
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define categories (Alphabets + Common Gestures)
alphabets_upper = list(string.ascii_uppercase)  # A-Z
alphabets_lower = list(string.ascii_lowercase)  # a-z
gestures = ["Thumbs_Up", "Thumbs_Down", "Stop", "Peace", "Fist", "Open_Hand"]

categories = alphabets_upper + alphabets_lower + gestures
dataset_size = 100  # Number of images per category

# Try initializing the camera
for cam_index in range(3):
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Camera initialized at index {cam_index}")
        break
    cap.release()

if not cap.isOpened():
    print("Error: Could not access the camera. Check permissions or index.")
    exit()

# Loop through each category (A-Z, a-z, and gestures)
for category in categories:
    class_path = os.path.join(DATA_DIR, category)
    os.makedirs(class_path, exist_ok=True)

    print(f'\n==== Collecting data for: {category} ====')

    # Wait for user confirmation
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Display instruction to the user
        cv2.putText(frame, f'Show {category} - Press "Q" to start!', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Collect dataset_size number of images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame. Skipping.")
            continue

        # Show current category on screen while recording
        cv2.putText(frame, f'Collecting: {category}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Save the image
        img_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()

