import os
import pickle
import cv2
import mediapipe as mp

# Directory where collected images are stored
DATA_DIR = './dataset'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Containers for features and labels
data = []
labels = []

# Loop through each class directory inside DATA_DIR
for dir_name in sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else float('inf')):
    class_dir = os.path.join(DATA_DIR, dir_name)
    if not os.path.isdir(class_dir) or not dir_name.isdigit():
        continue  # Skip non-numeric or invalid directories

    print(f"🔍 Processing class {dir_name}...")

    # Iterate over each image file in the class directory
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable files

        # Convert image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Extract normalized landmark positions for one hand
            x_coords, y_coords = [], []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)

                # Create feature vector: normalize landmarks relative to min x and y
                feature_vector = []
                min_x, min_y = min(x_coords), min(y_coords)
                for lm in hand_landmarks.landmark:
                    feature_vector.append(lm.x - min_x)
                    feature_vector.append(lm.y - min_y)

                data.append(feature_vector)
                labels.append(int(dir_name))

print(f"✅ Processed {len(data)} samples. Saving to data.pickle...")

# Save the extracted data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Dataset saved. Classes included: {sorted(set(labels))}")
