import os
import cv2

# Directory where collected images will be stored
DATA_DIR = './dataset'
os.makedirs(DATA_DIR, exist_ok=True)

# Automatically detect the number of class folders
class_folders = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
print(f"Detected class folders: {class_folders}")

# Number of images per class to collect
dataset_size = 100

# Initialize camera using DirectShow backend to avoid MSMF errors
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera. Please check the camera index.")
    exit(1)

while True:
    label = input("\nEnter the class index to collect data for (or 'exit' to quit): ")
    if label.lower() == 'exit':
        break

    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        print(f"Class folder '{label}' not found. Please create it first.")
        continue

    print(f"\nCollecting data for class {label}")
    print("Press 'q' to start capturing images for this class.")

    # Wait for user to press 'q' to begin
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            continue
        cv2.putText(frame, "Ready? Press 'q' to start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = len(os.listdir(class_dir))  # Continue from existing count
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            continue

        cv2.putText(frame,
                    f'Class {label} - Image {counter+1}/{dataset_size}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Collection interrupted by user.")
            break

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f"Finished collecting data for class {label} ({counter} images).")

# Release resources
cap.release()
cv2.destroyAllWindows()
