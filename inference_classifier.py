import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Correct label mapping: 0–25 = A–Z, 26–35 = 0–9
labels_dict = {
    **{i: chr(65 + i) for i in range(26)},     # A–Z
    **{i + 26: str(i) for i in range(10)}      # 0–9
}

# Sentence logic
sentence = ""
prediction_history = deque(maxlen=20)
last_prediction = ""
last_time = time.time()

font_path = cv2.FONT_HERSHEY_SIMPLEX  # Monopoly font not supported, fallback used

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_ui = frame.copy()
    predicted_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_ui, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        x_coords, y_coords = [], []
        data_aux = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_coords.append(lm.x)
                y_coords.append(lm.y)

            min_x, min_y = min(x_coords), min(y_coords)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_class = int(prediction[0])
        predicted_label = labels_dict[predicted_class]

        current_time = time.time()
        prediction_history.append(predicted_label)

        if prediction_history.count(predicted_label) >= 15 and predicted_label != last_prediction:
            sentence += predicted_label + " "
            last_prediction = predicted_label
            last_time = current_time

        x1, y1 = int(min(x_coords) * W) - 10, int(min(y_coords) * H) - 10
        x2, y2 = int(max(x_coords) * W) + 10, int(max(y_coords) * H) + 10

        cv2.rectangle(frame_ui, (x1, y1), (x2, y2), (255, 255, 255), 4)
        cv2.putText(frame_ui, predicted_label, (x1, y1 - 10), font_path, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Draw UI Bars
    cv2.rectangle(frame_ui, (0, H - 60), (W, H), (255, 255, 255), -1)
    cv2.putText(frame_ui, f"Sentence: {sentence.strip()}", (10, H - 20), font_path, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Interpreter', frame_ui)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()