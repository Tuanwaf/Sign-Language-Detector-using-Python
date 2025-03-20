import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: '1', 6: '2', 7: '3', 8: '4', 9: '5'}

# Color dictionary for different labels
colors_dict = {
    'A': (255, 0, 0),  # Red
    'B': (0, 255, 0),  # Green
    'C': (0, 0, 255),  # Blue
    'D': (255, 255, 0),  # Cyan
    'E': (255, 0, 255),  # Magenta
    '1': (0, 255, 255),  # Yellow
    '2': (128, 0, 128),  # Purple
    '3': (128, 128, 0),  # Olive
    '4': (0, 128, 128),  # Teal
    '5': (0,0,0)  # Lime
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 20
        y1 = int(min(y_) * H) - 20
        x2 = int(max(x_) * W) + 20
        y2 = int(max(y_) * H) + 20

        # Predict the letter and its probability
        prediction = model.predict([np.asarray(data_aux)])
        probabilities = model.predict_proba([np.asarray(data_aux)])
        max_probability = np.max(probabilities)
        predicted_character = labels_dict[int(prediction[0])]
        accuracy_percentage = max_probability * 100

        # Get the color for the predicted character
        color = colors_dict[predicted_character]

        # Draw rectangle and text with predicted letter and accuracy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, f'{predicted_character} ({accuracy_percentage:.2f}%)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
