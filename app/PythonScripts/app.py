from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#static_image_mode=False: Process video streams instead of static images.
#max_num_hands=2: Detect up to two hands.
#min_detection_confidence=0.5: Minimum confidence for detecting hands.
#min_tracking_confidence=0.5: Minimum confidence for tracking hands.
#mp_drawing = mp.solutions.drawing_utils: Access the drawing utilities to visualize hand landmarks.

# Function to recognize gestures based on hand landmarks
def recognize_gesture(hand_landmarks):
    landmarks_y = [hand_landmarks.landmark[i].y for i in range(21)]
    avg_y = sum(landmarks_y) / len(landmarks_y)
    std_dev_y = math.sqrt(sum((y - avg_y) ** 2 for y in landmarks_y) / len(landmarks_y))

#landmarks_y = [hand_landmarks.landmark[i].y for i in range(21)]: Extract the y-coordinates of all 21 landmarks.
#avg_y = sum(landmarks_y) / len(landmarks_y): Calculate the average y-coordinate.
#std_dev_y = math.sqrt(...): Calculate the standard deviation of the y-coordinates.

    if std_dev_y < 0.03:
        return "Closed"

#if std_dev_y < 0.03:: Check if the standard deviation is small, indicating a closed hand.
#return "Closed": Return "Closed" if the hand is closed.

    else:
        thumb_y = hand_landmarks.landmark[4].y
        index_y = hand_landmarks.landmark[8].y
        middle_y = hand_landmarks.landmark[12].y
        ring_y = hand_landmarks.landmark[16].y
        pinkie_y = hand_landmarks.landmark[20].y

        if thumb_y < index_y and index_y < middle_y and middle_y < ring_y and ring_y < pinkie_y:
            return "A"
        elif thumb_y > index_y and thumb_y > middle_y and thumb_y > ring_y and thumb_y > pinkie_y:
            return "B"
        elif middle_y < thumb_y and ring_y < thumb_y:
            return "C"
        elif index_y < thumb_y and middle_y > thumb_y and ring_y > thumb_y and pinkie_y > thumb_y:
            return "D"
        elif index_y < thumb_y and middle_y < thumb_y and ring_y < thumb_y and pinkie_y < thumb_y:
            return "E"
        elif thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinkie_y:
            return "F"
        elif index_y < thumb_y and middle_y > thumb_y and ring_y < thumb_y and pinkie_y < thumb_y:
            return "G"
        elif index_y < thumb_y and middle_y > thumb_y and ring_y > thumb_y and pinkie_y < thumb_y:
            return "H"
        elif index_y < thumb_y and middle_y < thumb_y and ring_y > thumb_y and pinkie_y > thumb_y:
            return "I"
        elif thumb_y > index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinkie_y:
            return "J"
        elif index_y > thumb_y and middle_y < thumb_y and ring_y < thumb_y and pinkie_y < thumb_y:
            return "K"
        elif index_y > thumb_y and middle_y < thumb_y and ring_y < thumb_y and pinkie_y < thumb_y:
            return "L"
        elif index_y > thumb_y and middle_y > thumb_y and ring_y < thumb_y and pinkie_y < thumb_y:
            return "M"
        elif index_y > thumb_y and middle_y > thumb_y and ring_y > thumb_y and pinkie_y < thumb_y:
            return "N"
        elif index_y < thumb_y and middle_y < thumb_y and ring_y < thumb_y and pinkie_y > thumb_y:
            return "O"
        elif thumb_y < index_y and thumb_y > middle_y and thumb_y > ring_y and thumb_y < pinkie_y:
            return "P"
        elif thumb_y < index_y and thumb_y < middle_y and thumb_y > ring_y and thumb_y < pinkie_y:
            return "Q"
        elif thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y > pinkie_y:
            return "R"
        elif thumb_y > index_y and thumb_y > middle_y and thumb_y < ring_y and thumb_y > pinkie_y:
            return "S"
        elif thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinkie_y:
            return "T"
        elif index_y > thumb_y and middle_y > thumb_y and ring_y < thumb_y and pinkie_y > thumb_y:
            return "U"
        elif index_y < thumb_y and middle_y > thumb_y and ring_y < thumb_y and pinkie_y > thumb_y:
            return "V"
        elif index_y < thumb_y and middle_y > thumb_y and ring_y > thumb_y and pinkie_y > thumb_y:
            return "W"
        elif index_y > thumb_y and middle_y < thumb_y and ring_y > thumb_y and pinkie_y < thumb_y:
            return "X"
        elif thumb_y > index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y > pinkie_y:
            return "Y"
        elif thumb_y > index_y and thumb_y > middle_y and thumb_y < ring_y and thumb_y < pinkie_y:
            return "Z"
        else:
            return "Unknown"

def detect_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)
            if gesture is not None:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = detect_hand_landmarks(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Adding use_reloader=False to prevent reloader issues
