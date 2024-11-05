from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import math
import mysql.connector
from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Database configuration
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'tuturdb'
}

# Map gestures to task IDs
gesture_task_map = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6
}

# Save image to database
def save_frame_to_db(image, gesture, task_progress_id, task_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id = 4
        is_correct = random.randint(0, 1)
        query = """
        INSERT INTO task_progress 
        (task_progress_id, user_id, task_id, is_correct, timestamp, image, gesture) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (task_progress_id, user_id, task_id, is_correct, timestamp, image, gesture))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully inserted into database: {task_progress_id}, {user_id}, {task_id}, {is_correct}, {timestamp}, gesture={gesture}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        print(f"Failed to insert into database: {task_progress_id}, {user_id}, {task_id}, {is_correct}, {timestamp}, gesture={gesture}")

# Function to recognize gestures based on hand landmarks
def recognize_gesture(hand_landmarks):
    landmarks_y = [hand_landmarks.landmark[i].y for i in range(21)]
    avg_y = sum(landmarks_y) / len(landmarks_y)
    std_dev_y = math.sqrt(sum((y - avg_y) ** 2 for y in landmarks_y) / len(landmarks_y))

    if std_dev_y < 0.03:
        return "Closed"
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
        # Add more conditions for other gestures here
        else:
            return "Unknown"

# Function to detect hand landmarks from a frame
def detect_hand_landmarks(frame, task_progress_id, task_id):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        detected_gestures = set()  # To track detected gestures in the current frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)
            if gesture and gesture not in detected_gestures:
                detected_gestures.add(gesture)
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', frame)
                task_id = gesture_task_map.get(gesture, task_id)
                save_frame_to_db(buffer.tobytes(), gesture, task_progress_id, task_id)
    return frame

# Video generator function
def gen():
    cap = cv2.VideoCapture(0)
    task_progress_id = 1
    task_id = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = detect_hand_landmarks(frame, task_progress_id, task_id)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        task_progress_id += 1

@app.route('/')
def index():
    return render_template('assessment.blade.php')

@app.route('/video_feed_assessment')
def video_feed_assessment():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Machine Learning functions
def load_data_from_database(host, user, password, database, table):
    connection_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
    engine = create_engine(connection_str)
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)
    print("Data types:\n", df.dtypes)
    print("First few rows:\n", df.head())
    return df

def preprocess_data(df):
    x = df.drop(['is_correct', 'user_id', 'image', 'gesture', 'timestamp'], axis=1)
    y = df['is_correct']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, x_test.index

def train_random_forest(x_train, y_train, n_estimators=8, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(x_train, y_train)
    return clf

#x_train: The feature matrix of the training data.
#y_train: The target labels of the training data.
#n_estimators: The number of trees in the forest (default is 8).
#random_state: Controls the randomness of the estimator (default is 42).

def evaluate_model(clf, x_train, x_test, y_train, y_test):
    mean_cv_scores = cross_val_score(clf, x_train, y_train, cv=5).mean()
    score_training = accuracy_score(y_train, clf.predict(x_train))
    score_testing = accuracy_score(y_test, clf.predict(x_test))
    return mean_cv_scores, score_training, score_testing

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()

def plot_predictions(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['id'], results_df['actual'], label='Actual', marker='o')
    plt.plot(results_df['id'], results_df['predicted'], label='Predicted', marker='x')
    plt.xlabel('ID')
    plt.ylabel('is_correct')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

def save_predictions_to_database(engine, df, table_name):
    with engine.connect() as connection:
        connection.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            actual BOOLEAN,
            predicted BOOLEAN,
            fluency_improvement VARCHAR(3)
        )
        """))
    df.to_sql(table_name, engine, if_exists='append', index=False)

@app.route('/run_ml_analysis')
def run_ml_analysis():
    host = "localhost"
    user = "root"
    password = ""
    database = "tuturdb"
    table = "task_progress"
    prediction_table = "task_progress_predictions"

    df = load_data_from_database(host, user, password, database, table)
    x_train, x_test, y_train, y_test, test_indices = preprocess_data(df)
    clf = train_random_forest(x_train, y_train)
    y_pred = clf.predict(x_test)
    fluency_improvement = ['Yes' if pred == 1 else 'No' for pred in y_pred]
    results_df = pd.DataFrame({
        'user_id': df.loc[test_indices, 'user_id'],
        'actual': y_test,
        'predicted': y_pred,
        'fluency_improvement': fluency_improvement
    })
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'id'}, inplace=True)
    print("Predictions:")
    print(results_df)
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
    save_predictions_to_database(engine, results_df, prediction_table)
    results_df_from_db = load_data_from_database(host, user, password, database, prediction_table)
    mean_cv_scores, score_training, score_testing = evaluate_model(clf, x_train, x_test, y_train, y_test)
    print("Mean cross-validation: ", mean_cv_scores)
    print("Scores on training: ", score_training)
    print("Scores on testing: ", score_testing)
    plot_confusion_matrix(y_test, y_pred)
    plot_predictions(results_df_from_db)
    num_yes = fluency_improvement.count('Yes')
    num_no = fluency_improvement.count('No')
    if num_yes > num_no:
        print("It seems the user is improving!")
    elif num_yes < num_no:
        print("It seems the user needs more practice.")
    else:
        print("The user's performance is balanced.")
    return "Machine Learning analysis complete. Check the console for details."

if __name__ == "__main__":
    app.run(debug=True)

