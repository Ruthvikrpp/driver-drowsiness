from flask import Flask, render_template, Response, session, jsonify
import cv2
import tensorflow as tf
import numpy as np
import pygame
from pygame import mixer
import os
import time  # Import the time module

pygame.mixer.init()

app = Flask(__name__)
app.secret_key = "your_very_secret_key"

# --- Model and Cascade Loading ---
try:
    model = tf.keras.models.load_model('inceptionv3.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

base_dir = os.path.dirname(os.path.abspath(__file__))
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(base_dir, 'haarcascade_eye.xml')

if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print("Face cascade loaded.")
else:
    print(f"Error: Face cascade XML not found at {face_cascade_path}")
    face_cascade = None

if os.path.exists(eye_cascade_path):
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    print("Eye cascade loaded.")
else:
    print(f"Error: Eye cascade XML not found at {eye_cascade_path}")
    eye_cascade = None

drowsiness_score = 0
alarm_on = False
head_movements = 0
head_threshold = 5
movement_detected = False
no_face_start_time = None  # To track when "No Face Detected" starts
no_face_threshold_seconds = 1.5  # Duration for triggering alarm
previous_face_position = None # To store the previous face position


def preprocess(eye_frame):
    processed_frame = cv2.resize(eye_frame, (80, 80))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = processed_frame / 255.0
    return processed_frame


def detect_drowsiness(frame, previous_frame=None):
    global drowsiness_score, alarm_on, head_movements, head_threshold, movement_detected, no_face_start_time, previous_face_position

    if model is None or face_cascade is None or eye_cascade is None:
        cv2.putText(frame, "Error: Model or Cascades not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, "Error"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    status_text = "No Face Detected"
    current_face_position = None

    if len(faces) > 0:
        # Face detected, reset the no_face timer
        no_face_start_time = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            current_face_position = (x + w // 2, y + h // 2)

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(eyes) == 0:
                status_text = "Eyes Not Detected"
            else:
                status_text = "Eyes Open"

            eyes_closed = False
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
                processed_eye = preprocess(eye_roi)
                prediction = model.predict(processed_eye)
                if prediction[0][0] > prediction[0][1]:
                    eyes_closed = True
                    break

            if eyes_closed:
                drowsiness_score += 1
                status_text = "Eyes Closed"
                if drowsiness_score > 10:
                    if not alarm_on and not pygame.mixer.music.get_busy():
                        try:
                            pygame.mixer.music.load('alarm.wav')
                            pygame.mixer.music.play(-1)
                            alarm_on = True
                            print("Alarm PLAYING (Eyes Closed)")
                        except pygame.error as e:
                            print(f"Pygame error playing sound: {e}")
            else:
                status_text = "Eyes Open"
                if alarm_on:
                    if previous_frame is not None and previous_face_position is not None and current_face_position is not None:
                        distance = ((current_face_position[0] - previous_face_position[0]) ** 2 +
                                    (current_face_position[1] - previous_face_position[1]) ** 2) ** 0.5
                        if distance > 20:
                            movement_detected = True

                    if not eyes_closed and movement_detected:
                        head_movements += 1
                        print(f"Head Movement Detected: {head_movements}")
                        if head_movements >= head_threshold:
                            pygame.mixer.music.stop()
                            pygame.mixer.music.unload()
                            alarm_on = False
                            drowsiness_score = 0
                            head_movements = 0
                            movement_detected = False
                            print("Alarm STOPPED (Eyes Open & Head Backup)")
                    elif not eyes_closed:
                        movement_detected = False
            break  # Process first detected face
        previous_face_position = current_face_position #update previous face position
    else:
        # No face detected
        if no_face_start_time is None:
            no_face_start_time = time.time()
        elif time.time() - no_face_start_time >= no_face_threshold_seconds:
            if not alarm_on and not pygame.mixer.music.get_busy():
                try:
                    pygame.mixer.music.load('alarm.wav')
                    pygame.mixer.music.play(-1)
                    alarm_on = True
                    print("Alarm PLAYING (No Face Detected)")
                except pygame.error as e:
                    print(f"Pygame error playing sound: {e}")

    cv2.putText(frame, f"Score: {drowsiness_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if alarm_on:
        cv2.putText(frame, "ALARM ON", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, status_text



def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    previous_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        processed_frame, _ = detect_drowsiness(frame, previous_frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Error: Could not encode frame.")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        previous_frame = frame.copy()

    cap.release()
    print("Video capture released.")



@app.route('/')
def index():
    session['drowsiness_score'] = 0
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/turnOffAlarm', methods=['POST'])
def turn_off_alarm():
    global alarm_on, drowsiness_score, head_movements, movement_detected, no_face_start_time
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        print("Alarm turned OFF by button.")
    alarm_on = False
    drowsiness_score = 0
    head_movements = 0
    movement_detected = False
    no_face_start_time = None  # Reset the no-face timer when manually turned off
    return jsonify(message='Alarm turned off successfully', alarm_status=False)



if __name__ == '__main__':
    pygame.mixer.init()
    if not pygame.mixer.get_init():
        print("Error: Pygame mixer could not be initialized.")
    else:
        print("Pygame mixer initialized.")
    app.run(debug=True)
