from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import json
import os

app = Flask(__name__)

# Load model and class indices
model = load_model("C:/Users/thula/Documents/Mini_Project/Training File/asl_better_model.h5")
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
index_to_label = {v: k for k, v in class_indices.items()}

# Create uploads directory if it doesn't exist
os.makedirs('static/uploads', exist_ok=True)


# Home Page
@app.route('/')
def index():
    return render_template('home.html')


# Webcam Live Detection Page
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


# Upload Image Page
@app.route('/upload')
def upload():
    return render_template('upload.html')


# Image Prediction Route
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return "No file uploaded."
    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = index_to_label[predicted_class_index]

        return render_template('result.html', label=predicted_label, image=filepath)


# Real-Time Frame Generator
def gen_frames():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = 300, 100, 600, 400
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (64, 64))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_array = img_to_array(roi_rgb) / 255.0
            roi_array = np.expand_dims(roi_array, axis=0)

            prediction = model.predict(roi_array, verbose=0)
            predicted_class = index_to_label[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            threshold = 70

            if confidence > threshold:
                display_text = f"Predicted: {predicted_class}"
                display_conf = f"Confidence: {confidence:.2f}%"
                color = (0, 0, 255)
            else:
                display_text = "Waiting for clearer sign..."
                display_conf = None
                color = (255, 165, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            if display_conf:
                cv2.putText(frame, display_conf, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()
        print("Camera released")


# Video Streaming Route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the App
if __name__ == '__main__':
    app.run(debug=True)
