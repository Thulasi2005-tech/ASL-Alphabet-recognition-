"# ASL-Alphabet-recognition-" 
# ASL Alphabet Recognition System

A Flask-based web app for real-time American Sign Language (ASL) alphabet recognition using Convolutional Neural Networks (CNNs). It supports both live webcam detection and image upload prediction, making it a powerful tool for bridging communication gaps for the Deaf and hard-of-hearing community.

---

## Key Features

### Live Webcam ASL Detection
- Real-time hand gesture recognition using a webcam.
- High confidence predictions with on-screen feedback.
- Adjustable threshold for accurate classification.

### Image Upload ASL Detection
- Upload images of ASL alphabets for instant prediction.
- Automatic image resizing and pre-processing for model compatibility.

### Flask Integration
- Clean, user-friendly web interface built with Flask.
- Separate pages for live webcam detection, image upload, and home screen.

### Efficient Model Handling
- Uses a pre-trained CNN model for fast and accurate predictions.
- Supports image processing with OpenCV and Keras.

---

## Project Structure

ASL-Recognition/
│
├── application.py # Flask backend for model integration
├── templates/
│ ├── home.html # Main home page
│ ├── webcam.html # Live webcam detection page
│ ├── upload.html # Image upload page
│ └── result.html # Image prediction result page
├── static/
│ └── uploads/ # Directory for uploaded images
├── asl_better_model.h5 # Pre-trained ASL alphabet recognition model
├── class_indices.json # Class label mapping for model predictions
└── README.md # Project documentation (this file)

yaml
Copy
Edit

---

## 🚀 How to Run

### Clone the Repository
```bash
git clone https://github.com/your-username/ASL-Recognition.git
cd ASL-Recognition
Install Dependencies
bash
Copy
Edit
pip install flask keras numpy opencv-python
Run the Application
bash
Copy
Edit
python application.py
Access the Web App
Open your browser and go to: http://127.0.0.1:5000

🛠️ Future Enhancements
Full ASL word and sentence recognition.

Support for dynamic gestures and two-hand detection.

Mobile and edge device deployment for offline usage.

Multilingual support for other sign languages.

