from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import gdown

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model6.h5')
MODEL_FILE_ID = '1Ep5f-w_viKroovioBpZSE2paZccBlFaf'  # Replace with your actual file ID
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'

# Create necessary directories
UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Function to square crop the frames
def square_crop_frame(image):
    height, width = image.shape[:2]
    min_dimension = min(height, width)
    start_x = (width - min_dimension) // 2
    start_y = (height - min_dimension) // 2
    return image[start_y:start_y + min_dimension, start_x:start_x + min_dimension]

# Process video frames
def process_video_frames(video_path, max_frames=0, resize_dims=(IMG_SIZE, IMG_SIZE)):
    capture = cv2.VideoCapture(video_path)
    processed_frames = []
    try:
        while True:
            read_success, frame = capture.read()
            if not read_success:
                break
            frame = square_crop_frame(frame)
            frame = cv2.resize(frame, resize_dims)
            frame = frame[..., ::-1]  # Convert BGR to RGB
            processed_frames.append(frame)
            if max_frames > 0 and len(processed_frames) >= max_frames:
                break
    finally:
        capture.release()
    return np.array(processed_frames)

# Extract features from frames
def extract_video_features(video_frames, feature_extractor):
    features = []
    for frame in video_frames:
        frame = np.expand_dims(frame, axis=0)
        feature = feature_extractor.predict(frame)
        features.append(feature)
    return np.array(features).squeeze()

# Build feature extractor
def build_feature_extractor(model_name='ResNet50'):
    base_model_class = getattr(keras.applications, model_name)
    base_model = base_model_class(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = getattr(keras.applications, model_name.lower()).preprocess_input

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)
    outputs = base_model(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_feature_extractor")
    return model

feature_extractor = build_feature_extractor()

# Prepare model input
def prepare_input(features, max_seq_length=MAX_SEQ_LENGTH):
    num_features = features.shape[-1]
    input_features = np.zeros((1, max_seq_length, num_features), dtype="float32")
    input_mask = np.zeros((1, max_seq_length), dtype=bool)
    frames_to_use = min(max_seq_length, features.shape[0])
    input_features[0, :frames_to_use] = features[:frames_to_use]
    input_mask[0, :frames_to_use] = True
    return input_features, input_mask

@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    video_frames = process_video_frames(video_path, max_frames=MAX_SEQ_LENGTH)
    video_features = extract_video_features(video_frames, feature_extractor)

    input_features, input_mask = prepare_input(video_features)

    frame_predictions = []
    frame_urls = []
    for i in range(video_features.shape[0]):
        temp_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        temp_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
        temp_features[0, 0] = video_features[i]
        temp_mask[0, 0] = True

        frame_prediction = model.predict([temp_features, temp_mask])
        frame_predictions.append("REAL" if frame_prediction[0] > 0.59136045 else "FAKE")

        frame_filename = os.path.join(FRAME_FOLDER, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, video_frames[i])
        frame_urls.append(f"http://localhost:5000/frames/frame_{i}.jpg")

    fake_count = frame_predictions.count("FAKE")
    total_frames = len(frame_predictions)
    final_prediction = "FAKE" if (fake_count / total_frames) > 0.3 else "REAL"

    return jsonify({
        "frame_predictions": frame_predictions,
        "frame_urls": frame_urls,
        "final_prediction": final_prediction
    })

@app.route("/frames/<filename>")
def serve_frame(filename):
    return send_from_directory(FRAME_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
