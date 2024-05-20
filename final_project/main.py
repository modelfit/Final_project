from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import torch as trch
import base64
import requests 

app = Flask(__name__)

TOKEN = '7021156918:AAF39N5VL6YgwnXaw91aaeH_eCY8eG7LBJM'
chat_id = 705535404

# Load models
MODEL_PATH_X = "C:\\Users\\hiash\\OneDrive\\سطح المكتب\\Project\\SeniorProject2Data\\FIRE_YOLOX_MODEL\\detect\\train\\weights\\best.pt"
MODEL_PATH_N = "C:\\Users\\hiash\\OneDrive\\سطح المكتب\\Project\\SeniorProject2Data\\FIRE_YOLON_MODEL\\detect\\train\\weights\\best.pt"

model_x = YOLO(MODEL_PATH_X, verbose=True)
model_n = YOLO(MODEL_PATH_N, verbose=True)

frame_counter = 0
zones = []

def generate_zones(frame_width, frame_height, num_rows, num_cols):
    zone_width = frame_width // num_cols
    zone_height = frame_height // num_rows
    return [(col * zone_width, row * zone_height, (col + 1) * zone_width, (row + 1) * zone_height) 
            for row in range(num_rows) for col in range(num_cols)]

def get_zone(x, y, zones):
    for idx, (x1, y1, x2, y2) in enumerate(zones):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return (x1, y1, x2, y2)
    return None

def decode_frame(data):
    frame_data = base64.b64decode(data['frame'])
    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame

def predict_with_model_n(frame):
    return model_n.predict(frame, verbose=True)

def predict_with_model_x(frame):
    return model_x.predict(frame, verbose=True)

def process_detections(results, frame, thresh_n, thresh_x, num_rows, num_cols):
    global frame_counter, zones

    detected_zones = []

    for result in results:
        for box in result.boxes:
            if box.conf > thresh_n:
                frame_counter += 1
                if frame_counter % thresh_n == 0:
                    results_x = predict_with_model_x(frame)
                    for res_x in results_x:
                        if res_x.boxes:
                            max_conf = trch.max(res_x.boxes.conf).item()
                            if max_conf >= thresh_x:
                                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                                zone = get_zone(x_center, y_center, zones)
                                zone_index = calculate_zone_index(x_center, y_center, frame.shape[1], frame.shape[0], num_cols, num_rows)
                                detected_zones.append({'zone': zone, 'index': zone_index, 'confidence': max_conf})
                                message = f"fire detected at zone:{zone_index} with confidence of {max_conf}"
                                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
                                r = requests.get(url)
                                print(r.json())


    return detected_zones

def calculate_zone_index(x, y, frame_width, frame_height, num_cols, num_rows):
    col_width = frame_width / num_cols
    row_height = frame_height / num_rows
    col = int(x // col_width)
    row = int(y // row_height)
    return col + row * num_cols
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    return frame_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global zones

    data = request.get_json()
    frame = decode_frame(data)

    thresh_n = float(data.get('thresh_n', 0.5))
    thresh_x = float(data.get('thresh_x', 0.7))
    num_rows = data.get('num_rows', 6)
    num_cols = data.get('num_cols', 6)

    if not zones:
        zones = generate_zones(frame.shape[1], frame.shape[0], num_rows, num_cols)

    results_n = predict_with_model_n(frame)
    detected_zones = process_detections(results_n, frame, thresh_n, thresh_x, num_rows, num_cols)

    frame_data = encode_frame(frame)

    return jsonify({'detected_zones': detected_zones, 'frame': frame_data})
if __name__ == '__main__':
    app.run(debug=True)
