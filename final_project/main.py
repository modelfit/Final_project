from typing import List
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import torch as trch
import base64
import requests
import os

app = Flask(__name__)


TELEGRAM_CHAT_IDs = [705535404,822395344]
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
MODEL_PATH_X = os.path.join(script_dir, "best_x.pt")
MODEL_PATH_N = os.path.join(script_dir, "best_n.pt")
model_x = YOLO(MODEL_PATH_X, verbose=False)
model_n = YOLO(MODEL_PATH_N, verbose=False)

zones = []



def send_telegram_message(message:str,ids:List[int] = TELEGRAM_CHAT_IDs):
    results = []
    TOKEN = '7021156918:AAF39N5VL6YgwnXaw91aaeH_eCY8eG7LBJM'
    for id in ids:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={id}&text={message}"
        r = requests.get(url)
        if r.ok:
            results.append(r.json())
        else:
            results.append(None)
    return results





def generate_zones(frame_width, frame_height, num_rows, num_cols):
    zone_width = frame_width // num_cols
    zone_height = frame_height // num_rows
    return [(col * zone_width, row * zone_height, (col + 1) * zone_width, (row + 1) * zone_height) 
            for row in range(num_rows) for col in range(num_cols)]

def generate_normalized_zones(num_rows, num_cols):
    zone_width_rel = 1.0 / num_cols
    zone_height_rel = 1.0 / num_rows
    zones = []
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * zone_width_rel
            y1 = row * zone_height_rel
            x2 = (col + 1) * zone_width_rel
            y2 = (row + 1) * zone_height_rel
            zones.append([x1, y1, x2, y2])
    return zones

def get_zone(x, y, zones, frame_width, frame_height):
    normalized_x = x / frame_width
    normalized_y = y / frame_height
    for idx, (x1, y1, x2, y2) in enumerate(zones):
        if x1 <= normalized_x <= x2 and y1 <= normalized_y <= y2:
            return (x1, y1, x2, y2)
    return None

def decode_frame(data):
    frame_data = base64.b64decode(data['frame'])
    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame

def predict_with_model_n(frame):
    return model_n.predict(frame, verbose=False)

def predict_with_model_x(frame):
    return model_x.predict(frame, verbose=False)

def process_detections(results, frame, thresh_n, thresh_x, num_rows, num_cols, zones):
    global frame_counter

    detected_zones = []

    for result in results:
        for box in result.boxes:
            if box.conf > thresh_n:
                # Directly use model_x prediction for further processing
                results_x = predict_with_model_x(frame)
                for res_x in results_x:
                    if res_x.boxes:
                        max_conf = trch.max(res_x.boxes.conf).item()
                        if max_conf >= thresh_x:
                            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                            zone = get_zone(x_center, y_center, zones, frame.shape[1], frame.shape[0])
                            zone_index = calculate_zone_index(x_center, y_center, frame.shape[1], frame.shape[0], num_cols, num_rows)
                            detected_zones.append({'zone': zone, 'index': zone_index, 'confidence': max_conf})
                            message = f"fire detected at zone:{zone_index+1} with confidence of {max_conf}"
                            send_telegram_message(message)

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
    print(f"Received thresholds: thresh_n={thresh_n}, thresh_x={thresh_x}")  # Debugging line



    if not zones:
        zones = generate_normalized_zones(num_rows, num_cols)
    results_n = predict_with_model_n(frame)
    #print(f"Initial detections (model_n): {results_n}")
    detected_zones = process_detections(results_n, frame, thresh_n, thresh_x, num_rows, num_cols, zones)
    print(f"Detected zones after processing: {detected_zones}")

    frame_data = encode_frame(frame)

    return jsonify({'detected_zones': detected_zones, 'frame': frame_data})

if __name__ == '__main__':
    app.run(debug=True)
