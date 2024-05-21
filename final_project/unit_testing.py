import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, json
import base64
import cv2
import numpy as np
from your_module import app, send_telegram_message, generate_zones, generate_normalized_zones, decode_frame, predict_with_model_n, predict_with_model_x, process_detections, get_zone, encode_frame, calculate_zone_index

class TestFireDetectionApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('requests.get')
    def test_send_telegram_message(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {'ok': True}
        mock_get.return_value = mock_response

        result = send_telegram_message('Test message')
        self.assertTrue(all(r['ok'] for r in result))

    def test_generate_zones(self):
        frame_width, frame_height = 640, 480
        num_rows, num_cols = 4, 4
        zones = generate_zones(frame_width, frame_height, num_rows, num_cols)
        self.assertEqual(len(zones), num_rows * num_cols)
        self.assertEqual(zones[0], (0, 0, 160, 120))

    def test_generate_normalized_zones(self):
        num_rows, num_cols = 4, 4
        zones = generate_normalized_zones(num_rows, num_cols)
        self.assertEqual(len(zones), num_rows * num_cols)
        self.assertEqual(zones[0], [0.0, 0.0, 0.25, 0.25])

    def test_decode_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        decoded_frame = decode_frame({'frame': frame_data})
        self.assertEqual(decoded_frame.shape, frame.shape)

    @patch('your_module.model_n.predict')
    def test_predict_with_model_n(self, mock_predict):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_predict.return_value = [mock_result]
        result = predict_with_model_n(frame)
        self.assertEqual(result, [mock_result])

    @patch('your_module.model_x.predict')
    def test_predict_with_model_x(self, mock_predict):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_predict.return_value = [mock_result]
        result = predict_with_model_x(frame)
        self.assertEqual(result, [mock_result])

    @patch('your_module.predict_with_model_x')
    def test_process_detections(self, mock_predict_with_model_x):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result_x = MagicMock()
        mock_result_x.boxes = [MagicMock(conf=0.8)]
        mock_predict_with_model_x.return_value = [mock_result_x]

        results = [MagicMock(boxes=[MagicMock(conf=0.6, xyxy=[[100, 100, 200, 200]])])]
        num_rows, num_cols = 4, 4
        zones = generate_normalized_zones(num_rows, num_cols)
        detected_zones = process_detections(results, frame, 0.5, 0.7, num_rows, num_cols, zones)

        self.assertTrue(len(detected_zones) > 0)

    def test_get_zone(self):
        zones = generate_normalized_zones(4, 4)
        frame_width, frame_height = 640, 480
        zone = get_zone(320, 240, zones, frame_width, frame_height)
        self.assertIsNotNone(zone)

    def test_encode_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_data = encode_frame(frame)
        self.assertIsInstance(frame_data, str)

    def test_calculate_zone_index(self):
        frame_width, frame_height = 640, 480
        num_cols, num_rows = 4, 4
        index = calculate_zone_index(320, 240, frame_width, frame_height, num_cols, num_rows)
        self.assertEqual(index, 10)

if __name__ == '__main__':
    unittest.main()
