from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

model = YOLO('model/best.pt')
model.conf = 0.25

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No image URL provided'}), 400

        image_url = data['url']
        resp = requests.get(image_url)
        if resp.status_code != 200:
            return jsonify({'error': 'Could not download image'}), 400

        # Convert to numpy array and decode
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400

        # Run YOLO model
        results = model(img, save=True)
        print(results)

        return jsonify({
            'message': 'Prediction successful',
            'detections': [{
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'box': box.xyxy[0].tolist()
            } for box in results[0].boxes]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
