from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Load your pre-trained model here
model = load_model('traffic_sign_model.h5')  # Adjusted path

@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html file

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    # Decode the image
    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))  # Resize to your model's input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)

        # Map predicted class to traffic sign labels (update the labels accordingly)
        sign_labels = {0: 'Stop Sign', 1: 'No Parking', 2: 'Traffic Signal', 3: 'One Way', 4: 'Narrow Bridge'}
        response = {
            'success': True,
            'predictions': [{'sign': sign_labels[i], 'confidence': float(predictions[0][i] * 100)} for i in range(len(predictions[0]))]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
