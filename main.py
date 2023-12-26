from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

class PneumoniaDetector:
    def __init__(self):
        self.image_path = ""
        self.model = load_model('best_model.h5')

    def perform_pneumonia_prediction(self):
        try:
            if self.image_path:
                img = Image.open(self.image_path)
                img = img.resize((150, 150))
                img = np.array(img) / 255.0
                img = img.reshape(-1, 150, 150, 1)

                prediction = self.model.predict(img)[0][0]
                result = 'Normal' if prediction > 0.5 else 'Pneumonia'

                return result
            else:
                return "No image selected."
        except Exception as e:
            return f"Error occurred during pneumonia prediction: {e}"

class BrainTumorDetector:
    def __init__(self):
        self.image_path = ""
        self.model = load_model('brain_model.h5')

    def perform_brain_tumor_prediction(self):
        try:
            if self.image_path:
                img = Image.open(self.image_path)
                img = img.resize((128, 128))
                x = np.array(img)
                x = x.reshape(1, 128, 128, 3)
                res = self.model.predict(x)
                classification = np.argmax(res)

                prediction_text = "Tumor" if classification == 0 else "Not a tumor"
                return prediction_text
            else:
                return "No image selected."
        except Exception as e:
            return f"Error occurred during brain tumor prediction: {e}"

pneumonia_detector = PneumoniaDetector()
brain_tumor_detector = BrainTumorDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/predict_pneumonia', methods=['POST'])
def predict_pneumonia():
    result = pneumonia_detector.perform_pneumonia_prediction()
    return jsonify({'prediction': result})

@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor():
    result = brain_tumor_detector.perform_brain_tumor_prediction()
    return jsonify({'prediction': result})


@app.route('/brain_tumor')
def brain_tumor():
    return render_template('brain_tumor.html')

@app.route('/upload_pneumonia', methods=['POST'])
def upload_pneumonia():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = 'temp_image.jpg'
    file.save(file_path)
    pneumonia_detector.image_path = file_path

    return jsonify({'message': 'Image uploaded for pneumonia detection'})

@app.route('/upload_brain_tumor', methods=['POST'])
def upload_brain_tumor():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = 'temp_image.jpg'
    file.save(file_path)
    brain_tumor_detector.image_path = file_path

    return jsonify({'message': 'Image uploaded for brain tumor detection'})

if __name__ == '__main__':
    app.run(debug=True)