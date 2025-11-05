import os
from flask import Flask, render_template, request
import numpy as np
import joblib

# Define base and template directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load model and scaler safely
MODEL_PATH = os.path.join(BASE_DIR, 'BreastCancerPredictionModel.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('index.html', prediction_text="Error: Model or Scaler not loaded.")

    try:
        # Extract input features from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Scale the input
        scaled_features = scaler.transform(final_features)

        # Predict
        prediction = model.predict(scaled_features)
        output = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer Detected)"

        return render_template('index.html', prediction_text=f'🩺 Cancer Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'⚠️ Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
