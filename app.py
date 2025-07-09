from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and encoders
et_crop = joblib.load("extra_tree_crop_model.pkl")
et_fert = joblib.load("extra_tree_fertilizer_model.pkl")
le_soil = joblib.load("label_encoder_soil.pkl")
le_crop = joblib.load("label_encoder_crop.pkl")
le_fert = joblib.load("label_encoder_fertilizer.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    soil_color = data['soil_color']
    nitrogen = float(data['nitrogen'])
    phosphorus = float(data['phosphorus'])
    potassium = float(data['potassium'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])
    temperature = float(data['temperature'])

    soil_color_enc = le_soil.transform([soil_color])[0]
    input_features = np.array([[soil_color_enc, nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

    # Predict crop
    crop_probs = et_crop.predict_proba(input_features)[0]
    best_crop_idx = np.argmax(crop_probs)
    best_crop = le_crop.inverse_transform([best_crop_idx])[0]

    # Predict fertilizer
    fert_features = np.array([[best_crop_idx, soil_color_enc, nitrogen, phosphorus, potassium, ph, rainfall, temperature]])
    fert_probs = et_fert.predict_proba(fert_features)[0]
    best_fert_idx = np.argmax(fert_probs)
    best_fert = le_fert.inverse_transform([best_fert_idx])[0]

    # Alternatives
    alt_crop_indices = crop_probs.argsort()[-4:][::-1]
    alt_crop_indices = [idx for idx in alt_crop_indices if idx != best_crop_idx][:3]
    alternatives = []
    for idx in alt_crop_indices:
        alt_crop = le_crop.inverse_transform([idx])[0]
        fert_features_alt = np.array([[idx, soil_color_enc, nitrogen, phosphorus, potassium, ph, rainfall, temperature]])
        fert_probs_alt = et_fert.predict_proba(fert_features_alt)[0]
        alt_fert_idx = np.argmax(fert_probs_alt)
        alt_fert = le_fert.inverse_transform([alt_fert_idx])[0]
        alternatives.append({'crop': alt_crop, 'fertilizer': alt_fert})

    return jsonify({
        'crop': best_crop,
        'fertilizer': best_fert,
        'alternatives': alternatives
    })

if __name__ == '__main__':
    app.run(debug=True)
