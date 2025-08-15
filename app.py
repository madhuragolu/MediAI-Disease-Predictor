from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder='web_temp', static_folder='static')


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pkl_files" / "model1.pkl"
ENCODER_PATH = BASE_DIR / "pkl_files" / "Disease_encoder1.pkl"
FEATURES_PATH = BASE_DIR / "pkl_files" / "trained_features1.pkl"

model = joblib.load(MODEL_PATH)
disease_encoder = joblib.load(ENCODER_PATH)
feature_list = joblib.load(FEATURES_PATH)

PRECAUTIONS_CSV = BASE_DIR / "dataset" / "symptom_precaution.csv"

precautions_df = pd.read_csv(PRECAUTIONS_CSV)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle symptom input and return prediction + precautions."""
    
    symptoms_list = request.form.getlist('symptoms')
    input_data = {symptom: 0 for symptom in feature_list}
    
    for s in symptoms_list:
        if s in input_data:
            input_data[s] = 1

    input_df = pd.DataFrame([input_data])[feature_list]

    predicted_index = model.predict(input_df)[0]
    predicted_disease = disease_encoder.inverse_transform([predicted_index])[0]

    disease_data = precautions_df[precautions_df['Disease'].str.strip() == predicted_disease.strip()]
    
    if not disease_data.empty:
        row = disease_data.iloc[0]
        precautions = [str(row[f'Precaution_{i}']) for i in range(1, 5)]
    else:
        precautions = ["No precautions found"]

    return render_template(
        'result.html',
        prediction=predicted_disease,
        precautions=precautions
    )

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000) 
