from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='web_temp')

model = joblib.load('MediAI-Disease-Predictor/p_app/model.pkl')
Disease_label = joblib.load('MediAI-Disease-Predictor/p_app/Disease_encoder.pkl')
trained_features = joblib.load('MediAI-Disease-Predictor/p_app/trained_features.pkl')

mix_dataset = pd.read_csv('MediAI-Disease-Predictor/dataset/symptom_precaution.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptom_text = request.form['symptoms']
        user_symptoms = [s.strip().replace(" ", "_").lower() for s in symptom_text.split(',')]

        input_dict = {symptom: 0 for symptom in trained_features}
        for symptom in user_symptoms:
            if symptom in input_dict:
                input_dict[symptom] = 1

        input_df = pd.DataFrame([input_dict])[trained_features]

        prediction = model.predict(input_df)[0]
        predicted_disease = Disease_label.inverse_transform([prediction])[0]

        precaution_row = mix_dataset[mix_dataset['Disease'].str.strip() == predicted_disease.strip()]

        if not precaution_row.empty:
            row = precaution_row.iloc[0]
            precautions = [
                str(row['Precaution_1']),
                str(row['Precaution_2']),
                str(row['Precaution_3']),
                str(row['Precaution_4'])
            ]
        else:
            precautions = ["No precautions available"]

        return render_template('result.html',
                               prediction=predicted_disease,
                               precautions=precautions)

if __name__ == '__main__':
    

    app.run(debug=True)
