from flask import Flask, render_template, request
import joblib
import pandas as pd

# Flask app setup
app = Flask(__name__, template_folder="web_temp")



model = joblib.load("MediAI-Disease-Predictor/p_app/model.pkl")
disease_encoder = joblib.load("MediAI-Disease-Predictor/p_app/Disease_encoder.pkl")
feature_list = joblib.load("MediAI-Disease-Predictor/p_app/trained_features.pkl")

precaution_data = pd.read_csv("MediAI-Disease-Predictor/dataset/symptom_precaution.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        sympt_txt = request.form.get("symptoms", "")
        symptoms_entered = [sympt.strip().replace(" ", "_").lower() for sympt in sympt_txt.split(",")]
        input_values = {feature: 0 for feature in feature_list}
        for s in symptoms_entered:
            if s in input_values:
                input_values[s] = 1
        df_input = pd.DataFrame([input_values])[feature_list]
        pred_index = model.predict(df_input)[0]
        disease_name = disease_encoder.inverse_transform([pred_index])[0]
        matched_row = precaution_data[precaution_data["Disease"].str.strip() == disease_name.strip()]

        if not matched_row.empty:
            row = matched_row.iloc[0]
            precautions_list = [
                str(row["Precaution_1"]),
                str(row["Precaution_2"]),
                str(row["Precaution_3"]),
                str(row["Precaution_4"])
            ]
        else:
            precautions_list = ["No specific precautions found"]

        return render_template("result.html",
                               prediction=disease_name,
                               precautions=precautions_list)

if __name__ == "__main__":
    app.run(debug=True)
