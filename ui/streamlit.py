import streamlit as st
import numpy as np
import joblib


model = joblib.load("decision_tree_model.joblib")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("💓 Heart Disease Risk Prediction")
st.markdown("Enter patient data to predict risk of heart disease.")


age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina?", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)


st.markdown("### Chest Pain Type:")
cp_1 = st.checkbox("Chest Pain Type 1")
cp_2 = st.checkbox("Chest Pain Type 2")
cp_3 = st.checkbox("Chest Pain Type 3")

st.markdown("### Resting ECG:")
restecg_1 = st.checkbox("Rest ECG 1")
restecg_2 = st.checkbox("Rest ECG 2")

st.markdown("### Slope of ST Segment:")
slope_1 = st.checkbox("Slope 1")
slope_2 = st.checkbox("Slope 2")

st.markdown("### Thalassemia:")
thal_1 = st.checkbox("Thal 1")
thal_2 = st.checkbox("Thal 2")
thal_3 = st.checkbox("Thal 3")


sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

input_data = np.array([
    age, chol, thalach, oldpeak, ca,
    int(cp_1), int(cp_2), int(cp_3),
    int(restecg_1), int(restecg_2),
    int(slope_1), int(slope_2),
    int(thal_1), int(thal_2), int(thal_3),
    sex_val, fbs_val, exang_val, trestbps
]).reshape(1, -1)


if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")