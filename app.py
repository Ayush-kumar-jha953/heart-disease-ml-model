import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Heart Failure Predictor", layout="centered")
st.title("❤️ Heart Failure Risk Predictor")
st.write("Input patient clinical metrics below to predict the probability of heart disease.")

# 2. Load the Saved Model
@st.cache_resource
def load_model():
    with open('heart_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.subheader("Patient Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    # Sex mapping: Male=0, Female=1
    sex_input = st.selectbox("Sex", options=["Male", "Female"])
    sex = 0 if sex_input == "Male" else 1
    
    # ChestPainType mapping: ASY=0, ATA=1, NAP=2, TA=3
    cp_input = st.selectbox("Chest Pain Type", options=["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina (TA)"])
    cp_map = {"Asymptomatic (ASY)": 0, "Atypical Angina (ATA)": 1, "Non-Anginal Pain (NAP)": 2, "Typical Angina (TA)": 3}
    chest_pain = cp_map[cp_input]
    
    cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=0.0, max_value=500.0, value=200.0)
    
    # FastingBS mapping: No=0, Yes=1 (if FastingBS > 120 mg/dl)
    fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    fasting_bs = 1 if fbs_input == "Yes" else 0

with col2:
    # RestingECG mapping: Normal=0, ST=1, LVH=2
    ecg_input = st.selectbox("Resting ECG Results", options=["Normal", "ST-T wave abnormality (ST)", "Left ventricular hypertrophy (LVH)"])
    ecg_map = {"Normal": 0, "ST-T wave abnormality (ST)": 1, "Left ventricular hypertrophy (LVH)": 2}
    resting_ecg = ecg_map[ecg_input]
    
    # ExerciseAngina mapping: No=0, Yes=1
    angina_input = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    exercise_angina = 1 if angina_input == "Yes" else 0
    
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    # ST_Slope mapping: Flat=0, Up=1, Down=2
    slope_input = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Flat", "Up", "Down"])
    slope_map = {"Flat": 0, "Up": 1, "Down": 2}
    st_slope = slope_map[slope_input]

# 4. Prediction Logic
if st.button("Analyze Risk Profile", type="primary"):
    # Features arranged in the EXACT order of your final dataframe columns:
    # ['Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    input_features = np.array([[sex, chest_pain, cholesterol, fasting_bs, resting_ecg, exercise_angina, oldpeak, st_slope]])
    
    # Predict prediction class and probability
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1] * 100
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"### ⚠️ High Risk: Heart Disease Detected")
        st.write(f"The model estimates a **{probability:.2f}%** probability of heart failure based on these clinical metrics.")
    else:
        st.success(f"### ✅ Low Risk: Normal Profile")
        st.write(f"The model estimates a **{probability:.2f}%** probability of heart failure. Patient profile appears low risk.")
