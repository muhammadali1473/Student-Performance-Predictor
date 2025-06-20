import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("student_predictor/models/predictor_model.pkl")

st.set_page_config(page_title="🎓 Student Performance Predictor", page_icon="🎓")

st.title("📊 Student Final Score Predictor")
st.markdown("Enter the student details to predict their expected final exam score.")

# Inputs
hours = st.slider("Hours Studied", 0.0, 12.0, step=0.5)
sleep = st.slider("Sleep Hours", 0.0, 12.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, step=1)
participation = st.slider("Class Participation (%)", 0, 100, step=1)

if st.button("🎯 Predict Final Score"):
    input_data = np.array([[hours, sleep, attendance, participation]])
    prediction = model.predict(input_data)
    st.success(f"🎓 Predicted Final Score: **{prediction[0]:.2f}**")
