import numpy as np
import pickle
import streamlit as st
from pathlib import Path

file_path = Path('Heart_Disease_Prediction-Web-Application/trained_model.sav')
loaded_model = None

try:
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The file '{file_path}' was not found. Please check the file path.")

def heart_disease_prediction(input):
    global loaded_model
    if loaded_model is None:
        st.error("Model not loaded. Please check the file path.")
        return

    in_np = np.asarray(input, dtype=float)
    np_reshape = in_np.reshape(1, -1)
    prediction = loaded_model.predict(np_reshape)

    if prediction[0] == 0:
        return "The person does not have heart disease"
    else:
        return "The person has heart disease"

def main():
    st.title('Heart Disease Prediction Web App')

    age = st.text_input('Age')
    sex = st.text_input('Sex Type')
    cp = st.text_input('Chest Pain type')
    trestbps = st.text_input('Persons Resting Blood Pressure(mm)')
    chol = st.text_input('Cholesterol value')
    fbs = st.text_input('Fasting Blood Sugar')
    restecg = st.text_input('Resting Electrocardiographic Results')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest ')
    slope = st.text_input('Slope of the Peak Exercise ST Segment')
    ca = st.text_input('Number of Major Vessels')
    thal = st.text_input('A Blood Disorder Called Thalassemia Value')

    prediction = ''

    if st.button('Heart disease Prediction'):
        prediction = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    st.success(prediction)

if __name__ == '__main__':
    main()
