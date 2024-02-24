import os
import numpy as np
import pickle
import streamlit as st

# Get the directory of the script
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'trained_model.sav')

# Check if the file exists before attempting to open
if os.path.exists(file_path):
    try:
        loaded_model = pickle.load(open(file_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error(f"Error: File '{file_path}' not found.")

def heart_disease_prediction(input):
    # Convert empty strings to NaN (Not a Number)
    input = [np.nan if value == '' else value for value in input]

    try:
        in_np = np.asarray(input, dtype=float)
    except ValueError as ve:
        st.error(f"Error converting input to NumPy array: {ve}")
        return

    np_reshape = in_np.reshape(1, -1)
    prediction = loaded_model.predict(np_reshape)

    print("Prediction:", prediction)

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
        input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction = heart_disease_prediction(input_values)

    st.success(prediction)

if __name__ == '__main__':
    main()
