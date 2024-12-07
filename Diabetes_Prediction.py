import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import base64
import tempfile

# Load the trained model and scaler
with open('Diabetes_Prediction_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the preprocessed dataset to fit the scaler
data = pd.read_csv('preprocessed_diabetes_data.csv')

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on numerical features
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler.fit(data[numerical_features])

# Streamlit application for Diabetes Prediction
def main():
    st.title("Diabetes Prediction App")

    # User inputs
    name = st.text_input("Enter Name")
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.selectbox("Heart Disease", ("No", "Yes"))
    smoking_history = st.selectbox("Smoking History", ("never", "current", "formerly", "No Info", "ever", "not current"))
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=80, max_value=300, value=140, step=1)

    # Convert categorical inputs to numeric
    gender_numeric = 1 if gender == "Male" else 0
    hypertension_numeric = 1 if hypertension == "Yes" else 0
    heart_disease_numeric = 1 if heart_disease == "Yes" else 0
    smoking_history_numeric = {
        "never": 0,
        "current": 1,
        "formerly": 2,
        "No Info": 3,
        "ever": 4,
        "not current": 5
    }[smoking_history]

    # Add other 12 dummy features or predefined features as needed
    # These could be either statistical features or constants
    # Here, we assume 12 additional features like cholesterol, insulin, etc.
    # You need to adapt this part based on the actual dataset.

    extra_features = np.random.rand(12)  # This is just a placeholder for the other 12 features

    # Create feature vector with 8 user inputs + 12 additional features
    inputs = np.array([[age, bmi, HbA1c_level, blood_glucose_level]])
    scaled_inputs = scaler.transform(inputs)
    feature_vector = np.concatenate((
        [gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_history_numeric], 
        scaled_inputs.flatten(), 
        extra_features  # Add additional dummy features here
    )).reshape(1, -1)

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(feature_vector)
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            name_prefix = "Mr." if gender == "Male" else "Ms." if gender == "Female" else ""

            # Display the medical report
            st.markdown(f"### Medical Report for {name_prefix} {name}")
            st.markdown(
                f"""
                **Patient Name:** {name_prefix} {name}  
                **Gender:** {gender}  
                **Age:** {age}  
                **Hypertension:** {hypertension}  
                **Heart Disease:** {heart_disease}  
                **Smoking History:** {smoking_history}  
                **BMI:** {bmi}  
                **HbA1c Level:** {HbA1c_level}  
                **Blood Glucose Level:** {blood_glucose_level}  
                """, unsafe_allow_html=True
            )

            # Highlight the prediction result
            st.markdown(f"### <span style='color:maroon;'>Prediction: {result}</span>", unsafe_allow_html=True)
            
            # Print the personalized message
            if result == "Diabetic":
                message = "Take Care of your Health, Have a NICE Day"
            else:
                message = "Congrats! You seem to be Healthy, Have a NICE Day"

            st.markdown(f"#### {message}", unsafe_allow_html=True)

            # Generate and display PDF
            pdf_buffer = generate_pdf(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
            st.download_button(label="Download PDF", data=pdf_buffer, file_name='Medical_Report.pdf', mime='application/pdf')

            # Generate and display Image
            img_buffer = generate_image(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
            st.download_button(label="Download Image", data=img_buffer, file_name='Medical_Report.png', mime='image/png')

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# PDF and image generation functions...

if __name__ == "__main__":
    main()
