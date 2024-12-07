import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import tempfile

# Model va skalerni yuklash
try:
    with open('Diabetes_Prediction_Model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    st.error(f"Fayl topilmadi: {e}")
    st.stop()

# Streamlit ilovasi
def main():
    st.title("Diabetni Bashorat Qilish Ilovasi")
    
    # Foydalanuvchi kiritishlari
    name = st.text_input("Ismingizni kiriting")
    gender = st.selectbox("Jinsingiz", ("Erkak", "Ayol"))
    age = st.number_input("Yoshingiz", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox("Gipertenziya", ("Yo'q", "Bor"))
    heart_disease = st.selectbox("Yurak kasalligi", ("Yo'q", "Bor"))
    smoking_history = st.selectbox("Chekish tarixi", ("Chekmagan", "Chekayotgan", "Avval chekkan", "Ma'lumot yo'q", "Hozirda chekmaydi"))
    bmi = st.number_input("BMI (Tana Massasi Indeksi)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c darajasi", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("Qondagi glyukoza darajasi", min_value=80, max_value=300, value=140, step=1)
    
    # Kategoriyalarni raqamlarga aylantirish
    gender_numeric = 1 if gender == "Erkak" else 0
    hypertension_numeric = 1 if hypertension == "Bor" else 0
    heart_disease_numeric = 1 if heart_disease == "Bor" else 0
    smoking_mapping = {
        "Chekmagan": 0,
        "Chekayotgan": 1,
        "Avval chekkan": 2,
        "Ma'lumot yo'q": 3,
        "Hozirda chekmaydi": 4
    }
    smoking_history_numeric = smoking_mapping[smoking_history]
    
    # Xususiyatlar vektori
    inputs = np.array([[age, bmi, HbA1c_level, blood_glucose_level]])
    scaled_inputs = scaler.transform(inputs)
    feature_vector = np.concatenate(([gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_history_numeric], scaled_inputs.flatten())).reshape(1, -1)
    
    # Bashorat qilish
    if st.button("Bashorat qilish"):
        prediction = model.predict(feature_vector)
        result = "Diabetik" if prediction[0] == 1 else "Sog'lom"
        name_prefix = "Janob" if gender == "Erkak" else "Xonim"
        
        # Tibbiy hisobotni ko'rsatish
        st.markdown(f"### {name_prefix} {name} uchun Tibbiy Hisobot")
        st.markdown(f"""
            **Ism:** {name_prefix} {name}  
            **Jinsi:** {gender}  
            **Yoshi:** {age}  
            **Gipertenziya:** {hypertension}  
            **Yurak Kasalligi:** {heart_disease}  
            **Chekish Tarixi:** {smoking_history}  
            **BMI:** {bmi}  
            **HbA1c Darajasi:** {HbA1c_level}  
            **Qondagi Glyukoza Darajasi:** {blood_glucose_level}  
        """, unsafe_allow_html=True)
        st.markdown(f"### Bashorat: **{result}**", unsafe_allow_html=True)
        
        # PDF va rasm generatsiyasi
        pdf_buffer = generate_pdf(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
        st.download_button(label="PDF-ni yuklab olish", data=pdf_buffer, file_name='Tibbiy_Hisobot.pdf', mime='application/pdf')
        
        img_buffer = generate_image(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result)
        st.download_button(label="Rasmni yuklab olish", data=img_buffer, file_name='Tibbiy_Hisobot.png', mime='image/png')

def generate_pdf(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    name_prefix = "Janob" if gender == "Erkak" else "Xonim"
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"Tibbiy Hisobot - {name_prefix} {name}", ln=True, align='C')
    pdf.ln(10)
    report_data = [
        ("Ism", f"{name_prefix} {name}"),
        ("Jinsi", gender),
        ("Yoshi", age),
        ("Gipertenziya", hypertension),
        ("Yurak Kasalligi", heart_disease),
        ("Chekish Tarixi", smoking_history),
        ("BMI", bmi),
        ("HbA1c Darajasi", HbA1c_level),
        ("Qondagi Glyukoza Darajasi", blood_glucose_level),
        ("Bashorat", result)
    ]
    pdf.set_font("Arial", size=12)
    for key, value in report_data:
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    with open(temp_file.name, 'rb') as f:
        pdf_buffer = f.read()
    temp_file.close()
    return pdf_buffer

def generate_image(name, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, result):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    name_prefix = "Janob" if gender == "Erkak" else "Xonim"
    report_text = (
        f"Tibbiy Hisobot - {name_prefix} {name}\n\n"
        f"Ism: {name_prefix} {name}\n"
        f"Jinsi: {gender}\n"
        f"Yoshi: {age}\n"
        f"Gipertenziya: {hypertension}\n"
        f"Yurak Kasalligi: {heart_disease}\n"
        f"Chekish Tarixi: {smoking_history}\n"
        f"BMI: {bmi}\n"
        f"HbA1c Darajasi: {HbA1c_level}\n"
        f"Qondagi Glyukoza Darajasi: {blood_glucose_level}\n\n"
        f"Bashorat: {result}"
    )
    ax.text(0.5, 0.5, report_text, fontsize=10, ha='center', va='center', wrap=True)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer.getvalue()

if __name__ == "__main__":
    main()
