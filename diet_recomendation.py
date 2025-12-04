import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os

# Page Configuration
st.set_page_config(
    page_title="Diet Recommendation System",
    page_icon="🥗",
    layout="wide"
)

# --- 1. FUNCIÓN PARA CARGAR EL MODELO ---
@st.cache_resource
def load_model():
    # Verifica que el archivo exista en la misma carpeta
    file_path = 'diet_recommendation_model.pkl'
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Cargamos el modelo al iniciar la app
model = load_model()

# Title and Intro
st.title("🥗 Diet Recommendation System")
st.write("Welcome! Please enter your bio-metrics and preferences below to get a personalized diet plan.")

# Alerta si no encuentra el modelo
if model is None:
    st.error("⚠️ Error: The file 'diet_recommendation_model.pkl' was not found. Please make sure it is in the same folder as this script.")

# --- SECTION 1: Personal Information ---
st.header("1. Personal Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", options=["Male", "Female"])

with col2:
    height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)

with col3:
    # Auto-calculate BMI
    bmi_score = weight_kg / ((height_cm / 100) ** 2)
    st.metric(label="Calculated BMI", value=f"{bmi_score:.2f}")
    
    physical_activity = st.selectbox(
        "Physical Activity Level", 
        options=["Sedentary", "Moderate", "Active"]
    )

# --- SECTION 2: Medical & Health Metrics ---
st.header("2. Health Metrics")
col4, col5, col6 = st.columns(3)

with col4:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=90)
    blood_pressure = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)

with col5:
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
    severity = st.selectbox("Condition Severity", options=["Mid", "Moderate", "Severe"])

with col6:
    exercise_hours = st.number_input("Weekly Exercise Hours", 0.0, 20.0, 3.5)
    adherence_chance = st.slider("Adherence to Diet Plan (%)", 0, 100, 80)

# --- NUEVO CAMPO NECESARIO POR EL ERROR ---
st.write("---")
col_extra, _ = st.columns(2)
with col_extra:
    # El modelo pide explícitamente disease_Hypertension, Obesity, Unknown
    # Asumimos que "None" o "Healthy" mapea a Unknown o simplemente ceros en las otras.
    # Para asegurar compatibilidad con "Unknown", lo incluimos.
    disease_status = st.selectbox(
        "Existing Medical Condition (Disease)", 
        options=["None/Healthy", "Hypertension", "Obesity", "Unknown"]
    )

# --- SECTION 3: Preferences & Restrictions ---
st.header("3. Diet Preferences")

col7, col8 = st.columns(2)

with col7:
    cuisine = st.selectbox("Preferred Cuisine", ["Indian", "Italian", "Mexican"])
    allergy_type = st.selectbox("Allergies", ["None", "Peanuts", "Gluten"])

with col8:
    dietary_restrictions = st.selectbox("Dietary Restrictions", ["None", "Low_Sugar", "Low_Sodium"])
    imbalance_score = st.slider("Dietary Nutrient Imbalance Score", 0.0, 1.0, 0.2)

daily_calories = st.number_input("Daily Caloric Intake (Current)", 1000, 5000, 2000)


# --- PROCESSING DATA FOR MODEL ---
if st.button("Generate Recommendation"):
    if model is None:
        st.error("Cannot generate recommendation without the model file.")
    else:
        try:
            # --- 2. MAPEO DE VARIABLES (Transformar Texto a Números) ---
            
            severity_map = {"Mid": 0, "Moderate": 1, "Severe": 2}
            activity_map = {"Active": 0, "Moderate": 1, "Sedentary": 2} 
            diet_restrict_map = {"Low_Sodium": 0, "Low_Sugar": 1, "None": 2} 
            
            # Codificación One-Hot
            is_male = 1 if gender == "Male" else 0
            
            is_indian = 1 if cuisine == "Indian" else 0
            is_italian = 1 if cuisine == "Italian" else 0
            is_mexican = 1 if cuisine == "Mexican" else 0
            
            is_no_allergy = 1 if allergy_type == "None" else 0
            is_peanuts = 1 if allergy_type == "Peanuts" else 0
            
            # --- NUEVA LÓGICA PARA DISEASE ---
            # El modelo espera: disease_Hypertension, disease_Obesity, disease_Unknown
            is_hypertension = 1 if disease_status == "Hypertension" else 0
            is_obesity = 1 if disease_status == "Obesity" else 0
            is_unknown = 1 if disease_status == "Unknown" else 0 # O "None/Healthy" si así se entrenó, pero usaremos Unknown para ser seguros
            
            # --- 3. CREAR EL DATAFRAME DE ENTRADA ---
            # El orden aquí es CRÍTICO. Debe ser IDÉNTICO al del entrenamiento.
            # Basado en tu error: ...cuisine_Mexican, disease_Hypertension, disease_Obesity, disease_Unknown, age...
            
            input_data = {
                'severity': severity_map[severity],
                'physical_activity_level': activity_map[physical_activity],
                'dietary_restrictions': diet_restrict_map[dietary_restrictions],
                'gender_Male': is_male,
                'allergies_No_Allergies': is_no_allergy,
                'allergies_Peanuts': is_peanuts,
                'preferred_cuisine_Indian': is_indian,
                'preferred_cuisine_Italian': is_italian,
                'preferred_cuisine_Mexican': is_mexican,
                
                # --- AQUÍ AGREGAMOS LAS COLUMNAS FALTANTES ---
                'disease_Hypertension': is_hypertension,
                'disease_Obesity': is_obesity,
                'disease_Unknown': is_unknown,
                # ---------------------------------------------
                
                'age': age,
                'weight': weight_kg,
                'height': height_cm,
                'bmi': bmi_score,
                'daily_caloric_intake': daily_calories,
                'cholesterol': cholesterol,
                'blood_pressure': blood_pressure,
                'glucose': glucose,
                'exercise_hours': exercise_hours,
                'adherence_to_diet_plan': adherence_chance,
                'dietary_nutrient_imbalance_score': imbalance_score
            }
            
            df_input = pd.DataFrame([input_data])
            
            # --- 4. PREDICCIÓN REAL ---
            prediction_index = model.predict(df_input)[0]
            
            target_categories = ['Balanced', 'Low_Sodium', 'Low_Carb']
            final_recommendation = target_categories[int(prediction_index)]
            
            # --- 5. MOSTRAR RESULTADOS ---
            st.success("Recommendation Generated Successfully!")
            st.markdown(f"### 🥗 Recommended Diet: **{final_recommendation}**")
            
            if final_recommendation == "Balanced":
                st.info("💡 A balanced diet includes a variety of foods from all food groups: proteins, healthy fats, and carbohydrates.")
            elif final_recommendation == "Low_Sodium":
                st.info("💡 Focus on fresh foods (fruits, vegetables) and avoid processed items, canned soups, and salty snacks.")
            elif final_recommendation == "Low_Carb":
                st.info("💡 Prioritize proteins (meat, fish, eggs) and healthy fats. Limit sugar, bread, pasta, and rice.")
                
            with st.expander("Show Debug Data (Input Vector)"):
                st.dataframe(df_input)
                st.write(f"Raw Model Prediction Code: {prediction_index}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Hint: Check if the 'input_data' dictionary keys match exactly the column names used in training.")