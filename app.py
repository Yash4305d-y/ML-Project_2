import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="Heart Health AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI cards, gradient buttons, and metrics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #E63946 0%, #D62828 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #D62828 0%, #B71C1C 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. Model & Scaler Artifact Loader
# ==========================================
@st.cache_resource
def load_artifacts():
    """Loads the KNN model, Scaler, and Feature Columns safely."""
    # List of possible paths to check automatically (models/ folder OR main root folder)
    possible_paths = [
        # Check inside models/ folder first
        (
            "models/knn_heart_model.pkl",
            "models/scaler_heart.pkl",
            "models/columsn.pkl",
        ),
        # Check in main root directory if not in models/ folder
        ("knn_heart_model.pkl", "scaler_heart.pkl", "columsn.pkl"),
        # Check alternative common naming just in case
        ("models/knn_model.pkl", "models/scaler.pkl", "models/columsn.pkl"),
        ("knn_model.pkl", "scaler.pkl", "columsn.pkl"),
    ]

    for model_path, scaler_path, col_path in possible_paths:
        try:
            loaded_model = joblib.load(model_path)
            loaded_scaler = joblib.load(scaler_path)
            loaded_columns = joblib.load(col_path)
            return loaded_model, loaded_scaler, loaded_columns
        except FileNotFoundError:
            continue

    # If none of the paths work, display a clear error in the UI
    st.error(
        "🚨 Could not find the model files! Please check that `knn_heart_model.pkl`, `scaler_heart.pkl`, and `columsn.pkl` are inside your `models/` folder."
    )
    return None, None, None


# Explicitly define variables by calling the loader function
model, scaler, feature_columns = load_artifacts()

# ==========================================
# 3. Sidebar Information
# ==========================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100
    )
    st.title("Model Details")
    st.markdown(
        """
    **Algorithm:** K-Nearest Neighbors (KNN)  
    **Evaluation:** Selected for highest accuracy  
    **Inputs:** 11 Clinical Parameters  
    """
    )
    st.divider()
    st.info(
        "ℹ️ **Disclaimer:** This tool is intended for educational and predictive demonstration purposes only. Always consult a medical professional for clinical evaluations."
    )

# ==========================================
# 4. Main Interface Form
# ==========================================
st.markdown(
    "<h1 class='main-header'>❤️ Heart Disease Risk Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-header'>Fill in the clinical parameters below to evaluate the patient's likelihood of heart disease using our top-performing KNN Model.</p>",
    unsafe_allow_html=True,
)

if model is not None and scaler is not None:
    with st.form("patient_data_form"):
        col1, col2 = st.columns(2, gap="large")

        # Column 1: Demographics & Vitals
        with col1:
            st.subheader("👤 Patient Demographics & Vitals")

            age = st.slider("Age (years)", 18, 100, 50)
            sex = st.selectbox("Sex", options=["Male", "Female"])

            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=220,
                value=120,
            )
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=80,
                max_value=600,
                value=200,
            )
            max_hr = st.slider(
                "Maximum Heart Rate Achieved (bpm)", 60, 220, 150
            )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=["No (≤ 120 mg/dl)", "Yes (> 120 mg/dl)"],
            )

        # Column 2: Clinical & Electrocardiogram Features
        with col2:
            st.subheader("🩺 Clinical Features & ECG Findings")

            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[
                    "ASY: Asymptomatic",
                    "ATA: Atypical Angina",
                    "NAP: Non-Anginal Pain",
                    "TA: Typical Angina",
                ],
            )

            resting_ecg = st.selectbox(
                "Resting ECG Results",
                options=[
                    "Normal",
                    "ST: Normal ST-T Wave Abnormality",
                    "LVH: Left Ventricular Hypertrophy",
                ],
            )

            exercise_angina = st.selectbox(
                "Exercise-Induced Angina", options=["No", "Yes"]
            )

            oldpeak = st.number_input(
                "Oldpeak (ST depression induced by exercise)",
                min_value=-2.0,
                max_value=7.0,
                value=0.0,
                step=0.1,
            )

            st_slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=["Up: Upsloping", "Flat: Flat", "Down: Downsloping"],
            )

        st.divider()
        submit_btn = st.form_submit_button(
            "🔍 Predict Heart Disease Risk", use_container_width=True
        )

    # ==========================================
    # 5. Data Mapping & Inference Logic
    # ==========================================
    if submit_btn:
        # Preprocessing categorical variables into One-Hot Encoded features matching `columsn.pkl`
        sex_m = 1 if sex == "Male" else 0
        fasting_bs_val = 1 if "Yes" in fasting_bs else 0
        exercise_angina_y = 1 if exercise_angina == "Yes" else 0

        # Chest Pain Encoding
        cp_ata = 1 if "ATA" in chest_pain_type else 0
        cp_nap = 1 if "NAP" in chest_pain_type else 0
        cp_ta = 1 if "TA" in chest_pain_type else 0

        # Resting ECG Encoding
        ecg_normal = 1 if resting_ecg == "Normal" else 0
        ecg_st = 1 if "ST:" in resting_ecg else 0

        # ST Slope Encoding
        slope_flat = 1 if "Flat" in st_slope else 0
        slope_up = 1 if "Up" in st_slope else 0

        # Construct DataFrame with exact column sequence
        input_dict = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs_val,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_M": sex_m,
            "ChestPainType_ATA": cp_ata,
            "ChestPainType_NAP": cp_nap,
            "ChestPainType_TA": cp_ta,
            "RestingECG_Normal": ecg_normal,
            "RestingECG_ST": ecg_st,
            "ExerciseAngina_Y": exercise_angina_y,
            "ST_Slope_Flat": slope_flat,
            "ST_Slope_Up": slope_up,
        }

        input_df = pd.DataFrame([input_dict])

        # Ensure correct column ordering based on saved feature column list
        if feature_columns is not None:
            input_df = input_df[feature_columns]

        # Standardize features using the saved scaler
        scaled_inputs = scaler.transform(input_df)

        # Make predictions
        prediction = model.predict(scaled_inputs)[0]
        prediction_proba = model.predict_proba(scaled_inputs)[0]

        # Output Visualisation
        st.markdown("### 📊 Assessment Analysis Results")

        res_col1, res_col2 = st.columns([1, 1], gap="medium")

        with res_col1:
            if prediction == 1:
                st.error("🚨 **High Risk Detected**")
                st.markdown(
                    f"""
                    The model predicts a **High Probability of Heart Disease**.  
                    * **Confidence Score:** `{prediction_proba[1]*100:.1f}%`
                    """
                )
            else:
                st.success("✅ **Low Risk Detected**")
                st.markdown(
                    f"""
                    The model predicts a **Low Likelihood of Heart Disease**.  
                    * **Confidence Score:** `{prediction_proba[0]*100:.1f}%`
                    """
                )

        with res_col2:
            st.metric(
                label="Heart Disease Risk Factor Probability",
                value=f"{prediction_proba[1]*100:.1f}%",
                delta="Elevated" if prediction == 1 else "Normal",
                delta_color="inverse" if prediction == 1 else "normal",
            )
            st.progress(float(prediction_proba[1]))