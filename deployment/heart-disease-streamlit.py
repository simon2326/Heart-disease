"""
heart_disease_streamlit_lr.py

Author: Simon Correa Marin
GitHub: https://github.com/simon2326/Heart-disease

A Streamlit app that lets users input patient data in a
medical‐style layout (on the LEFT), and then immediately
displays a “Medical History Summary” (on the RIGHT) showing
exactly which choices were made. The banner image is centered
at roughly half the page width, and the form is organized by
“Demographics,” “Vital Signs,” and “Clinical Measurements.”
"""

import os
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

# ------------------------------------------
# 1. PAGE CONFIG & CSS
# ------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Predictor (LR) ❤️",
    page_icon="💖",
    layout="wide"
)

# Inject CSS for “medical‐style” section boxes and “alert” cards
st.markdown(
    """
    <style>
    /* Banner container spacing */
    .banner-container {
        margin-bottom: 2rem;
    }
    /* Medical‐style section background */
    .section-box {
        background-color: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #dde2e6;
    }
    /* High‐risk & low‐risk cards */
    .risk-high {
        background-color: #ffe5e5;  /* light red */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ff4b4b;  /* strong red border */
        margin-top: 1rem;
    }
    .risk-low {
        background-color: #e5ffe5;  /* light green */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #4bff4b;  /* strong green border */
        margin-top: 1rem;
    }
    /* Form headings */
    .section-heading {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2b2e33;
    }
    /* Summary labels */
    .summary-label {
        font-weight: 600;
        color: #2b2e33;
    }
    /* Medical‐style section background with reduced height and a new color */
    .section-box {
        background-color: #e0f7fa;  /* light pastel blue */
        padding: 0.4rem 1rem;       /* reduce vertical padding to make box shorter */
        border-radius: 0.5rem;
        margin-bottom: 1rem;       /* slightly smaller bottom margin */
        border: 1px solid #b2ebf2; /* a complementary blue border */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# 2. LOAD MODEL (FROM ../pipeline/models/)
# ------------------------------------------

@st.cache_resource
def load_model(model_path: str) -> Pipeline:
    """
    Load the trained LogisticRegression pipeline from disk.
    """
    with st.spinner("Loading model..."):
        return load(model_path)

# ------------------------------------------
# 3. MAIN APP LAYOUT & PREDICTION
# ------------------------------------------

def main() -> None:
    """
    Main entry point for the Streamlit app.
    """
    # 3.1 Banner image (centered at half page width)
    this_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(this_file)
    banner_path = os.path.join(project_dir, "heart.jpg")

    if os.path.exists(banner_path):
        cola, colb, colc = st.columns([1, 2, 1])
        with colb:
            st.image(banner_path, use_container_width=True)
    else:
        st.warning("🚨 Could not find 'heart.jpg' in deployment/ folder.")

    st.markdown("## 🚑 Patient Heart Risk Assessment")
    st.markdown(
        "This application uses a trained **Logistic Regression** model "
        "to estimate a patient’s likelihood of having heart disease."
    )
    st.markdown("---")

    # 3.2 Create two columns: LEFT for the form, RIGHT for the summary/prediction
    left_col, right_col = st.columns([1, 1])

    # -------------------------
    # 3.3 LEFT COLUMN: FORM
    # -------------------------
    with left_col:
        st.markdown('<div class="section-heading">📝 Patient Medical Data</div>', unsafe_allow_html=True)

        # 3.3.1 Demographics Section
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">👤 Demographics</div>', unsafe_allow_html=True)
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            age = st.number_input(
                "Age (years):",
                min_value=1, max_value=120, value=50, step=1,
                help="Patient's age in years"
            )
            sex = st.selectbox(
                "Sex:",
                options=[1, 0],
                format_func=lambda x: "Male" if x == 1 else "Female",
                help="1 = Male, 0 = Female"
            )
        with demo_col2:
            rest_bp = st.number_input(
                "Resting Blood Pressure (mm Hg):",
                min_value=50, max_value=300, value=120, step=1,
                help="Patient's resting blood pressure"
            )
            chol = st.number_input(
                "Serum Cholesterol (mg/dl):",
                min_value=100, max_value=600, value=200, step=1,
                help="Patient's serum cholesterol level"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 3.3.2 Vital Signs Section
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">❤️ Vital Signs</div>', unsafe_allow_html=True)
        vital_col1, vital_col2 = st.columns(2)
        with vital_col1:
            max_hr = st.number_input(
                "Max Heart Rate Achieved:",
                min_value=50, max_value=250, value=150, step=1,
                help="Peak heart rate achieved during exercise"
            )
            old_peak = st.slider(
                "ST Depression (old_peak):",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="ST depression induced by exercise relative to rest"
            )
        with vital_col2:
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl:",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="1 = True (FBS > 120 mg/dl), 0 = False"
            )
            exang = st.selectbox(
                "Exercise Induced Angina:",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="1 = Yes, 0 = No"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 3.3.3 Clinical Measurements Section
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
        clin_col1, clin_col2 = st.columns(2)
        with clin_col1:
            chest_pain = st.selectbox(
                "Chest Pain Type:",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical Angina",
                    2: "Atypical Angina",
                    3: "Non-anginal Pain",
                    4: "Asymptomatic"
                }[x],
                help="Type of chest pain"
            )
            rest_ecg = st.selectbox(
                "Resting ECG Results:",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Normal",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x],
                help="Resting ECG results"
            )
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment:",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Upsloping",
                    2: "Flat",
                    3: "Downsloping"
                }[x],
                help="Slope of ST segment peak"
            )
        with clin_col2:
            ca = st.selectbox(
                "Number of Major Vessels (0–3) Colored by Fluoroscopy:",
                options=[0, 1, 2, 3],
                help="Number of major vessels colored by fluoroscopy"
            )
            thal = st.selectbox(
                "Thalassemia Type:",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "Normal",
                    6: "Fixed Defect",
                    7: "Reversible Defect"
                }[x],
                help="Thalassemia classification"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # 3.4 RIGHT COLUMN: SUMMARY & PREDICTION
    # -------------------------
    with right_col:
        st.markdown('<div class="section-heading">📝 Medical History Summary</div>', unsafe_allow_html=True)

        # Build a small summary table of the inputs
        summary_html = f"""
        <div class="section-box">
          <div style="display: flex; justify-content: space-between;">
            <div><span class="summary-label">Age:</span> {age} years</div>
            <div><span class="summary-label">Sex:</span> {"Male" if sex == 1 else "Female"}</div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <div><span class="summary-label">Rest BP:</span> {rest_bp} mm Hg</div>
            <div><span class="summary-label">Cholesterol:</span> {chol} mg/dl</div>
          </div>
        </div>

        <div class="section-box">
          <div style="display: flex; justify-content: space-between;">
            <div><span class="summary-label">Max HR Achieved:</span> {max_hr} bpm</div>
            <div><span class="summary-label">ST Depression:</span> {old_peak:.1f}</div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <div><span class="summary-label">FBS >120:</span> {"Yes" if fbs == 1 else "No"}</div>
            <div><span class="summary-label">Exercise Angina:</span> {"Yes" if exang == 1 else "No"}</div>
          </div>
        </div>

        <div class="section-box">
          <div style="display: flex; justify-content: space-between;">
            <div><span class="summary-label">Chest Pain Type:</span>
              {"Typical Angina" if chest_pain==1 else
               "Atypical Angina" if chest_pain==2 else
               "Non-anginal Pain" if chest_pain==3 else
               "Asymptomatic"}
            </div>
            <div><span class="summary-label">Resting ECG:</span>
              {"Normal" if rest_ecg==0 else 
               "ST-T Wave Abnormality" if rest_ecg==1 else
               "Left Ventricular Hypertrophy"}
            </div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <div><span class="summary-label">Slope ST Segment:</span>
              {"Upsloping" if slope==1 else "Flat" if slope==2 else "Downsloping"}
            </div>
            <div><span class="summary-label">Major Vessels (0–3):</span> {ca}</div>
          </div>
          <div style="margin-top: 0.5rem;">
            <span class="summary-label">Thalassemia Type:</span>
            {"Normal" if thal==3 else "Fixed Defect" if thal==6 else "Reversible Defect"}
          </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

        # 3.4.1 Load and run the model
        model_path = os.path.join(
            project_dir, "..", "pipeline", "models", "heart_disease_lr_model.joblib"
        )
        model = load_model(model_path)

        # Build a small single‐row DataFrame exactly as training expects
        input_df = pd.DataFrame(
            {
                "age":        [age],
                "sex":        [sex],
                "chest_pain": [chest_pain],
                "rest_bp":    [rest_bp],
                "chol":       [chol],
                "fbs":        [fbs],
                "rest_ecg":   [rest_ecg],
                "max_hr":     [max_hr],
                "exang":      [exang],
                "old_peak":   [old_peak],
                "slope":      [slope],
                "ca":         [ca],
                "thal":       [thal],
            }
        )

        pred = model.predict(input_df)[0]
        proba_arr = model.predict_proba(input_df)[0]
        proba = proba_arr[1] if proba_arr.shape[0] == 2 else (1.0 if model.classes_[0] == 1 else 0.0)

        # 3.4.2 Display the result in a styled “medical alert” card
        if pred == 1:
            st.markdown(
                f"""
                <div class="risk-high">
                  <h3>⚠️ HIGH RISK of Heart Disease (Probability: {proba:.2f})</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="risk-low">
                  <h3>✅ LOW RISK of Heart Disease (Probability: {proba:.2f})</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption("Model: LogisticRegression over numeric‐encoded features")

if __name__ == "__main__":
    main()