"""
heart_disease_batch.py

Author: Simon Correa Marin
GitHub: https://github.com/simon2326/Heart-disease

Defines `batch_tab()`, which:
  - injects minimal UI
  - lets user upload a CSV with string‐encoded categorical columns
  - maps those columns to numeric codes exactly as in training
  - drops any rows with missing values, runs model.predict, and shows/downloads results
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

@st.cache_resource
def load_model(model_path: str) -> Pipeline:
    """Load the trained LogisticRegression pipeline from disk."""
    with st.spinner("Loading trained model…"):
        return load(model_path)

# Same mappings used during training:
SEX_MAP = {"male": 1, "female": 0}
CHEST_MAP = {
    "typical":      1,
    "nontypical":   2,
    "nonanginal":   3,
    "asymptomatic": 4,
}
REST_ECG_MAP = {
    "normal":                         0,
    "st-t wave abnormality":          1,
    "left ventricular hypertrophy":   2,
}
THAL_MAP = {
    "normal":     3,
    "fixed":      6,
    "reversable": 7,
}

def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Map 'sex','chest_pain','rest_ecg','thal' from string→numeric
    2) Coerce numeric columns
    3) Drop rows missing any of the 13 features
    """
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Uploaded data is not a DataFrame.")

    required_cols = [
        "age",       # numeric
        "sex",       # string: male/female
        "chest_pain",# string: typical/nontypical/nonanginal/asymptomatic
        "rest_bp",   # numeric
        "chol",      # numeric
        "fbs",       # numeric or “0/1”
        "rest_ecg",  # string: normal/st-t wave abnormality/left ventricular hypertrophy
        "max_hr",    # numeric
        "exang",     # numeric or “0/1”
        "old_peak",  # numeric
        "slope",     # numeric
        "ca",        # numeric
        "thal",      # string: normal/fixed/reversable
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Map strings → numeric
    df["sex"] = df["sex"].astype(str).str.strip().str.lower().map(SEX_MAP)
    df["chest_pain"] = df["chest_pain"].astype(str).str.strip().str.lower().map(CHEST_MAP)
    df["rest_ecg"] = df["rest_ecg"].astype(str).str.strip().str.lower().map(REST_ECG_MAP)
    df["thal"] = df["thal"].astype(str).str.strip().str.lower().map(THAL_MAP)

    # Convert boolean‐like and numeric columns
    df["fbs"] = pd.to_numeric(df["fbs"], errors="coerce")
    df["exang"] = pd.to_numeric(df["exang"], errors="coerce")

    for col in ["age", "rest_bp", "chol", "max_hr", "old_peak", "slope", "ca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any row with NaN in required columns
    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
    return df_clean

def batch_tab() -> None:
    """
    Renders the “Batch Prediction” tab:
    - CSV uploader
    - preview of raw DataFrame
    - preprocessing → numeric
    - model.predict → show & download results
    """
    st.subheader("📤 Upload your CSV file (string‐encoded)")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help=(
            "CSV must contain exactly these columns:\n"
            "- age, sex, chest_pain, rest_bp, chol, fbs, rest_ecg,\n"
            "- max_hr, exang, old_peak, slope, ca, thal"
        ),
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV. Details:\n{e}")
            return

        st.write("🔍 Preview of uploaded data (first 5 rows):")
        st.dataframe(df.head())

        # Preprocess
        try:
            df_clean = preprocess_batch_data(df)
        except ValueError as ve:
            st.warning(f"⚠️ {ve}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error during preprocessing:\n{e}")
            return

        if st.button("Predict Disease Risk (Batch)"):
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "pipeline",
                "models",
                "heart_disease_lr_model.joblib",
            )
            model = load_model(model_path)

            preds = model.predict(df_clean)
            try:
                proba = model.predict_proba(df_clean)[:, 1]
            except Exception:
                single_class = model.classes_[0]
                proba = [1.0 if single_class == 1 else 0.0] * len(df_clean)

            df_result = df_clean.copy()
            df_result["Predicted Label"] = preds.astype(int)
            df_result["Predicted Probability"] = proba

            st.success("✅ Batch predictions completed!")
            st.dataframe(df_result)

            csv_out = df_result.to_csv(index=False)
            st.download_button(
                "💾 Download results as CSV",
                data=csv_out,
                file_name="heart_disease_batch_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("🔎 Please upload a CSV file with the required 13 columns.")
        st.caption("Sample format (first 5 of 21 rows):")
        sample_path = Path(__file__).parent / "heart_disease_batch_sample.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path).head(5)
            st.dataframe(sample_df)
        else:
            st.write("No sample CSV found in deployment/ folder.")