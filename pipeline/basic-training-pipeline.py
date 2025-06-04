"""
basic-training-pipeline.py

Author: Simon Correa Marin
GitHub: https://github.com/simon2326/Heart-disease

A training pipeline for Heart Disease classification using LogisticRegression.
Reads a preprocessed data, applies numeric encoding,
performs 5-fold stratified cross-validation (CV), and if the recall exceeds 0.70 on the test set,
saves the trained pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score

# ------------------------------------------
# 1. CONSTANTS & GLOBAL SETTINGS
# ------------------------------------------

# Determine locations relative to this script
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
LOCAL_CSV = PROJECT_ROOT / "data" / "02_intermediate" / "hd_type_fixed.csv"

TARGET_COL = "disease"
BASELINE_RECALL = 0.70

# ------------------------------------------
# 2. NUMERIC ENCODING MAPS (match hd_type_fixed.csv)
# ------------------------------------------

SEX_MAP = {"male": 1, "female": 0}

CHEST_MAP = {
    "typical":      1,  # "typical" → 1
    "nontypical":   2,  # "nontypical" → 2
    "nonanginal":   3,  # "nonanginal" → 3
    "asymptomatic": 4,  # "asymptomatic" → 4
}

REST_ECG_MAP = {
    "normal":                       0,
    "st-t wave abnormality":        1,
    "left ventricular hypertrophy": 2,
}

# CSV uses "reversable" spelling
THAL_MAP = {
    "normal":     3,
    "fixed":      6,
    "reversable": 7,
}

# ------------------------------------------
# 3. LOAD & MAP THE LOCAL CSV
# ------------------------------------------

def load_and_map_numeric() -> pd.DataFrame:
    """
    1. Read hd_type_fixed.csv from disk.
    2. Map categorical columns to integer codes using predefined maps.
    3. Convert numeric-like columns to float, coercing invalid values to NaN.
    4. Drop any row that contains NaN in any required column.
    """
    df = pd.read_csv(LOCAL_CSV)

    # Map string columns to numeric codes
    df["sex"] = df["sex"].astype(str).str.strip().str.lower().map(SEX_MAP)
    df["chest_pain"] = df["chest_pain"].astype(str).str.strip().str.lower().map(CHEST_MAP)
    df["rest_ecg"] = df["rest_ecg"].astype(str).str.strip().str.lower().map(REST_ECG_MAP)
    df["thal"] = df["thal"].astype(str).str.strip().str.lower().map(THAL_MAP)

    # Convert boolean-like columns to numeric (coerce errors → NaN) and cast to Int64
    df["fbs"] = pd.to_numeric(df["fbs"], errors="coerce").astype("Int64")
    df["exang"] = pd.to_numeric(df["exang"], errors="coerce").astype("Int64")

    # Convert remaining numeric columns (coerce invalid → NaN)
    for col in ["age", "rest_bp", "chol", "max_hr", "old_peak", "slope", "ca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert target to 0/1
    df["disease"] = df["disease"].astype("bool").astype("int")

    # Drop rows with any NaN in required columns
    df = df.dropna(
        subset=[
            "sex",
            "chest_pain",
            "rest_ecg",
            "thal",
            "fbs",
            "exang",
            "age",
            "rest_bp",
            "chol",
            "max_hr",
            "old_peak",
            "slope",
            "ca",
            "disease",
        ]
    )

    return df

# ------------------------------------------
# 4. BUILD PREPROCESSOR & LOGISTIC REGRESSION PIPELINE
# ------------------------------------------

def build_preprocessor_numeric() -> ColumnTransformer:
    """
    Create a ColumnTransformer that applies median imputation to all 13 numeric features.
    """
    numeric_cols = [
        "age",
        "sex",
        "chest_pain",
        "rest_bp",
        "chol",
        "fbs",
        "rest_ecg",
        "max_hr",
        "exang",
        "old_peak",
        "slope",
        "ca",
        "thal",
    ]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_model_pipeline_lr(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Integrate the numeric preprocessor with a LogisticRegression classifier.
    """
    lr = LogisticRegression(
        solver="liblinear",  # suitable for smaller datasets
        random_state=42,
        max_iter=1000,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lr),
    ])
    return pipeline

# ------------------------------------------
# 5. GRID SEARCH (STRATIFIED 5-FOLD CV) FOR RECALL
# ------------------------------------------

def train_and_select_model_lr(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Perform GridSearchCV over LogisticRegression's C parameter (values [0.01,0.1,1,10,100]),
    optimizing for recall. Uses StratifiedKFold(n_splits=5) to ensure both classes
    are present in each fold.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    grid.fit(X_train, y_train)
    print("\n=== Best Hyperparameters (LogisticRegression) ===")
    print(grid.best_params_)

    return grid.best_estimator_

# ------------------------------------------
# 6. EVALUATE ON TEST SET & SAVE IF RECALL > 0.70
# ------------------------------------------

def evaluate_and_save_lr(
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Compute recall and ROC AUC on the held-out test set. If recall exceeds
    BASELINE_RECALL, save the pipeline under pipeline/models_lr/heart_disease_lr_model.joblib.
    """
    y_pred = best_pipeline.predict(X_test)
    recall_val = recall_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(
        y_test,
        best_pipeline.predict_proba(X_test)[:, 1]
    )

    print(
        f"\nEvaluation on Test Set:\n"
        f"  Recall : {recall_val:.4f}\n"
        f"  ROC AUC: {roc_auc_val:.4f}"
    )

    if recall_val > BASELINE_RECALL:
        print(f"✅ Model passed baseline (recall {recall_val:.4f} > {BASELINE_RECALL})")
        out_dir = THIS_FILE.parent / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "heart_disease_lr_model.joblib"
        dump(best_pipeline, model_path, protocol=5)
        print(f"Model saved to: {model_path}")
    else:
        msg = (
            f"❌ Model did not pass baseline: recall {recall_val:.4f} ≤ {BASELINE_RECALL}. "
            "No model saved."
        )
        print(msg)
        raise ValueError(msg)

# ------------------------------------------
# 7. MAIN ENTRYPOINT
# ------------------------------------------

def main() -> None:
    print("1. Loading and mapping data to numeric codes (from local CSV)...")
    df = load_and_map_numeric()

    # Split features/target
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    print("2. Splitting train/test (80/20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("3. Building numeric preprocessing pipeline…")
    preprocessor = build_preprocessor_numeric()

    print("4. Building LogisticRegression pipeline…")
    lr_pipeline = build_model_pipeline_lr(preprocessor)

    print("5. Training + hyperparameter selection (Stratified 5-fold CV)…")
    best_lr_pipeline = train_and_select_model_lr(lr_pipeline, X_train, y_train)

    print("6. Evaluating on test set and possibly saving model…")
    evaluate_and_save_lr(best_lr_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()