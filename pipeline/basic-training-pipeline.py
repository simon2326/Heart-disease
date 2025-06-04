"""
train_heart_disease_model.py

Author: Simon Correa Marin
GitHub: https://github.com/simon2326/Heart-disease

Description:
This script implements a full training pipeline for the Heart Disease classification model.
It includes steps for data loading, type correction, preprocessing, model training with
hyperparameter tuning, evaluation, and conditional saving based on a recall threshold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ------------------------------------------
# 1. CONSTANTS AND GLOBAL SETTINGS
# ------------------------------------------

# URL for downloading the cleaned Heart Disease CSV
URL_DATA = (
    "https://github.com/JoseRZapata/"
    "Data_analysis_notebooks/raw/refs/heads/main/data/datasets/corazon_data.csv"
)

# Columns to load from the dataset
COLUMNS_TO_USE = [
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
    "disease",
]

# Feature groups as defined in the notebooks
CATEGORICAL_COLS = ["chest_pain", "slope", "ca", "rest_ecg", "thal", "sex"]
DISCRETE_NUMERIC = ["age", "max_hr", "chol", "rest_bp", "fbs", "exang"]
CONTINUOUS_NUMERIC = ["old_peak"]

# Sub-groups of categoricals
NOMINAL_CATEGORICAL = ["chest_pain", "rest_ecg", "thal", "sex"]
ORDINAL_CATEGORICAL = ["ca", "slope"]

# Target column and baseline recall threshold
TARGET_COL = "disease"
BASELINE_SCORE = 0.70

# ------------------------------------------
# 2. DATA LOADING & TYPE CORRECTION
# ------------------------------------------


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw heart disease CSV from the remote URL,
    selecting only the specified columns.
    """
    df = pd.read_csv(URL_DATA, usecols=COLUMNS_TO_USE, low_memory=False, na_values="?")
    return df


def correct_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type corrections identical to DataExploration.ipynb:
    1. Clean numeric-like strings in nominal category columns (excluding 'ca' and 'slope').
    2. Cast categorical columns to pandas 'category'.
    3. Convert numeric and boolean-like columns to float via to_numeric.
    4. Convert target column to integer (0/1).
    """
    for col in CATEGORICAL_COLS:
        if col not in ["ca", "slope"]:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, str) and not x.isnumeric() else np.nan
            )
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
    for col in DISCRETE_NUMERIC + CONTINUOUS_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = df[TARGET_COL].astype("bool").astype("int")
    return df


# ------------------------------------------
# 3. BUILD PREPROCESSING PIPELINE
# ------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a ColumnTransformer with separate pipelines for:
      - Numeric features (median imputation)
      - Nominal categorical features (most frequent imputation + one-hot encoding)
      - Ordinal categorical features (most frequent imputation + ordinal encoding)
    Note: boolean-like features are included in numeric.
    """
    numeric_features = DISCRETE_NUMERIC + CONTINUOUS_NUMERIC
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("nominal_cat", nominal_pipe, NOMINAL_CATEGORICAL),
            ("ordinal_cat", ordinal_pipe, ORDINAL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ------------------------------------------
# 4. MODEL PIPELINE & TUNING
# ------------------------------------------


def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Integrate the preprocessor with a RandomForestClassifier into a single pipeline.
    """
    rf_clf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_clf),
        ]
    )
    return pipeline


def train_and_select_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """
    Perform GridSearchCV over the RandomForestClassifier in the pipeline.
    Returns the best estimator (pipeline with best hyperparameters).
    """
    param_grid = {
        "model__max_depth": [4, 5, 7, 9, 10],
        "model__max_features": [2, 3, 4, 5, 6, 7, 8, 9],
        "model__criterion": ["gini", "entropy"],
    }
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    print("\n=== Best Hyperparameters ===")
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_and_save(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the best pipeline on the test set using recall_score.
    If recall exceeds BASELINE_SCORE, save the model as a .joblib file under 'models/'.
    """
    y_pred = best_pipeline.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"\nEvaluation recall on test set: {recall:.4f}")

    if recall > BASELINE_SCORE:
        print(f"Model passed baseline (recall {recall:.4f} > {BASELINE_SCORE})")
        output_dir = Path(__file__).resolve().parent / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "heart_disease_rf_model.joblib"
        dump(best_pipeline, model_path, protocol=5)
        print(f"Model saved to: {model_path}")
    else:
        msg = f"Model did not pass baseline: recall {recall:.4f} ≤ {BASELINE_SCORE}. Aborting save."
        print(msg)
        raise ValueError(msg)


# ------------------------------------------
# 5. MAIN SCRIPT EXECUTION
# ------------------------------------------


def main() -> None:
    # Step 1: Load and correct types
    print("1. Loading raw data...")
    df = load_raw_data()

    print("2. Applying type corrections...")
    df = correct_types(df)

    # Step 2: Split features and target
    print("3. Splitting features and target...")
    X = df.drop(TARGET_COL, axis="columns")
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Step 3: Build preprocessor and full pipeline
    print("4. Building preprocessing pipeline...")
    preprocessor = build_preprocessor()

    print("5. Building full model pipeline...")
    model_pipeline = build_model_pipeline(preprocessor)

    # Step 4: Train and hyperparameter selection
    print("6. Training and hyperparameter selection (GridSearchCV)...")
    best_pipeline = train_and_select_model(model_pipeline, X_train, y_train)

    # Step 5: Evaluate and conditional save
    print("7. Evaluating and possibly saving the final model...")
    evaluate_and_save(best_pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
