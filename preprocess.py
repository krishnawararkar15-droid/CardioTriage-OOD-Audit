"""
CardioTriage - Clinical Preprocessing & Out-of-Distribution Partitioning Pipeline
Veritas AI International High School Hackathon 2026 - AI + Health Track

This script implements:
1. Target transformation: Multi-class `num` (0-4) -> Binary `target` (0: No disease, 1: Disease present).
2. Physiological sanity checks: Replace unmeasured values (e.g. chol == 0, trestbps == 0) with NaN.
3. Strict Train/Test Partition:
   - Cleveland: 80% Train (N=242), 20% Held-out In-Distribution Test (N=61), stratified by target.
   - OOD Test Cohorts (Zero leakage, never seen during model training):
     * Hungary (N=294)
     * Switzerland (N=123)
     * VA Long Beach (N=200)
4. Zero-Leakage Imputation & Scaling:
   - Median imputation and standard scaling are fit EXCLUSIVELY on the Cleveland training set.
   - Applied symmetrically to Cleveland Test and all external OOD cohorts.
5. Demographic fairness slice preservation (Sex, Age groups).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 13 clinical predictors (excluding target 'num' and 'origin')
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def clean_physiological_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    In clinical cardiac registries, values of 0 for resting blood pressure (trestbps)
    or serum cholesterol (chol) represent unmeasured / missing data rather than
    physiological reality (cholesterol = 0 mg/dl is incompatible with life).
    We convert these to NaN so our median imputer handles them soundly.
    """
    df_clean = df.copy()
    if "chol" in df_clean.columns:
        df_clean.loc[df_clean["chol"] == 0, "chol"] = np.nan
    if "trestbps" in df_clean.columns:
        df_clean.loc[df_clean["trestbps"] == 0, "trestbps"] = np.nan
    return df_clean


def run_preprocessing():
    print("=" * 65)
    print(" CardioTriage - Executing Clinical Preprocessing Pipeline")
    print("=" * 65)

    # 1. Load raw datasets
    cleveland_raw = pd.read_csv(os.path.join(DATA_DIR, "cleveland_raw.csv"))
    hungary_raw = pd.read_csv(os.path.join(DATA_DIR, "hungary_raw.csv"))
    switzerland_raw = pd.read_csv(os.path.join(DATA_DIR, "switzerland_raw.csv"))
    va_raw = pd.read_csv(os.path.join(DATA_DIR, "va_long_beach_raw.csv"))

    print(f"[*] Ingested raw cohorts:")
    print(f"    - Cleveland:   {len(cleveland_raw)} records")
    print(f"    - Hungary:     {len(hungary_raw)} records")
    print(f"    - Switzerland: {len(switzerland_raw)} records")
    print(f"    - VA Long B.:  {len(va_raw)} records")

    # 2. Target binarization (0 = healthy, 1-4 = cardiac disease present)
    for name, df in [("Cleveland", cleveland_raw), ("Hungary", hungary_raw), 
                     ("Switzerland", switzerland_raw), ("VA", va_raw)]:
        df["target"] = (df["num"] > 0).astype(int)

    # 3. Clean physiological anomalies
    cleveland_clean = clean_physiological_anomalies(cleveland_raw)
    hungary_clean = clean_physiological_anomalies(hungary_raw)
    switzerland_clean = clean_physiological_anomalies(switzerland_raw)
    va_clean = clean_physiological_anomalies(va_raw)

    # 4. Strict Train / Test Split on Cleveland (In-Distribution)
    # Stratified by target to maintain balanced positive/negative ratio
    X_clev = cleveland_clean[FEATURE_COLS]
    y_clev = cleveland_clean["target"]

    X_train_raw, X_test_clev_raw, y_train, y_test_clev = train_test_split(
        X_clev, y_clev, test_size=0.20, random_state=42, stratify=y_clev
    )

    print(f"\n[*] Cleveland Data Split:")
    print(f"    - Training set:        {len(X_train_raw)} records (Disease prevalence: {y_train.mean():.2%})")
    print(f"    - Held-out Test set:   {len(X_test_clev_raw)} records (Disease prevalence: {y_test_clev.mean():.2%})")

    # 5. Strict Zero-Leakage Imputation & Feature Scaling
    # Fit imputer ONLY on training set
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)

    # Transform Cleveland Train and Test
    X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=FEATURE_COLS, index=X_train_raw.index)
    X_test_clev_imp = pd.DataFrame(imputer.transform(X_test_clev_raw), columns=FEATURE_COLS, index=X_test_clev_raw.index)

    # Fit scaler on imputed DataFrame with feature names preserved
    scaler = StandardScaler()
    scaler.fit(X_train_imp)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train_imp), columns=FEATURE_COLS, index=X_train_raw.index)
    X_test_clev_scaled = pd.DataFrame(scaler.transform(X_test_clev_imp), columns=FEATURE_COLS, index=X_test_clev_raw.index)

    # Transform External OOD Cohorts using the Cleveland-trained preprocessor
    # Hungary
    X_hung_raw = hungary_clean[FEATURE_COLS]
    y_hung = hungary_clean["target"]
    X_hung_imp = pd.DataFrame(imputer.transform(X_hung_raw), columns=FEATURE_COLS, index=X_hung_raw.index)
    X_hung_scaled = pd.DataFrame(scaler.transform(X_hung_imp), columns=FEATURE_COLS, index=X_hung_raw.index)

    # Switzerland
    X_swiz_raw = switzerland_clean[FEATURE_COLS]
    y_swiz = switzerland_clean["target"]
    X_swiz_imp = pd.DataFrame(imputer.transform(X_swiz_raw), columns=FEATURE_COLS, index=X_swiz_raw.index)
    X_swiz_scaled = pd.DataFrame(scaler.transform(X_swiz_imp), columns=FEATURE_COLS, index=X_swiz_raw.index)

    # VA Long Beach
    X_va_raw = va_clean[FEATURE_COLS]
    y_va = va_clean["target"]
    X_va_imp = pd.DataFrame(imputer.transform(X_va_raw), columns=FEATURE_COLS, index=X_va_raw.index)
    X_va_scaled = pd.DataFrame(scaler.transform(X_va_imp), columns=FEATURE_COLS, index=X_va_raw.index)

    # 6. Save Processed Files
    datasets_to_save = {
        "X_train.csv": X_train_imp,
        "X_train_scaled.csv": X_train_scaled,
        "y_train.csv": y_train.to_frame(),

        "X_test_cleveland.csv": X_test_clev_imp,
        "X_test_cleveland_scaled.csv": X_test_clev_scaled,
        "y_test_cleveland.csv": y_test_clev.to_frame(),

        "X_test_hungary.csv": X_hung_imp,
        "X_test_hungary_scaled.csv": X_hung_scaled,
        "y_test_hungary.csv": y_hung.to_frame(),

        "X_test_switzerland.csv": X_swiz_imp,
        "X_test_switzerland_scaled.csv": X_swiz_scaled,
        "y_test_switzerland.csv": y_swiz.to_frame(),

        "X_test_va.csv": X_va_imp,
        "X_test_va_scaled.csv": X_va_scaled,
        "y_test_va.csv": y_va.to_frame(),
    }

    for filename, df_obj in datasets_to_save.items():
        save_path = os.path.join(PROCESSED_DIR, filename)
        df_obj.to_csv(save_path, index=True)

    # 7. Serialize Artifacts (Pipeline Transformers & Metadata)
    pipeline_artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "training_medians": dict(zip(FEATURE_COLS, imputer.statistics_)),
        "metadata": {
            "n_train": len(X_train_raw),
            "n_test_cleveland": len(X_test_clev_raw),
            "n_ood_hungary": len(X_hung_raw),
            "n_ood_switzerland": len(X_swiz_raw),
            "n_ood_va": len(X_va_raw)
        }
    }
    joblib.dump(pipeline_artifacts, os.path.join(PROCESSED_DIR, "preprocessor.joblib"))

    print("\n[+] Processed datasets saved to `processed_data/` directory:")
    for k in datasets_to_save.keys():
        print(f"    - {k}")
    print("[+] Preprocessor and scalers serialized to `processed_data/preprocessor.joblib`")

    # 8. Forensic Summary & Sanity Verification
    print("\n" + "=" * 65)
    print(" PREPROCESSING VERIFICATION & INTEGRITY CHECK")
    print("=" * 65)
    print(f"Learned Median Imputation Values (from Cleveland Train):")
    for feat, med in pipeline_artifacts["training_medians"].items():
        print(f"    - {feat:10s}: {med:.2f}")

    print("\nTarget Prevalence Across Cohorts:")
    print(f"    - Cleveland Train (In-Dist): {y_train.mean():.2%} positive ({y_train.sum()}/{len(y_train)})")
    print(f"    - Cleveland Test (In-Dist):  {y_test_clev.mean():.2%} positive ({y_test_clev.sum()}/{len(y_test_clev)})")
    print(f"    - Hungary (OOD):            {y_hung.mean():.2%} positive ({y_hung.sum()}/{len(y_hung)})")
    print(f"    - Switzerland (OOD):        {y_swiz.mean():.2%} positive ({y_swiz.sum()}/{len(y_swiz)})")
    print(f"    - VA Long Beach (OOD):      {y_va.mean():.2%} positive ({y_va.sum()}/{len(y_va)})")

    # Check for zero NaNs in processed matrices
    all_clean = all(df.isna().sum().sum() == 0 for df in [X_train_imp, X_test_clev_imp, X_hung_imp, X_swiz_imp, X_va_imp])
    assert all_clean, "Integrity check failed: NaNs detected in processed outputs!"
    print("\n[SUCCESS] Integrity verified: 0 NaNs across all processed cohorts. Zero data leakage ensured.")
    print("=" * 65)


if __name__ == "__main__":
    run_preprocessing()
