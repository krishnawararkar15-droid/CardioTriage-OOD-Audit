"""
CardioTriage - Data Ingestion & Multi-Hospital Ingestion Pipeline
Veritas AI International High School Hackathon 2026 - AI + Health Track

Ingests raw data across four medical institutions:
1. Cleveland Clinic Foundation (Cleveland, USA)
2. Hungarian Institute of Cardiology (Budapest, Hungary)
3. University Hospital, Zurich (Switzerland)
4. V.A. Medical Center (Long Beach, California, USA)
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np

# Directory setup
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Standard 14 clinical features defined by Detrano et al. / UCI Machine Learning Repository
FEATURE_NAMES = [
    "age",        # Age in years
    "sex",        # 1 = male, 0 = female
    "cp",         # Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)
    "trestbps",   # Resting blood pressure in mm Hg
    "chol",       # Serum cholesterol in mg/dl
    "fbs",        # Fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",    # Resting ECG (0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy)
    "thalach",    # Maximum heart rate achieved
    "exang",      # Exercise induced angina (1=yes, 0=no)
    "oldpeak",    # ST depression induced by exercise relative to rest
    "slope",      # Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
    "ca",         # Number of major vessels (0-3) colored by flourosopy
    "thal",       # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num"         # Angiographic heart disease status (0 = <50% stenosis, 1-4 = >50% stenosis)
]

# URLs for raw datasets from official UCI Machine Learning Repository and reliable academic mirrors
UCI_SOURCES = {
    "Cleveland": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/datasets/heart-disease-uci/master/data/processed.cleveland.data"
    ],
    "Hungary": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "https://raw.githubusercontent.com/datasets/heart-disease-uci/master/data/processed.hungarian.data"
    ],
    "Switzerland": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "https://raw.githubusercontent.com/datasets/heart-disease-uci/master/data/processed.switzerland.data"
    ],
    "VA Long Beach": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
        "https://raw.githubusercontent.com/datasets/heart-disease-uci/master/data/processed.va.data"
    ]
}

# Unified CSV backup source (standard Kaggle/Kaggle mirror format with 'dataset' or 'origin' column)
UNIFIED_BACKUP_URLS = [
    "https://raw.githubusercontent.com/redwankarimsony/heart-disease-data/main/heart_disease_uci.csv",
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/heart.csv"
]


def download_file(urls, destination_path):
    """Attempt download from multiple mirror URLs."""
    for url in urls:
        try:
            print(f"[*] Downloading from: {url}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response, open(destination_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[+] Successfully downloaded: {destination_path}")
            return True
        except Exception as e:
            print(f"[-] Failed with {url}: {e}. Trying fallback...")
    return False


def load_hospital_raw(name, urls):
    """Load individual hospital raw data into a DataFrame with standard schema."""
    filename = f"raw_{name.lower().replace(' ', '_')}.data"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        success = download_file(urls, filepath)
        if not success:
            return None

    try:
        # Raw UCI data files are comma-separated with '?' as missing value
        df = pd.read_csv(filepath, header=None, names=FEATURE_NAMES, na_values="?", sep=",", skipinitialspace=True)
        # Clean potential trailing spaces or formatting issues
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["origin"] = name
        return df
    except Exception as e:
        print(f"[-] Error reading {filepath}: {e}")
        return None


def fetch_from_unified_backup():
    """Fallback: fetch the unified multi-hospital CSV if individual raw files fail."""
    unified_path = os.path.join(DATA_DIR, "heart_disease_uci.csv")
    if not os.path.exists(unified_path):
        success = download_file(UNIFIED_BACKUP_URLS, unified_path)
        if not success:
            raise RuntimeError("Could not download heart disease datasets from any source.")

    df_raw = pd.read_csv(unified_path)
    # Normalize column names
    col_map = {
        'dataset': 'origin',
        'trestbps': 'trestbps',
        'chol': 'chol',
        'thalch': 'thalach',
        'thalach': 'thalach'
    }
    df_raw.rename(columns=col_map, inplace=True)
    return df_raw


def run_data_ingestion():
    print("=" * 60)
    print(" CardioTriage - Ingesting Multi-Hospital Cardiac Datasets")
    print("=" * 60)

    hospital_dfs = []
    failed_hospitals = []

    for hospital_name, urls in UCI_SOURCES.items():
        df = load_hospital_raw(hospital_name, urls)
        if df is not None and len(df) > 0:
            hospital_dfs.append(df)
            print(f"[OK] Loaded {hospital_name}: {len(df)} patient records.")
        else:
            failed_hospitals.append(hospital_name)

    if hospital_dfs and len(failed_hospitals) == 0:
        combined_df = pd.concat(hospital_dfs, ignore_index=True)
    else:
        print("[!] Using unified multi-hospital repository fallback...")
        combined_df = fetch_from_unified_backup()

    # Save individual hospital CSVs and combined dataset
    combined_csv_path = os.path.join(DATA_DIR, "heart_disease_multihospital_raw.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\n[+] Saved full multi-hospital dataset to: {combined_csv_path}")

    # Generate individual hospital CSVs
    for origin_name in combined_df["origin"].unique():
        sub_df = combined_df[combined_df["origin"] == origin_name]
        safe_name = origin_name.lower().replace(" ", "_")
        sub_path = os.path.join(DATA_DIR, f"{safe_name}_raw.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"    - {origin_name} saved ({len(sub_df)} records) -> {sub_path}")

    # Print Clinical Dataset Forensics
    print("\n" + "=" * 60)
    print(" DATASET FORENSIC AUDIT (Multi-Hospital Breakdown)")
    print("=" * 60)
    print(f"Total Combined Records: {len(combined_df)}")
    print(f"Features: {list(combined_df.columns)}")
    print("\nRecords per Hospital:")
    print(combined_df["origin"].value_counts())

    print("\nMissing Values per Hospital:")
    missing_by_origin = combined_df.groupby("origin").apply(lambda x: x.isnull().sum())
    print(missing_by_origin[["trestbps", "chol", "thalach", "ca", "thal", "slope"]])

    print("\nTarget Distribution (`num` 0: No disease, 1-4: Disease present):")
    print(combined_df.groupby("origin")["num"].value_counts(dropna=False).unstack().fillna(0))

    print("\n[SUCCESS] Phase 1 Data Ingestion completed successfully.")
    return combined_df


if __name__ == "__main__":
    run_data_ingestion()
