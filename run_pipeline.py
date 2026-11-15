"""
CardioTriage - Master End-to-End Execution Pipeline
Veritas AI International High School Hackathon 2026 - AI + Health Track

Executes all pipeline stages deterministically in sequence:
1. Data Ingestion (UCI multi-hospital datasets, N=920)
2. Preprocessing & Zero-Leakage Partitioning
3. Cost-Sensitive Training & Asymmetric Threshold Optimization
4. Multi-Hospital OOD, Demographic Fairness & SHAP Forensics Audits
"""

import sys
import time

def run_master_pipeline():
    start_time = time.time()
    print("=" * 75)
    print(" CARDIOTRIAGE: FULL REPRODUCIBILITY PIPELINE")
    print(" Veritas AI International High School Hackathon 2026 | AI + Health")
    print(" Developed by: Krishna Sunil Wararkar (ST. Paul High School)")
    print("=" * 75)

    # Stage 1: Ingestion
    print("\n>>> STAGE 1: INGESTING MULTI-HOSPITAL CARDIAC DATASETS...")
    import data_loader
    data_loader.run_data_ingestion()

    # Stage 2: Preprocessing
    print("\n>>> STAGE 2: CLINICAL PREPROCESSING & ZERO-LEAKAGE SPLITTING...")
    import preprocess
    preprocess.run_preprocessing()

    # Stage 3: Model Training & Threshold Optimization
    print("\n>>> STAGE 3: TRAINING BASELINES & OPTIMIZING CLINICAL THRESHOLD...")
    import train_model
    train_model.run_training_pipeline()

    # Stage 4: Scientific Audits & Explainability
    print("\n>>> STAGE 4: EXECUTING MULTI-HOSPITAL OOD & SHAP FORENSICS AUDITS...")
    import audit
    audit.run_full_audit()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f" [SUCCESS] Master pipeline executed completely in {elapsed:.2f} seconds.")
    print(" All models, metrics, and audit figures are generated in `models/` and `outputs/`.")
    print(" To launch the interactive triage interface, run: python app.py")
    print("=" * 75)

if __name__ == "__main__":
    run_master_pipeline()
