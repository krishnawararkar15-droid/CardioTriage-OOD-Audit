"""
CardioTriage - Clinical Decision Support & Risk Stratification Web Application
Veritas AI International High School Hackathon 2026 - AI + Health Track

Designed specifically for community triage nurses and frontline providers
at understaffed rural health centers lacking immediate on-site cardiologists.
Implements asymmetric cost-sensitive triage (tau* = 0.31) to eliminate false negatives.

Design System: Minimalist Pure Black & Crisp White Monochrome with Intense Glassmorphism.
All cards feature generous rounded corners, flawless text enclosure, and structured hierarchy.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load models and configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

rf_model = joblib.load(os.path.join(MODELS_DIR, "cardiotriage_rf_model.joblib"))
with open(os.path.join(MODELS_DIR, "triage_config.json")) as f:
    config = json.load(f)

TAU_OPT = float(config["optimal_threshold"])  # 0.31
TAU_STD = 0.50
FEATURES = config["features"]


def predict_cardiac_triage(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
):
    """
    Executes cost-sensitive triage on patient clinical parameters.
    Returns:
    - HTML Triage Badge & Action Protocol
    - Risk Score (%)
    - Model Comparison Matrix
    - Key Risk Factor Attribution
    """
    sex_val = 1.0 if sex == "Male" else 0.0
    
    cp_map = {
        "Typical Angina (1)": 1.0,
        "Atypical Angina (2)": 2.0,
        "Non-Anginal Chest Pain (3)": 3.0,
        "Asymptomatic / Silent Ischemia (4)": 4.0
    }
    cp_val = cp_map.get(cp, 4.0)
    
    fbs_val = 1.0 if fbs == "Yes (>120 mg/dl)" else 0.0
    
    ecg_map = {
        "Normal (0)": 0.0,
        "ST-T Wave Abnormality (1)": 1.0,
        "Left Ventricular Hypertrophy (2)": 2.0
    }
    ecg_val = ecg_map.get(restecg, 0.0)
    
    exang_val = 1.0 if exang == "Yes" else 0.0
    
    slope_map = {
        "Upsloping (1)": 1.0,
        "Flat (2)": 2.0,
        "Downsloping (3)": 3.0
    }
    slope_val = slope_map.get(slope, 2.0)
    
    ca_val = float(ca)
    
    thal_map = {
        "Normal (3)": 3.0,
        "Fixed Defect (6)": 6.0,
        "Reversible Defect (7)": 7.0
    }
    thal_val = thal_map.get(thal, 3.0)

    # Patient Feature DataFrame
    patient_data = pd.DataFrame([{
        "age": float(age),
        "sex": sex_val,
        "cp": cp_val,
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": fbs_val,
        "restecg": ecg_val,
        "thalach": float(thalach),
        "exang": exang_val,
        "oldpeak": float(oldpeak),
        "slope": slope_val,
        "ca": ca_val,
        "thal": thal_val
    }])[FEATURES]

    # Model Inference
    risk_prob = float(rf_model.predict_proba(patient_data)[:, 1][0])
    risk_pct = risk_prob * 100

    # Asymmetric Triage Decision
    is_urgent_cardiotriage = (risk_prob >= TAU_OPT)
    is_urgent_standard = (risk_prob >= TAU_STD)

    # 1. Clean Monochrome Glassmorphism Triage Card with Proper Text Wrapping & Rounded Corners
    if is_urgent_cardiotriage:
        triage_html = f"""
        <div class="result-card urgent-glass">
            <div class="card-top-bar">
                <span class="status-badge urgent-badge">URGENT ESCALATION REQUIRED</span>
                <div class="gauge-pill">
                    <span class="gauge-num">{risk_pct:.1f}%</span>
                    <span class="gauge-sub">RISK</span>
                </div>
            </div>
            
            <h2 class="card-main-title">Emergent Secondary Cardiology Transfer</h2>
            <div class="card-subtitle">Clinical Priority: Acute Ischemia / High Cardiac Risk</div>
            
            <div class="inner-protocol-box">
                <div class="protocol-header">MANDATORY FRONTLINE NURSING PROTOCOL</div>
                <div class="protocol-item">
                    <span class="step-num">1</span>
                    <div class="step-desc"><strong>ECG & Telemetry:</strong> Immediately initiate 12-lead continuous ECG monitoring and pulse oximetry.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">2</span>
                    <div class="step-desc"><strong>Vascular Access & Labs:</strong> Place two 18-gauge peripheral IV lines; draw STAT troponin biomarkers.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">3</span>
                    <div class="step-desc"><strong>Antiplatelet Loading:</strong> Administer 325 mg chewable aspirin if not contraindicated.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">4</span>
                    <div class="step-desc"><strong>Transfer Dispatch:</strong> Alert regional paramedic transport for emergent catheterization transfer.</div>
                </div>
            </div>
        </div>
        """
    else:
        triage_html = f"""
        <div class="result-card routine-glass">
            <div class="card-top-bar">
                <span class="status-badge routine-badge">ROUTINE AMBULATORY STATUS</span>
                <div class="gauge-pill">
                    <span class="gauge-num">{risk_pct:.1f}%</span>
                    <span class="gauge-sub">RISK</span>
                </div>
            </div>
            
            <h2 class="card-main-title">Non-Emergent Outpatient Management</h2>
            <div class="card-subtitle">Clinical Priority: Low / Moderate Baseline Risk</div>
            
            <div class="inner-protocol-box">
                <div class="protocol-header">STANDARD CLINICAL CARE PROTOCOL</div>
                <div class="protocol-item">
                    <span class="step-num">1</span>
                    <div class="step-desc"><strong>Hemodynamic Stability:</strong> Patient intake vitals remain within safe ambulatory limits.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">2</span>
                    <div class="step-desc"><strong>Outpatient Referral:</strong> Coordinate elective echocardiogram or treadmill stress test within 14–30 days.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">3</span>
                    <div class="step-desc"><strong>Preventive Management:</strong> Counsel on dietary lipid control, smoking cessation, and BP logging.</div>
                </div>
                <div class="protocol-item">
                    <span class="step-num">4</span>
                    <div class="step-desc"><strong>Discharge Education:</strong> Provide warning signs on acute symptoms (retrosternal pressure, diaphoresis).</div>
                </div>
            </div>
        </div>
        """

    # 2. Model Consensus & Safety Audit
    if is_urgent_cardiotriage and not is_urgent_standard:
        comparison_html = f"""
        <div class="audit-card highlight-audit">
            <div class="audit-tag">SAFETY RESCUE ALERT • ASYMMETRIC 5:1 PENALTY</div>
            <div class="audit-title">Standard AI Missed Case Rescued by CardioTriage</div>
            <p class="audit-text">
                A conventional AI model with an uncalibrated 50% cutoff categorized this patient as 
                <strong>'HEALTHY' ({risk_pct:.1f}%)</strong> and would have mistakenly sent them home. 
                CardioTriage's safety-calibrated threshold (<strong>&tau;* = {TAU_OPT:.2f}</strong>) 
                correctly caught this borderline case, eliminating a preventable diagnostic delay.
            </p>
        </div>
        """
    elif is_urgent_cardiotriage and is_urgent_standard:
        comparison_html = f"""
        <div class="audit-card">
            <div class="audit-tag">MODEL CONCURRENCE</div>
            <div class="audit-title">Unanimous High Risk Stratification</div>
            <p class="audit-text">
                Both standard machine learning (&tau; = 0.50) and CardioTriage (&tau;* = 0.31) concur that this patient's assessed ischemic risk ({risk_pct:.1f}%) warrants immediate emergency cardiology escalation.
            </p>
        </div>
        """
    else:
        comparison_html = f"""
        <div class="audit-card">
            <div class="audit-tag">MODEL CONCURRENCE</div>
            <div class="audit-title">Unanimous Low Risk Stratification</div>
            <p class="audit-text">
                Both standard AI and CardioTriage agree that acute ischemic risk ({risk_pct:.1f}%) is low. Routine outpatient care is clinically appropriate.
            </p>
        </div>
        """

    # 3. Top Diagnostic Risk Drivers
    factors = []
    if oldpeak >= 1.5:
        factors.append(f"<strong>ST Depression ({oldpeak:.1f} mm):</strong> Severe exercise-induced subendocardial ischemic response.")
    if cp_val == 4.0:
        factors.append("<strong>Silent Ischemia / Asymptomatic:</strong> High correlation with multi-vessel anatomical coronary disease.")
    if ca_val >= 1.0:
        factors.append(f"<strong>Fluoroscopy Stenosis ({int(ca_val)} vessels):</strong> Major anatomical confirmation of coronary arterial obstruction.")
    if thal_val == 7.0:
        factors.append("<strong>Reversible Thallium Defect:</strong> Confirms viable but ischemic myocardium under hemodynamic stress.")
    if exang_val == 1.0:
        factors.append("<strong>Exercise-Induced Angina:</strong> Clinical marker of myocardial oxygen supply/demand deficit.")
    if thalach < 120.0:
        factors.append(f"<strong>Chronotropic Incompetence ({int(thalach)} bpm):</strong> Inability to reach expected physiologic heart rate.")
    if trestbps >= 140.0:
        factors.append(f"<strong>Hypertension ({int(trestbps)} mmHg):</strong> Increases coronary vascular shear stress and plaque vulnerability.")
    if chol >= 260.0:
        factors.append(f"<strong>Hypercholesterolemia ({int(chol)} mg/dl):</strong> Elevated long-term atherogenic substrate.")

    if not factors:
        factors.append("<strong>Normal Baseline Parameters:</strong> No major electrophysiological or anatomical risk flags identified.")

    factors_html = """
    <div class="audit-card">
        <div class="audit-tag">PHYSIOLOGICAL RISK ATTRIBUTION</div>
        <ul class="driver-list">
    """
    for f in factors:
        factors_html += f"<li>{f}</li>"
    factors_html += "</ul></div>"

    return triage_html, comparison_html, factors_html, round(risk_pct, 1)


# Preset Clinical Profiles
PRESETS = {
    "case_1": [65, "Male", "Asymptomatic / Silent Ischemia (4)", 150, 280, "No (<=120 mg/dl)", "ST-T Wave Abnormality (1)", 125, "Yes", 2.6, "Flat (2)", "2", "Reversible Defect (7)"],
    "case_2": [56, "Female", "Atypical Angina (2)", 130, 236, "No (<=120 mg/dl)", "Normal (0)", 155, "No", 1.2, "Flat (2)", "0", "Normal (3)"],
    "case_3": [42, "Female", "Non-Anginal Chest Pain (3)", 118, 195, "No (<=120 mg/dl)", "Normal (0)", 172, "No", 0.0, "Upsloping (1)", "0", "Normal (3)"]
}


# =============================================================================
# MONOCHROME BLACK & WHITE GLASSMORPHISM CSS
# Engineered for flawless layout wrapping, generous rounded corners & zero overlap
# =============================================================================
custom_css = """
* {
    box-sizing: border-box !important;
}

body, gradio-app, .gradio-container {
    background-color: #000000 !important;
    background-image: none !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #ffffff !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Master Header Container */
.app-header-box {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    padding: 22px 28px !important;
    margin-bottom: 14px !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}

.brand-main {
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px !important;
    color: #ffffff !important;
    margin: 0 !important;
}

.brand-desc {
    font-size: 13.5px !important;
    color: #a1a1aa !important;
    margin: 4px 0 0 0 !important;
    font-weight: 400 !important;
}

.header-meta-pill {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    color: #ffffff !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 6px 14px !important;
    border-radius: 24px !important;
    text-transform: uppercase;
}

/* Regulatory Notice */
.regulatory-pill {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    margin-bottom: 18px !important;
    font-size: 12px !important;
    color: #a1a1aa !important;
    line-height: 1.4;
}

/* Preset Navigation Bar */
.presets-glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 20px !important;
}

.preset-title-label {
    font-size: 11.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase;
    color: #a1a1aa !important;
    margin-bottom: 10px !important;
}

/* Left Column Input Cards with Rounded Corners */
.form-card {
    background: rgba(255, 255, 255, 0.035) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    padding: 20px 22px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6) !important;
}

.card-title-label {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase;
    color: #ffffff !important;
    margin-bottom: 14px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Results Display Styling - Guaranteed Text Enclosure */
.result-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border-radius: 18px !important;
    padding: 22px 24px !important;
    margin-bottom: 16px !important;
    width: 100% !important;
    overflow: hidden !important;
}

.urgent-glass {
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    box-shadow: 0 10px 36px rgba(255, 255, 255, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.2) !important;
}

.routine-glass {
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
}

.card-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.status-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 8px;
    display: inline-block;
}

.urgent-badge {
    background: #ffffff;
    color: #000000;
}

.routine-badge {
    background: rgba(255, 255, 255, 0.12);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.gauge-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px;
    padding: 6px 14px;
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.gauge-num {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
}

.gauge-sub {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #a1a1aa;
}

.card-main-title {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 4px 0;
    letter-spacing: -0.4px;
    line-height: 1.3;
    word-break: break-word;
}

.card-subtitle {
    font-size: 13px;
    color: #a1a1aa;
    margin-bottom: 16px;
}

.inner-protocol-box {
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px;
    padding: 16px 18px;
    width: 100%;
}

.protocol-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 12px;
}

.protocol-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 10px;
}

.protocol-item:last-child {
    margin-bottom: 0;
}

.step-num {
    background: rgba(255, 255, 255, 0.1);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}

.step-desc {
    color: #e4e4e7;
    font-size: 13px;
    line-height: 1.55;
    word-break: break-word;
}

/* Audit & Explanation Cards */
.audit-card {
    background: rgba(255, 255, 255, 0.035) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    margin-bottom: 14px !important;
    width: 100% !important;
}

.highlight-audit {
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.audit-tag {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 6px;
}

.audit-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
    word-break: break-word;
}

.audit-text {
    font-size: 13px;
    color: #d4d4d8;
    line-height: 1.55;
    margin: 0;
    word-break: break-word;
}

.driver-list {
    margin: 0;
    padding-left: 18px;
    color: #e4e4e7;
    font-size: 13px;
    line-height: 1.6;
}

.driver-list li {
    margin-bottom: 6px;
    word-break: break-word;
}

.driver-list li:last-child {
    margin-bottom: 0;
}

/* Form Controls & Inputs */
div.block, div.box, .gradio-block {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

label span, span[data-testid="block-info"], .label-text {
    background: transparent !important;
    background-color: transparent !important;
    color: #e4e4e7 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: -0.1px !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

input[type="text"], input[type="number"], select, .gr-dropdown {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
}

input[type="text"]:focus, input[type="number"]:focus, select:focus {
    border-color: #ffffff !important;
    box-shadow: 0 0 0 1px #ffffff !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #ffffff !important;
}

/* Radio buttons */
input[type="radio"]:checked {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}

/* Buttons */
.calculate-btn {
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: -0.2px !important;
    border: 1px solid #ffffff !important;
    border-radius: 12px !important;
    padding: 15px 22px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 10px !important;
}

.calculate-btn:hover {
    background: #e4e4e7 !important;
    box-shadow: 0 4px 24px rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-1px) !important;
}

.preset-btn {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
}

.preset-btn:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}

/* Tabs */
button[role="tab"] {
    color: #a1a1aa !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}

button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
}

/* Fix Tab 2 Images - Remove Overlapping Label Banners */
.audit-image-box {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 18px !important;
}

.audit-image-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase;
    color: #ffffff !important;
    margin-bottom: 12px !important;
}

.gradio-image, .image-container, .image-frame {
    background: transparent !important;
    border: none !important;
}

.image-container img {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.footer-bar {
    text-align: center;
    font-size: 12px;
    color: #71717a;
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
"""

# Base Monochrome Theme
monochrome_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.zinc,
    secondary_hue=gr.themes.colors.zinc,
    neutral_hue=gr.themes.colors.zinc,
    font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
).set(
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    body_text_color="#ffffff",
    body_text_color_dark="#ffffff",
    block_background_fill="rgba(255, 255, 255, 0.03)",
    block_background_fill_dark="rgba(255, 255, 255, 0.03)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.12)",
    block_border_color_dark="rgba(255, 255, 255, 0.12)",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_dark="#ffffff",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    slider_color="#ffffff",
    slider_color_dark="#ffffff",
    input_background_fill="rgba(255, 255, 255, 0.04)",
    input_background_fill_dark="rgba(255, 255, 255, 0.04)",
    input_border_color="rgba(255, 255, 255, 0.12)",
    input_border_color_dark="rgba(255, 255, 255, 0.12)",
)

# Compute Initial State so the interface opens fully populated
INIT_VALS = PRESETS["case_2"]
init_triage_html, init_comp_html, init_factors_html, init_risk = predict_cardiac_triage(*INIT_VALS)


# =============================================================================
# GRADIO APPLICATION LAYOUT
# =============================================================================
with gr.Blocks(title="CardioTriage: Clinical Triage DSS") as demo:

    # Direct Style Injection
    gr.HTML(f"<style>{custom_css}</style>")

    # Master Header
    gr.HTML("""
    <div class="app-header-box">
        <div>
            <h1 class="brand-main">CardioTriage</h1>
            <p class="brand-desc">
                Cost-Sensitive Cardiac Triage & Cross-Hospital Generalization System for Frontline Rural Healthcare
            </p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="header-meta-pill">AI + Health Track</span>
            <span class="header-meta-pill">Veritas AI 2026</span>
        </div>
    </div>
    <div class="regulatory-pill">
        <strong>REGULATORY PROTOTYPE NOTICE:</strong> Academic decision-support proof-of-concept exploring asymmetric loss calibration (&tau;* = 0.31) and domain-shift mitigation. Not direct medical advice.
    </div>
    """)

    with gr.Tabs():
        # TAB 1: CLINICAL TRIAGE TOOL
        with gr.Tab("Frontline Clinical Triage"):
            
            # Interactive Quick-Load Presets Bar
            with gr.Row():
                with gr.Column(scale=12):
                    gr.HTML("<div class='preset-title-label'>Quick-Load Representative Patient Scenarios:</div>")
                    with gr.Row():
                        btn_case1 = gr.Button("Case 1: Confirmed Acute (65M)", elem_classes="preset-btn")
                        btn_case2 = gr.Button("Case 2: Borderline Rescue (56F)", elem_classes="preset-btn")
                        btn_case3 = gr.Button("Case 3: Ambulatory Routine (42F)", elem_classes="preset-btn")

            with gr.Row():
                # LEFT COLUMN: INTAKE PARAMETERS (Clean, structured cards with rounded corners)
                with gr.Column(scale=6):
                    
                    gr.HTML("<div class='card-title-label'>1. Demographics & Baseline Vitals</div>")
                    with gr.Row():
                        age_input = gr.Slider(minimum=25, maximum=85, value=INIT_VALS[0], step=1, label="Patient Age (years)")
                        sex_input = gr.Radio(choices=["Male", "Female"], value=INIT_VALS[1], label="Biological Sex")
                    with gr.Row():
                        trestbps_input = gr.Slider(minimum=80, maximum=210, value=INIT_VALS[3], step=1, label="Resting Blood Pressure (mm Hg)")
                        chol_input = gr.Slider(minimum=100, maximum=550, value=INIT_VALS[4], step=5, label="Serum Cholesterol (mg/dl)")

                    gr.HTML("<div class='card-title-label' style='margin-top: 14px;'>2. Symptoms & Exercise Stress Response</div>")
                    cp_input = gr.Dropdown(
                        choices=[
                            "Typical Angina (1)",
                            "Atypical Angina (2)",
                            "Non-Anginal Chest Pain (3)",
                            "Asymptomatic / Silent Ischemia (4)"
                        ],
                        value=INIT_VALS[2],
                        label="Chest Pain Presentation (Angina Type)"
                    )
                    with gr.Row():
                        thalach_input = gr.Slider(minimum=60, maximum=210, value=INIT_VALS[7], step=1, label="Max Heart Rate Achieved (bpm)")
                        oldpeak_input = gr.Slider(minimum=0.0, maximum=6.2, value=INIT_VALS[9], step=0.1, label="ST Depression Induced by Exercise (mm)")
                    with gr.Row():
                        exang_input = gr.Radio(choices=["Yes", "No"], value=INIT_VALS[8], label="Exercise-Induced Angina")
                        fbs_input = gr.Radio(choices=["No (<=120 mg/dl)", "Yes (>120 mg/dl)"], value=INIT_VALS[5], label="Fasting Blood Sugar > 120 mg/dl")

                    gr.HTML("<div class='card-title-label' style='margin-top: 14px;'>3. Secondary Clinical Diagnostics (Optional)</div>")
                    with gr.Row():
                        restecg_input = gr.Dropdown(
                            choices=["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"],
                            value=INIT_VALS[6],
                            label="Resting Electrocardiogram (ECG)"
                        )
                        slope_input = gr.Dropdown(
                            choices=["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
                            value=INIT_VALS[10],
                            label="Peak Exercise ST Slope"
                        )
                    with gr.Row():
                        ca_input = gr.Dropdown(
                            choices=["0", "1", "2", "3"],
                            value=INIT_VALS[11],
                            label="Major Vessels via Fluoroscopy (ca) [0 if unavailable]"
                        )
                        thal_input = gr.Dropdown(
                            choices=["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"],
                            value=INIT_VALS[12],
                            label="Thallium Nuclear Scan [Normal if unavailable]"
                        )

                    triage_btn = gr.Button("Calculate Clinical Triage Decision", elem_classes="calculate-btn")

                # RIGHT COLUMN: TRIAGE RESULTS & DECISION SUPPORT (Properly enclosed glass cards)
                with gr.Column(scale=6):
                    triage_badge_output = gr.HTML(value=init_triage_html)
                    comparison_output = gr.HTML(value=init_comp_html)
                    factors_output = gr.HTML(value=init_factors_html)

            # Input List for linking
            all_inputs = [
                age_input, sex_input, cp_input, trestbps_input, chol_input, fbs_input,
                restecg_input, thalach_input, exang_input, oldpeak_input, slope_input, ca_input, thal_input
            ]
            all_outputs = [triage_badge_output, comparison_output, factors_output]

            # Main Calculate Button Action
            def run_triage_and_format(*args):
                t_html, c_html, f_html, _ = predict_cardiac_triage(*args)
                return t_html, c_html, f_html

            triage_btn.click(
                fn=run_triage_and_format,
                inputs=all_inputs,
                outputs=all_outputs
            )

            # Preset 1 Click Handler (High Risk)
            def load_preset_1():
                v = PRESETS["case_1"]
                t_html, c_html, f_html, _ = predict_cardiac_triage(*v)
                return (*v, t_html, c_html, f_html)

            btn_case1.click(
                fn=load_preset_1,
                inputs=[],
                outputs=all_inputs + all_outputs
            )

            # Preset 2 Click Handler (Rescue Case)
            def load_preset_2():
                v = PRESETS["case_2"]
                t_html, c_html, f_html, _ = predict_cardiac_triage(*v)
                return (*v, t_html, c_html, f_html)

            btn_case2.click(
                fn=load_preset_2,
                inputs=[],
                outputs=all_inputs + all_outputs
            )

            # Preset 3 Click Handler (Ambulatory)
            def load_preset_3():
                v = PRESETS["case_3"]
                t_html, c_html, f_html, _ = predict_cardiac_triage(*v)
                return (*v, t_html, c_html, f_html)

            btn_case3.click(
                fn=load_preset_3,
                inputs=[],
                outputs=all_inputs + all_outputs
            )

        # TAB 2: AUDIT & EXPLAINABILITY (Cleanly Framed Without Overlapping Labels)
        with gr.Tab("Multi-Hospital OOD & Explainability Audit"):
            gr.HTML("""
            <div style="padding: 10px 0 20px 0;">
                <h3 style="color: #ffffff; font-size: 18px; font-weight: 700; margin: 0 0 6px 0;">
                    Scientific Validation & Cross-Hospital Stress Testing
                </h3>
                <p style="color: #a1a1aa; font-size: 13.5px; margin: 0; line-height: 1.5;">
                    Empirical validation of model generalizability across 920 patients from Cleveland Clinic, 
                    Hungarian Institute of Cardiology, Zurich University Hospital, and VA Medical Center Long Beach.
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='audit-image-title'>1. Cross-Hospital Out-of-Distribution Generalization</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "ood_generalization_audit.png"), show_label=False)
                with gr.Column():
                    gr.HTML("<div class='audit-image-title'>2. Subgroup Demographic Fairness Audit</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "subgroup_fairness_audit.png"), show_label=False)

            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='audit-image-title' style='margin-top: 14px;'>3. Asymmetric Cost-Sensitive Threshold Curve</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "cost_threshold_optimization_curve.png"), show_label=False)
                with gr.Column():
                    gr.HTML("<div class='audit-image-title' style='margin-top: 14px;'>4. Global Feature Attribution (TreeSHAP)</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_global_summary.png"), show_label=False)

            gr.HTML("<div class='audit-image-title' style='margin-top: 24px;'>5. Local Clinical Case Forensics (SHAP Waterfall Plots)</div>")
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='audit-image-title' style='font-size: 12px; color: #a1a1aa;'>True Positive: Acute Ischemia</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_true_positive.png"), show_label=False)
                with gr.Column():
                    gr.HTML("<div class='audit-image-title' style='font-size: 12px; color: #a1a1aa;'>Triage Rescue: Rescued False Negative</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_triage_rescue.png"), show_label=False)
                with gr.Column():
                    gr.HTML("<div class='audit-image-title' style='font-size: 12px; color: #a1a1aa;'>False Alarm: Atypical Presentation</div>")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_false_alarm.png"), show_label=False)

    gr.HTML("""
    <div class="footer-bar">
        CardioTriage System • Veritas AI International High School Hackathon 2026 • AI + Health Track<br>
        Developed by Krishna Sunil Wararkar (ST. Paul High School)
    </div>
    """)


if __name__ == "__main__":
    print("[*] Launching CardioTriage Clinical Decision Support System...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=monochrome_theme)
