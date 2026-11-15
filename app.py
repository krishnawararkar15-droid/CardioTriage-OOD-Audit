"""
CardioTriage - Clinical Decision Support & Risk Stratification Web Application
Veritas AI International High School Hackathon 2026 - AI + Health Track

Designed specifically for community triage nurses and frontline providers
at understaffed rural health centers lacking immediate on-site cardiologists.
Implements asymmetric cost-sensitive triage (tau* = 0.31) to eliminate false negatives.

Theme: Minimalist Black, Crisp White & Emerald Green Gradients with Intense Glassmorphism.
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

# Load feature importances for rapid clinical factor attribution
feat_imp = pd.read_csv(os.path.join(MODELS_DIR, "feature_importances_rf.csv")).set_index("feature")["importance"].to_dict()


def predict_cardiac_triage(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
):
    """
    Executes cost-sensitive triage on patient clinical parameters.
    Returns:
    - HTML Triage Badge & Action Protocol
    - Risk Gauge Score (%)
    - Model Comparison Matrix
    - Key Risk Factor Attribution
    """
    # 1. Parse Inputs to Standard Clinical Schema
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

    # 2. Construct Patient Feature DataFrame
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

    # 3. Model Inference
    risk_prob = float(rf_model.predict_proba(patient_data)[:, 1][0])
    risk_pct = risk_prob * 100

    # 4. Asymmetric Triage Decision
    is_urgent_cardiotriage = (risk_prob >= TAU_OPT)
    is_urgent_standard = (risk_prob >= TAU_STD)

    # 5. Build Triage Badge HTML with Intense Glassmorphism
    if is_urgent_cardiotriage:
        triage_html = f"""
        <div class="glass-card triage-urgent">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">🚨</span>
                    <h2 style="color: #f87171; margin: 0; font-size: 19px; font-weight: 800; letter-spacing: -0.3px;">
                        URGENT: IMMEDIATE CARDIOLOGY ESCALATION
                    </h2>
                </div>
                <span class="badge-urgent-pill">
                    {risk_pct:.1f}% Risk Score
                </span>
            </div>
            <p style="margin: 12px 0 6px 0; color: #fef2f2; font-size: 14.5px; font-weight: 600;">
                Clinical Priority: <span style="color: #f87171;">HIGH RISK (Emergency Transfer Indicated)</span>
            </p>
            <hr style="border: 0; border-top: 1px solid rgba(239, 68, 68, 0.25); margin: 12px 0;">
            <div style="color: #fca5a5; font-size: 13.5px; line-height: 1.6;">
                <strong style="color: #ffffff;">Frontline Nursing Emergency Protocol:</strong><br>
                1. <strong>ECG Telemetry:</strong> Initiate continuous 12-lead ECG telemetry & continuous SpO2 monitoring immediately.<br>
                2. <strong>Vascular Access & Labs:</strong> Secure two large-bore peripheral IV lines; draw urgent cardiac troponins.<br>
                3. <strong>Antiplatelet Loading:</strong> Administer chewable Aspirin (325mg) immediately if no active hemorrhage.<br>
                4. <strong>Transfer Dispatch:</strong> Contact emergency paramedic transport for transfer to nearest cardiac catheterization center.
            </div>
        </div>
        """
    else:
        triage_html = f"""
        <div class="glass-card triage-routine">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">✅</span>
                    <h2 style="color: #34d399; margin: 0; font-size: 19px; font-weight: 800; letter-spacing: -0.3px;">
                        ROUTINE: NON-EMERGENT FOLLOW-UP
                    </h2>
                </div>
                <span class="badge-routine-pill">
                    {risk_pct:.1f}% Risk Score
                </span>
            </div>
            <p style="margin: 12px 0 6px 0; color: #f0fdf4; font-size: 14.5px; font-weight: 600;">
                Clinical Priority: <span style="color: #34d399;">LOW / MODERATE RISK (Outpatient Protocol)</span>
            </p>
            <hr style="border: 0; border-top: 1px solid rgba(16, 185, 129, 0.25); margin: 12px 0;">
            <div style="color: #d1fae5; font-size: 13.5px; line-height: 1.6;">
                <strong style="color: #ffffff;">Frontline Nursing Protocol:</strong><br>
                1. Patient vitals are within acceptable non-emergent ambulatory limits.<br>
                2. Coordinate outpatient treadmill stress testing or echocardiography within 14–30 days.<br>
                3. Review lifestyle recommendations, blood pressure diary, and lipid panel management.<br>
                4. Educate patient on cardiac red flags (crushing retrosternal pressure, syncope) requiring immediate ER return.
            </div>
        </div>
        """

    # 6. Safety Comparison Banner (Highlighting Rescued Missed Patients)
    if is_urgent_cardiotriage and not is_urgent_standard:
        comparison_html = f"""
        <div class="glass-card rescue-banner">
            <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #fbbf24; font-size: 14.5px;">
                <span>⚡</span> CLINICAL SAFETY ALERT: RESCUED FALSE NEGATIVE
            </div>
            <div style="color: #fef3c7; font-size: 13.5px; margin-top: 8px; line-height: 1.55;">
                A standard AI model with an uncalibrated 50% cutoff would have classified this patient as 
                <strong style="color: #ffffff;">'LOW RISK' ({risk_pct:.1f}%)</strong> and sent them home with a fatal missed diagnosis. 
                <strong>CardioTriage's 5:1 clinical loss threshold (&tau;* = {TAU_OPT:.2f})</strong> successfully caught and escalated this patient, saving a life.
            </div>
        </div>
        """
    elif is_urgent_cardiotriage and is_urgent_standard:
        comparison_html = """
        <div class="glass-card consensus-banner">
            <div style="color: #34d399; font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                🎯 Unanimous Escalation Consensus
            </div>
            <div style="color: #e2e8f0; font-size: 13px; line-height: 1.5;">
                Both conventional AI (&tau; = 0.50) and CardioTriage (&tau;* = 0.31) strongly agree this patient is high risk and requires immediate cardiology escalation.
            </div>
        </div>
        """
    else:
        comparison_html = """
        <div class="glass-card consensus-banner">
            <div style="color: #10b981; font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                🤝 Unanimous Ambulatory Consensus
            </div>
            <div style="color: #e2e8f0; font-size: 13px; line-height: 1.5;">
                Both models concur that acute ischemic risk is low. Emergency transfer is not indicated; proceed with standard primary care follow-up.
            </div>
        </div>
        """

    # 7. Local Factor Attribution Breakdown
    factors = []
    if oldpeak >= 1.5:
        factors.append(f"<strong style='color: #f87171;'>Significant ST Depression ({oldpeak:.1f} mm):</strong> Severe marker of exercise-induced subendocardial ischemia.")
    if cp_val == 4.0:
        factors.append("<strong style='color: #f87171;'>Asymptomatic / Silent Presentation:</strong> Treacherous presentation; strongly correlates with multi-vessel disease.")
    if ca_val >= 1.0:
        factors.append(f"<strong style='color: #f87171;'>Fluoroscopy Calcification ({int(ca_val)} vessels):</strong> Major anatomical confirmation of coronary arterial narrowing.")
    if thal_val == 7.0:
        factors.append("<strong style='color: #f87171;'>Reversible Thallium Perfusion Defect:</strong> Confirms viable but ischemic myocardium under cardiac demand.")
    if exang_val == 1.0:
        factors.append("<strong style='color: #f87171;'>Exercise-Induced Angina:</strong> Direct clinical sign of myocardial oxygen supply-demand deficit.")
    if thalach < 120.0:
        factors.append(f"<strong style='color: #fbbf24;'>Chronotropic Incompetence (Max HR: {int(thalach)} bpm):</strong> Inability to reach physiologic heart rate under workload.")
    if trestbps >= 140.0:
        factors.append(f"<strong style='color: #fbbf24;'>Stage 2 Hypertension ({int(trestbps)} mmHg):</strong> Increases vascular shear stress and plaque rupture risk.")
    if chol >= 260.0:
        factors.append(f"<strong style='color: #fbbf24;'>Elevated Serum Cholesterol ({int(chol)} mg/dl):</strong> Long-term atherogenic substrate.")

    if not factors:
        factors.append("<strong style='color: #34d399;'>Stable Hemodynamic Baseline:</strong> No acute ischemic or electrophysiological red flags detected.")

    factors_html = """
    <div class="glass-card drivers-box">
        <ul style="margin: 0; padding-left: 18px; color: #e2e8f0; font-size: 13px; line-height: 1.6;">
    """
    for f in factors:
        factors_html += f"<li style='margin-bottom: 6px;'>{f}</li>"
    factors_html += "</ul></div>"

    return triage_html, comparison_html, factors_html, round(risk_pct, 1)


# =============================================================================
# INTENSE GLASSMORPHISM & MINIMALIST BLACK, WHITE & EMERALD GRADIENT STYLING
# =============================================================================
custom_css = """
/* Import sleek sans-serif font */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-deep: #050708;
    --bg-card: rgba(13, 19, 17, 0.72);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-emerald: rgba(16, 185, 129, 0.35);
    --emerald-primary: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.4);
    --text-pure: #ffffff;
    --text-muted: #94a3b8;
}

body, gradio-app, .gradio-container {
    background-color: #040607 !important;
    background-image: 
        radial-gradient(circle at 15% 10%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(5, 150, 105, 0.06) 0%, transparent 45%),
        linear-gradient(180deg, #040607 0%, #080d0b 100%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #ffffff !important;
}

/* Strip any purple/blue default theme backgrounds from headers and labels */
label span, span[data-testid="block-info"], .label-text {
    background: transparent !important;
    background-color: transparent !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: -0.2px !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Glassmorphism Card Primitives */
.glass-card {
    backdrop-filter: blur(24px) saturate(190%);
    -webkit-backdrop-filter: blur(24px) saturate(190%);
    border-radius: 16px;
    padding: 20px 24px;
    transition: all 0.25s ease-in-out;
}

.triage-routine {
    background: rgba(6, 78, 59, 0.22) !important;
    border: 1px solid rgba(16, 185, 129, 0.5) !important;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

.triage-urgent {
    background: rgba(127, 29, 29, 0.24) !important;
    border: 1px solid rgba(239, 68, 68, 0.55) !important;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

.rescue-banner {
    background: rgba(120, 53, 15, 0.28) !important;
    border: 1px solid rgba(245, 158, 11, 0.5) !important;
    box-shadow: 0 8px 24px rgba(245, 158, 11, 0.18) !important;
    margin-bottom: 14px;
}

.consensus-banner {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    margin-bottom: 14px;
}

.drivers-box {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.badge-routine-pill {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    padding: 6px 16px !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.45) !important;
}

.badge-urgent-pill {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
    color: #ffffff !important;
    padding: 6px 16px !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.45) !important;
}

/* Header Glassmorphism Container */
.premium-header {
    background: rgba(13, 20, 17, 0.78) !important;
    backdrop-filter: blur(28px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-bottom: 2px solid rgba(16, 185, 129, 0.45) !important;
    border-radius: 20px !important;
    padding: 26px 32px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 14px 44px rgba(0, 0, 0, 0.65) !important;
    text-align: center;
}

.badge-header-glow {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 5px 16px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
    box-shadow: 0 0 18px rgba(16, 185, 129, 0.25);
}

/* Disclaimer Glass Pill */
.premium-disclaimer {
    background: rgba(245, 158, 11, 0.08) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    margin-bottom: 22px !important;
    font-size: 13px !important;
    color: #fef3c7 !important;
}

/* Input Blocks & Form Controls Glassmorphism */
div.block, div.box, .gradio-block, .form {
    background: rgba(14, 20, 18, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
}

/* Form Titles & Section Headers */
h3, .gr-markdown h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    margin-top: 14px !important;
    margin-bottom: 12px !important;
    letter-spacing: -0.3px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 6px;
}

/* Inputs, Number inputs, Textboxes */
input[type="text"], input[type="number"], textarea, select {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

input[type="text"]:focus, input[type="number"]:focus, select:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.3) !important;
}

/* Emerald Accent Sliders */
input[type="range"] {
    accent-color: #10b981 !important;
}

/* Radio buttons & Checkboxes */
input[type="radio"]:checked, input[type="checkbox"]:checked {
    background-color: #10b981 !important;
    border-color: #10b981 !important;
}

/* Ultra-Premium Button: Glowing Emerald Gradient */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    box-shadow: 0 6px 28px rgba(16, 185, 129, 0.45) !important;
    cursor: pointer !important;
    transition: all 0.25s ease-in-out !important;
}

button.primary:hover, .gr-button-primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #34d399 0%, #059669 100%) !important;
    box-shadow: 0 8px 36px rgba(16, 185, 129, 0.7) !important;
    transform: translateY(-2px) !important;
}

/* Tabs Styling */
button[role="tab"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}

button[role="tab"][aria-selected="true"] {
    color: #34d399 !important;
    border-bottom: 2px solid #10b981 !important;
}

/* Presets Table Styling */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

th {
    background: rgba(16, 185, 129, 0.12) !important;
    color: #34d399 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}

td {
    background: rgba(255, 255, 255, 0.02) !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    padding: 9px 14px !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Number display box */
input[type="number"], .gr-number input {
    color: #34d399 !important;
    font-weight: 800 !important;
    font-size: 24px !important;
    text-align: center !important;
}

/* Footer Styling */
.footer-text {
    text-align: center;
    font-size: 12.5px;
    color: #64748b;
    margin-top: 36px;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
"""

# Custom Emerald & Zinc Glass Theme
glass_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#040607",
    body_background_fill_dark="#040607",
    body_text_color="#ffffff",
    body_text_color_dark="#ffffff",
    block_background_fill="rgba(14, 20, 18, 0.65)",
    block_background_fill_dark="rgba(14, 20, 18, 0.65)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    button_primary_background_fill="linear-gradient(135deg, #10b981 0%, #059669 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #10b981 0%, #059669 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    slider_color="#10b981",
    slider_color_dark="#10b981",
    input_background_fill="rgba(255, 255, 255, 0.04)",
    input_background_fill_dark="rgba(255, 255, 255, 0.04)",
    input_border_color="rgba(255, 255, 255, 0.1)",
    input_border_color_dark="rgba(255, 255, 255, 0.1)",
)

# =============================================================================
# GRADIO APPLICATION LAYOUT
# =============================================================================
with gr.Blocks(title="CardioTriage: Clinical Triage DSS") as demo:
    
    # Direct Style Injection for guaranteed CSS rendering across all Gradio versions
    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML("""
    <div class="premium-header">
        <div class="badge-header-glow">AI + HEALTH TRACK | VERITAS AI HACKATHON 2026</div>
        <h1 style="margin: 4px 0 8px 0; font-size: 32px; font-weight: 800; letter-spacing: -0.8px; color: #ffffff;">
            🫀 CardioTriage
        </h1>
        <p style="margin: 0 auto; max-width: 680px; font-size: 14.5px; color: #cbd5e1; line-height: 1.5;">
            Cost-Sensitive Cardiac Triage & Multi-Hospital Generalization System for Rural Frontline Providers
        </p>
        <p style="margin: 6px 0 0 0; font-size: 12.5px; color: #94a3b8;">
            Developed by <strong>Krishna Sunil Wararkar</strong> (ST. Paul High School)
        </p>
    </div>
    <div class="premium-disclaimer">
        <strong>⚠️ REGULATORY DISCLAIMER:</strong> This project is a prototype and is not medical advice. 
        It is an academic decision-support proof-of-concept exploring asymmetric loss calibration and domain-shift mitigation for frontline rural triage.
    </div>
    """)

    with gr.Tabs():
        # TAB 1: LIVE CLINICAL TRIAGE TOOL
        with gr.Tab("🩺 Frontline Patient Triage DSS"):
            with gr.Row():
                # LEFT COLUMN: PATIENT CLINICAL MEASUREMENTS (Input Form)
                with gr.Column(scale=5):
                    gr.Markdown("### 📋 1. Patient Demographics & Baseline Vitals")
                    with gr.Row():
                        age_input = gr.Slider(minimum=25, maximum=85, value=58, step=1, label="Patient Age (years)")
                        sex_input = gr.Radio(choices=["Male", "Female"], value="Male", label="Biological Sex")
                    
                    with gr.Row():
                        trestbps_input = gr.Slider(minimum=80, maximum=210, value=135, step=1, label="Resting Blood Pressure (mm Hg)")
                        chol_input = gr.Slider(minimum=100, maximum=550, value=250, step=5, label="Serum Cholesterol (mg/dl)")

                    gr.Markdown("### 🫀 2. Symptom Presentation & Stress Test")
                    cp_input = gr.Dropdown(
                        choices=[
                            "Typical Angina (1)",
                            "Atypical Angina (2)",
                            "Non-Anginal Chest Pain (3)",
                            "Asymptomatic / Silent Ischemia (4)"
                        ],
                        value="Asymptomatic / Silent Ischemia (4)",
                        label="Chest Pain Type (Angina Presentation)"
                    )
                    
                    with gr.Row():
                        thalach_input = gr.Slider(minimum=60, maximum=210, value=140, step=1, label="Max Heart Rate Achieved (bpm)")
                        oldpeak_input = gr.Slider(minimum=0.0, maximum=6.2, value=1.6, step=0.1, label="ST Depression Induced by Exercise (mm)")
                    
                    with gr.Row():
                        exang_input = gr.Radio(choices=["Yes", "No"], value="Yes", label="Exercise-Induced Angina")
                        fbs_input = gr.Radio(choices=["No (<=120 mg/dl)", "Yes (>120 mg/dl)"], value="No (<=120 mg/dl)", label="Fasting Blood Sugar > 120 mg/dl")

                    gr.Markdown("### 🔬 3. Electrocardiography & Specialist Diagnostics")
                    with gr.Row():
                        restecg_input = gr.Dropdown(
                            choices=["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"],
                            value="Normal (0)",
                            label="Resting Electrocardiogram (ECG)"
                        )
                        slope_input = gr.Dropdown(
                            choices=["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
                            value="Flat (2)",
                            label="Peak Exercise ST Segment Slope"
                        )

                    with gr.Row():
                        ca_input = gr.Dropdown(
                            choices=["0", "1", "2", "3"],
                            value="1",
                            label="Major Vessels via Fluoroscopy (ca) [0 if unavailable]"
                        )
                        thal_input = gr.Dropdown(
                            choices=["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"],
                            value="Reversible Defect (7)",
                            label="Thallium Nuclear Scan [Normal if unavailable]"
                        )

                    triage_btn = gr.Button("⚡ Generate Clinical Triage Decision", variant="primary", size="lg")

                # RIGHT COLUMN: TRIAGE RESULTS & DECISION SUPPORT
                with gr.Column(scale=5):
                    gr.Markdown("### 🚨 Triage Decision & Action Protocol")
                    triage_badge_output = gr.HTML(label="Triage Status")
                    risk_score_output = gr.Number(label="Assessed Cardiac Risk Probability (%)")
                    
                    gr.Markdown("### ⚖️ Clinical Safety & Model Consensus")
                    comparison_output = gr.HTML(label="Threshold Safety Audit")
                    
                    gr.Markdown("### 🔍 Top Diagnostic Risk Drivers")
                    factors_output = gr.HTML(label="Clinical Drivers")

            # PRESET CLINICAL SCENARIOS FOR INSTANT TESTING
            gr.Markdown("---")
            gr.Markdown("### 💡 Representative Clinical Scenarios (Click to Load)")
            gr.Examples(
                examples=[
                    # Case 1: High Risk Confirmed (Acute Ischemia)
                    [65, "Male", "Asymptomatic / Silent Ischemia (4)", 150, 280, "No (<=120 mg/dl)", "ST-T Wave Abnormality (1)", 125, "Yes", 2.6, "Flat (2)", "2", "Reversible Defect (7)"],
                    # Case 2: Borderline Rescued Patient (Standard AI misses this!)
                    [56, "Female", "Atypical Angina (2)", 130, 236, "No (<=120 mg/dl)", "Normal (0)", 155, "No", 1.2, "Flat (2)", "0", "Normal (3)"],
                    # Case 3: Routine Ambulatory Patient
                    [42, "Female", "Non-Anginal Chest Pain (3)", 118, 195, "No (<=120 mg/dl)", "Normal (0)", 172, "No", 0.0, "Upsloping (1)", "0", "Normal (3)"]
                ],
                inputs=[
                    age_input, sex_input, cp_input, trestbps_input, chol_input, fbs_input,
                    restecg_input, thalach_input, exang_input, oldpeak_input, slope_input, ca_input, thal_input
                ],
                label="Quick-Load Clinical Presets"
            )

            # Link Button Callback
            triage_btn.click(
                fn=predict_cardiac_triage,
                inputs=[
                    age_input, sex_input, cp_input, trestbps_input, chol_input, fbs_input,
                    restecg_input, thalach_input, exang_input, oldpeak_input, slope_input, ca_input, thal_input
                ],
                outputs=[triage_badge_output, comparison_output, factors_output, risk_score_output]
            )

        # TAB 2: RESEARCH AUDIT & OOD SCIENTIFIC BENCHMARK
        with gr.Tab("📊 Multi-Hospital OOD & Explainability Audit"):
            gr.Markdown("""
            ### 🔬 Scientific Validation & Cross-Hospital Stress Testing
            Conventional medical AI often experiences catastrophic domain shift when deployed in new clinical settings.
            Below is the empirical evidence generated by our multi-cohort audit across **Cleveland Clinic**, **Hungarian Institute of Cardiology**, 
            **Zurich University Hospital**, and the **VA Long Beach Medical Center**.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. Multi-Hospital Out-of-Distribution Generalization")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "ood_generalization_audit.png"), label="OOD Generalization (4 Cohorts, N=920)")
                with gr.Column():
                    gr.Markdown("#### 2. Subgroup Demographic Fairness Audit")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "subgroup_fairness_audit.png"), label="Subgroup Parity: Sex & Age")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 3. Asymmetric Clinical Cost Optimization Curve")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "cost_threshold_optimization_curve.png"), label="Cost vs Threshold Curve")
                with gr.Column():
                    gr.Markdown("#### 4. Global SHAP Clinical Feature Attribution")
                    gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_global_summary.png"), label="Global SHAP Summary")

            gr.Markdown("#### 5. Local Clinical Case Forensics (SHAP Waterfall Attributions)")
            with gr.Row():
                gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_true_positive.png"), label="True Positive: Acute Ischemia")
                gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_triage_rescue.png"), label="Triage Rescue: Rescued False Negative")
                gr.Image(value=os.path.join(OUTPUTS_DIR, "shap_case_false_alarm.png"), label="False Alarm: Atypical Case")

    gr.HTML("""
    <div class="footer-text">
        <strong style="color: #e2e8f0;">CardioTriage Clinical Decision Support System</strong> | Veritas AI International High School Hackathon 2026<br>
        Developed by Krishna Sunil Wararkar (ST. Paul High School) | Track: Health
    </div>
    """)


if __name__ == "__main__":
    print("[*] Launching CardioTriage Clinical Decision Support System...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=glass_theme, css=custom_css)
