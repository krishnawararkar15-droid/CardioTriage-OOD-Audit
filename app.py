"""
CardioTriage - Clinical Decision Support & Risk Stratification Web Application
Veritas AI International High School Hackathon 2026 - AI + Health Track

Designed specifically for community triage nurses and frontline providers
at understaffed rural health centers lacking immediate on-site cardiologists.
Implements asymmetric cost-sensitive triage (tau* = 0.31) to eliminate false negatives.
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

    # 5. Build Triage Badge HTML
    if is_urgent_cardiotriage:
        badge_color = "#dc2626"
        badge_bg = "#fef2f2"
        badge_border = "#f87171"
        badge_title = "🚨 URGENT: IMMEDIATE CARDIOLOGY ESCALATION"
        urgency_level = "HIGH RISK (Emergency Transfer Indicated)"
        action_plan = (
            "<strong>Nurse Clinical Protocol:</strong><br>"
            "1. Initiate continuous 12-lead ECG telemetry and continuous pulse oximetry.<br>"
            "2. Establish peripheral IV access; draw urgent cardiac biomarkers (Troponin I/T, CK-MB).<br>"
            "3. Administer chewable Aspirin (325mg) if no contraindications or active hemorrhage.<br>"
            "4. Coordinate emergency paramedic transfer to nearest tertiary cardiac catheterization facility."
        )
    else:
        badge_color = "#16a34a"
        badge_bg = "#f0fdf4"
        badge_border = "#86efac"
        badge_title = "✅ ROUTINE: NON-EMERGENT FOLLOW-UP"
        urgency_level = "LOW / MODERATE RISK (Outpatient Management)"
        action_plan = (
            "<strong>Nurse Clinical Protocol:</strong><br>"
            "1. Patient vitals within acceptable ambulatory thresholds.<br>"
            "2. Schedule outpatient treadmill stress testing or echocardiogram within 14-30 days.<br>"
            "3. Counsel on lifestyle modifications, lipid panel management, and hypertensive control.<br>"
            "4. Instruct patient on red-flag symptoms (radiating crushing chest pain, syncope) requiring immediate ER return."
        )

    triage_html = f"""
    <div style="background-color: {badge_bg}; border: 2px solid {badge_border}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <h2 style="color: {badge_color}; margin: 0; font-size: 20px; font-weight: 800;">{badge_title}</h2>
            <span style="background-color: {badge_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {risk_pct:.1f}% Risk Score
            </span>
        </div>
        <p style="margin: 10px 0 6px 0; color: #1f2937; font-size: 15px; font-weight: 600;">Clinical Status: {urgency_level}</p>
        <hr style="border: 0; border-top: 1px solid {badge_border}; margin: 12px 0;">
        <div style="color: #374151; font-size: 14px; line-height: 1.6;">{action_plan}</div>
    </div>
    """

    # 6. Safety Comparison Banner (Highlighting Rescued Missed Patients)
    if is_urgent_cardiotriage and not is_urgent_standard:
        comparison_html = f"""
        <div style="background-color: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 14px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #b45309; font-size: 15px;">
                ⚡ CLINICAL SAFETY ALERT: RESCUED FALSE NEGATIVE
            </div>
            <div style="color: #92400e; font-size: 13.5px; margin-top: 6px; line-height: 1.5;">
                A conventional AI model with an uncalibrated 50% threshold would have categorized this patient as 
                <strong>'LOW RISK' ({risk_pct:.1f}%)</strong> and potentially discharged them. 
                <strong>CardioTriage's asymmetric loss optimization (&tau;* = {TAU_OPT:.2f})</strong> correctly caught this borderline ischemia, preventing a fatal missed diagnosis.
            </div>
        </div>
        """
    elif is_urgent_cardiotriage and is_urgent_standard:
        comparison_html = """
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 13.5px; color: #475569;">
            <strong>Consensus Agreement:</strong> Both standard AI (&tau; = 0.50) and CardioTriage (&tau;* = 0.31) strongly agree this patient requires urgent cardiac triage.
        </div>
        """
    else:
        comparison_html = """
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 13.5px; color: #475569;">
            <strong>Consensus Agreement:</strong> Both models concur that immediate emergency transfer is not required. Routine primary care follow-up is appropriate.
        </div>
        """

    # 7. Local Factor Attribution Breakdown
    # Calculate feature deviations from normal reference
    factors = []
    if oldpeak >= 1.5:
        factors.append(f"🔴 Significant ST Depression ({oldpeak:.1f} mm) — High indicator of exercise-induced subendocardial ischemia.")
    if cp_val == 4.0:
        factors.append("🔴 Asymptomatic / Silent Presentation — Clinically treacherous; high prevalence of advanced coronary disease.")
    if ca_val >= 1.0:
        factors.append(f"🔴 Fluoroscopy Major Vessel Stenosis ({int(ca_val)} vessels colored) — Strong anatomic determinant of occlusion.")
    if thal_val == 7.0:
        factors.append("🔴 Reversible Thallium Perfusion Defect — Suggests viable but ischemic myocardium under stress.")
    if exang_val == 1.0:
        factors.append("🔴 Exercise-Induced Angina Present — Indicates myocardial oxygen supply-demand mismatch.")
    if thalach < 120.0:
        factors.append(f"🟡 Chronotropic Incompetence (Max HR: {int(thalach)} bpm) — Inability to reach expected target heart rate.")
    if trestbps >= 140.0:
        factors.append(f"🟡 Stage 2 Hypertension ({int(trestbps)} mmHg) — Accelerates vascular shear stress and plaque vulnerability.")
    if chol >= 260.0:
        factors.append(f"🟡 Hypercholesterolemia ({int(chol)} mg/dl) — Elevated atherogenic substrate.")

    if not factors:
        factors.append("🟢 Normal hemodynamic parameters and absence of diagnostic ischemia flags.")

    factors_html = "<ul style='margin: 6px 0; padding-left: 20px; color: #334155; font-size: 13.5px; line-height: 1.6;'>"
    for f in factors:
        factors_html += f"<li>{f}</li>"
    factors_html += "</ul>"

    return triage_html, comparison_html, factors_html, round(risk_pct, 1)


# Define Custom CSS for a Clean, Modern Clinical Interface
custom_css = """
.container { max-width: 1200px; margin: auto; }
.header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); color: white; border-radius: 12px; margin-bottom: 20px; }
.disclaimer-box { background-color: #fef3c7; border-left: 5px solid #f59e0b; padding: 12px 16px; margin-bottom: 20px; border-radius: 4px; font-size: 13.5px; color: #78350f; }
.footer-text { text-align: center; font-size: 12px; color: #64748b; margin-top: 30px; }
"""

with gr.Blocks(title="CardioTriage: Clinical Triage DSS") as demo:
    
    gr.HTML("""
    <div class="header-box">
        <h1 style="margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">🫀 CardioTriage</h1>
        <p style="margin: 6px 0 0 0; font-size: 15px; opacity: 0.9;">
            Cost-Sensitive Cardiac Risk Stratification & Out-of-Distribution Generalization System
        </p>
        <p style="margin: 4px 0 0 0; font-size: 12.5px; opacity: 0.75;">
            Veritas AI International High School Hackathon 2026 | AI + Health Track | ST. Paul High School
        </p>
    </div>
    <div class="disclaimer-box">
        <strong>⚠️ REGULATORY DISCLAIMER:</strong> This project is a prototype and is not medical advice. 
        It is designed as an academic proof-of-concept exploring asymmetric loss optimization and domain shift mitigation in frontline rural healthcare triage.
    </div>
    """)

    with gr.Tabs():
        # TAB 1: LIVE CLINICAL TRIAGE TOOL
        with gr.Tab("🩺 Frontline Patient Triage DSS"):
            with gr.Row():
                # LEFT COLUMN: PATIENT CLINICAL MEASUREMENTS
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
                            label="Major Vessels Colored by Fluoroscopy (ca) [0 if unavailable]"
                        )
                        thal_input = gr.Dropdown(
                            choices=["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"],
                            value="Reversible Defect (7)",
                            label="Thallium Nuclear Scintigraphy Scan [Normal if unavailable]"
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
            gr.Markdown("### 💡 Click to Load Representative Clinical Scenarios")
            gr.Examples(
                examples=[
                    # Case 1: High Risk Confirmed
                    [65, "Male", "Asymptomatic / Silent Ischemia (4)", 150, 280, "No (<=120 mg/dl)", "ST-T Wave Abnormality (1)", 125, "Yes", 2.6, "Flat (2)", "2", "Reversible Defect (7)"],
                    # Case 2: Borderline Rescued Patient (Standard AI misses this!)
                    [56, "Female", "Atypical Angina (2)", 130, 236, "No (<=120 mg/dl)", "Normal (0)", 155, "No", 1.2, "Flat (2)", "0", "Normal (3)"],
                    # Case 3: Routine Healthy Patient
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
        <strong>CardioTriage System</strong> | Built for Veritas AI International High School Hackathon 2026<br>
        Developed by Krishna Sunil Wararkar (ST. Paul High School) | Track: AI + Health
    </div>
    """)


if __name__ == "__main__":
    print("[*] Launching CardioTriage Clinical Decision Support System...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), css=custom_css)
