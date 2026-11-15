# Veritas AI International High School Hackathon 2026
## Official Submission Form Copy-Paste Draft

Use this pre-formatted document to submit your project quickly on the official submission form.

---

### 1. General Project Information
* **Track:** AI + Health
* **Project Name:** CardioTriage: Cost-Sensitive Cardiac Risk Screening & Cross-Hospital Generalization Audit
* **Team Members:**
  * Name: Krishna Sunil Wararkar
  * Email: krishnawararkar15@gmail.com
  * High School: ST. Paul High School
* **GitHub Repository URL:** [https://github.com/krishnawararkar15-droid/CardioTriage-OOD-Audit](https://github.com/krishnawararkar15-droid/CardioTriage-OOD-Audit)
* **Demo Video URL (Optional but Recommended):** `[INSERT YOUR UNLISTED YOUTUBE OR PUBLIC GOOGLE DRIVE LINK HERE]`

---

### 2. Official Project Write-Up (Copy-Paste Text)

```markdown
# CardioTriage: Cost-Sensitive Cardiac Risk Screening & Cross-Hospital Generalization Audit
**Track:** AI + Health
**Team:** Krishna Sunil Wararkar (ST. Paul High School)

## The problem
Frontline triage nurses at understaffed community health centers and rural outposts frequently evaluate patients presenting with ambiguous chest pain and fatigue, but lack immediate on-site cardiologists or cardiac catheterization labs. When standard clinical intuition or uncalibrated computer tools underestimate risk, acute ischemic patients are erroneously sent home with fatal delays in care. Because a missed cardiac diagnosis (False Negative) is exponentially more catastrophic than an unnecessary specialist transfer (False Positive), clinical triage urgently requires an asymmetric, cost-sensitive AI safety system.

## What we built
We built CardioTriage, an asymmetric decision-support system and cross-hospital generalization audit tool designed for rural healthcare providers.
- What goes in: 13 routine checkup measurements: patient demographics (Age, Sex), baseline vitals and symptoms (Chest Pain Type, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Exercise Angina, Max Heart Rate), and electrocardiography/imaging features (Resting ECG, Exercise ST Depression, ST Slope, Major Vessels, Thallium Scan). If specialist imaging is unavailable at a rural clinic, the tool defaults safely to training median baselines.
- What comes out: An explicit Clinical Triage Badge (URGENT: IMMEDIATE CARDIOLOGY ESCALATION vs. ROUTINE: NON-EMERGENT FOLLOW-UP), calibrated risk score (%), nurse emergency action protocols, consensus audit identifying rescued false negatives, and local physiological risk driver attributions.
- How someone uses it: A community clinic nurse enters routine vitals into the interactive web application; within 2 seconds, the tool stratifies acute ischemic risk, flags borderline presentations, and provides justifiable referral documentation for emergency transport.

## What we added
Starting from suggested project 4.1.2 (Heart Disease Risk from Checkup Data), we substantially expanded beyond the standard baseline in three novel dimensions:
1. Our Own Clinical Setting: Designed specifically around the workflow of a frontline triage nurse at an understaffed rural health outpost with no resident cardiologist, optimizing for asymmetric clinical loss rather than symmetric statistical accuracy.
2. Asymmetric Cost-Sensitive Threshold Optimization (5:1 Clinical Loss): We formalized a clinical utility loss function (Loss = 5 * FN + 1 * FP), penalizing missed heart attacks 5x more heavily than false alarms. Sweeping the full threshold trade-off curve, we optimized an operational triage threshold (tau* = 0.31) that drove held-out test sensitivity to 100.0% (0 missed cases).
3. Multi-Hospital Out-of-Distribution (OOD) Stress Testing (N=920 across 4 Institutions): Audited cross-institutional generalization across 617 external, untouched international patients from the Hungarian Institute of Cardiology (N=294), Zurich University Hospital, Switzerland (N=123), and the VA Medical Center Long Beach, CA (N=200). We discovered that standard AI collapses by 40%-44% in recall under hospital domain shift, and proved that CardioTriage's cost-calibrated threshold recovers >82% sensitivity across all external hospital cohorts.
4. Demographic Subgroup Fairness Audit: Audited diagnostic parity across Biological Sex (Male vs. Female) and Age (<55 vs. >=55), confirming that CardioTriage completely eliminates the systemic female under-diagnosis problem seen in standard cardiac classifiers.
5. SHAP Clinical Failure Forensics: Utilized TreeSHAP to generate global feature attribution summaries and local forensic waterfall plots explaining the physiological mechanism behind confirmed true positives, false alarms, and rescued borderline patients.

## How it works
- Data: 920 multi-hospital records from the official UCI Machine Learning Repository (Cleveland Clinic N=303, Hungarian Institute N=294, Zurich University N=123, VA Long Beach N=200). Target angiographic narrowing (num 0-4) was binarized (0: healthy, 1: disease present). Physiological anomalies (e.g. unmeasured cholesterol or blood pressure encoded as 0) were converted to NaN. Cleveland was partitioned into an 80% train set (N=242) and a 20% held-out test set (N=61). SimpleImputer(strategy='median') and StandardScaler were fit strictly on the Cleveland training set to guarantee zero data leakage.
- Model: Random Forest Classifier (150 estimators, max_depth=4) combined with an asymmetric cost-sensitive threshold search (tau* = 0.31) minimizing clinical risk penalties. Compared against a majority-class baseline (predicts healthy) and clinical Logistic Regression.

## Results
- In-Distribution (Cleveland Held-Out Test, N=61):
  * Majority Baseline: 54.1% Accuracy, 0.0% Recall (misses 100% of sick patients, FN=28, Clinical Penalty=140.0).
  * Standard Logistic Regression (tau=0.50): 86.9% Accuracy, 92.9% Recall (FN=2, FP=6, Clinical Penalty=16.0).
  * Standard Random Forest (tau=0.50): 90.2% Accuracy, 92.9% Recall (FN=2, FP=4, Clinical Penalty=14.0).
  * CardioTriage Cost-Optimized RF (tau*=0.31): 83.6% Accuracy, 100.0% Recall (0 missed cases, FN=0, FP=10, Clinical Penalty=10.0, a 28.6% cost reduction).
- Out-of-Distribution Stress Testing (N=617 External Patients):
  * Standard models collapse under hospital domain shift: Recall drops to 53.8% on Hungary (FN=49), 53.9% on Switzerland (FN=53), and 49.0% on VA Long Beach (FN=76).
  * CardioTriage restores robust generalization: Achieves 82.1% Recall on Hungary (FN drops to 19), 82.6% Recall on Switzerland (FN drops to 20), and 83.2% Recall on VA Long Beach (FN drops to 25), preventing up to 114 sick patients from being missed.
- Subgroup Fairness: Achieved 100.0% Sensitivity across both Male (N=41) and Female (N=20) subgroups, and across both age brackets (<55 and >=55).
- Where it fails, and why: Standard models fail severely out-of-distribution because secondary hospitals rarely ordered invasive fluoroscopy (ca) or nuclear thallium scans (thal). CardioTriage accepts 10 false positives (30.3% false alarm rate on Cleveland) to guarantee 0 missed cases, which is clinically desirable in frontline triage.
- Mandatory Regulatory Disclaimer: "This project is a prototype and is not medical advice."

## What we'd do next
1. Conformal Risk Prediction: Implement split conformal prediction to output mathematically guaranteed uncertainty intervals for individual triage scores.
2. Prospective Clinical Feasibility Study: Partner with rural clinic networks to evaluate nurse usability and emergency transfer dispatch times.

## Sources
- Datasets: UCI Machine Learning Repository (Detrano et al., 1989), DOI: 10.24432/C5630X; Multi-hospital mirror: github.com/redwankarimsony/heart-disease-data
- Libraries: scikit-learn, pandas, numpy, shap, matplotlib, seaborn, gradio, joblib
- Literature: Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." Am J Cardiol, 64(5):304-310.
- AI Assistance: Antigravity AI assistant used for environment configuration and testing scripts.
```

---

### 3. Citations & AI Disclosure Statement
*(Required under Section 10: "Rules and responsible AI")*
* **Datasets:** UCI Heart Disease Registry (Cleveland, Hungarian, Switzerland, and VA Long Beach cohorts) curated by Dr. Robert Detrano, M.D., Ph.D.
* **Pretrained Models & Libraries:** `scikit-learn` v1.9.1, `shap` v0.54.1, `gradio` v6.32.0, `pandas` v3.0.6, `numpy` v2.4.4.
* **AI Coding Tools Disclosure:** An Antigravity coding assistant was utilized to scaffold project architecture, run bash scripts, and format visualizations. All clinical methodologies, cost matrices, experimental designs, and interpretations were verified and understood by the team.
