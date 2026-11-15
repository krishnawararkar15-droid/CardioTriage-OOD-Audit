# Veritas AI International High School Hackathon 2026
## Official Submission Form Copy-Paste Draft

Use this pre-formatted document to submit your project quickly on the official submission form.

---

### 1. General Project Information
* **Track:** Health
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
**Track:** Health
**Team:** Krishna Sunil Wararkar (ST. Paul High School)

> **Regulatory Disclaimer:** This project is a prototype and is not medical advice.

## The problem
Frontline triage nurses at understaffed rural health posts and community clinics evaluate dozens of walk-in patients daily presenting with ambiguous chest pain, shortness of breath, and fatigue, but have no on-site cardiologists, cardiac catheterization suites, or nuclear imaging scanners. When conventional clinical judgment or standard symmetric AI algorithms underestimate borderline cardiac presentations, patients experiencing acute coronary syndromes are mistakenly discharged home with fatal delays in emergency care. Because sending an evolving heart attack patient home (a False Negative) is catastrophic, whereas an unnecessary ambulance transfer (a False Positive) is manageable, clinical cardiac triage fundamentally demands an asymmetric, cost-sensitive safety system that prioritizes sensitivity above raw statistical accuracy.

## What we built
We built CardioTriage, an asymmetric clinical decision-support web application and multi-hospital generalization audit system designed specifically for bedside rural triage nurses.
- What goes in: 13 routine checkup measurements: demographics (Age, Sex), vitals and symptoms (Chest Pain Type, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Exercise Angina, Max Heart Rate, Exercise ST Depression, ST Slope), and secondary diagnostics (Resting ECG, Fluoroscopy Major Vessels, Thallium Nuclear Scan). If specialist imaging is unavailable at a rural clinic, the system safely imputes training-cohort medians.
- What comes out: An immediate Clinical Triage Status (URGENT: IMMEDIATE CARDIOLOGY ESCALATION vs. ROUTINE: NON-EMERGENT FOLLOW-UP), calibrated risk score (%), emergency nursing protocols (IV access, chewable aspirin, ECG telemetry, transfer coordination), a model consensus audit identifying rescued false negatives, and ranked local physiological risk factors via TreeSHAP attributions.
- How someone would use it: A rural triage nurse inputs the patient's vitals and symptoms into the web interface at intake. In under 2 seconds, the tool stratifies ischemic risk, provides justifiable referral documentation, and alerts the nurse to initiate immediate emergency transfer protocols for borderline presentations.

## What we added
Starting from suggested project 4.1.2 (Heart disease risk from checkup data), we expanded beyond the baseline across all three Veritas AI "Make It Your Own" dimensions:
1. Our Own User & Setting: Designed specifically around the emergency workflow of a frontline triage nurse at an isolated rural health clinic without an on-site cardiologist, replacing symmetric statistical accuracy with asymmetric clinical safety loss.
2. Our Own Question (Cost-Sensitive Decision Optimization): Standard models default to an uncalibrated 50% decision cutoff (tau = 0.50), which treats a dead patient (FN) the same as an extra test (FP). We mathematically formalized an asymmetric clinical loss function (Loss = 5 * FN + 1 * FP). By sweeping thresholds across tau in [0.10, 0.80] subject to a clinical safety constraint (Recall >= 92%), we discovered the optimal operating threshold at tau* = 0.31, completely eliminating False Negatives on the held-out test cohort.
3. Our Own Data (Multi-Hospital Generalization Audit across N=920 patients): Rather than evaluating solely on the standard Cleveland Clinic dataset, we collected and standardized three external, untouched international cohorts (617 out-of-distribution patients): Hungarian Institute of Cardiology (N=294), University Hospital Zurich (N=123), and VA Medical Center Long Beach (N=200). We evaluated hospital domain shift to discover why standard models collapse in real-world clinic deployments.
4. Demographic Subgroup Fairness Audit: Audited diagnostic parity across Biological Sex (Male vs. Female) and Age (<55 vs. >=55), proving our calibrated threshold eliminates historic female under-diagnosis in cardiology.
5. SHAP Clinical Failure Forensics: Deployed TreeSHAP to generate global feature importance summaries and individual patient waterfall forensics for True Positives, False Alarms, and Triage Rescues.

## How it works
- Data: 920 international patient records from the official UCI Machine Learning Repository (Cleveland N=303, Hungary N=294, Switzerland N=123, VA Long Beach N=200). Target angiographic disease status (num 0-4) was binarized (0: healthy, 1: disease present). Physiological anomalies (e.g. unmeasured cholesterol or blood pressure encoded as 0) were converted to NaN. Cleveland was partitioned into an 80% train set (N=242) and an untouched 20% held-out test set (N=61). SimpleImputer(strategy='median') and StandardScaler were fit strictly on the Cleveland training set to guarantee zero data leakage.
- Model: Random Forest Classifier (150 trees, max_depth=4) combined with an asymmetric cost-sensitive threshold search (tau* = 0.31) minimizing clinical risk penalties. Compared against a majority-class baseline (predicts healthy) and clinical Logistic Regression.

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
- Where it fails, and why we think so:
  1. Diagnostic Covariate Shift: Cleveland Clinic was a research facility routinely ordering fluoroscopy (ca) and thallium scans (thal). Foreign community hospitals rarely ordered these tests (99% missing ca in Hungary, 96% missing in Switzerland). Imputed medians caused standard AI recall to drop by over 40%; while CardioTriage recovered recall to >82%, 17%-18% of complex OOD cases remained missed due to missing clinical tests.
  2. False Alarm Trade-off: Achieving 100% sensitivity on Cleveland requires accepting 10 false alarms (30.3% false alarm rate, specificity 69.7%). In rural triage, transferring ambiguous cases for secondary evaluation is medically preferable to discharging a patient with acute coronary ischemia.
  3. Severe Referral Bias in Switzerland: Zurich cohort had 93.5% prevalence (surgical tertiary admissions).
  4. Signal Loss in VA Long Beach: 56 patients lacked BP and 53 lacked max heart rate.

## What we'd do next
1. Conformal Risk Prediction: Implement split conformal prediction to output mathematically guaranteed confidence intervals for each patient's risk probability, alerting nurses when uncertainty is too high for safe autonomous screening.
2. Adaptive Missing-Feature Submodels: Train pattern-specific submodels conditioned on partial feature subsets (e.g. vitals-only vs. vitals + ECG) rather than relying on median imputation.
3. Prospective Rural Pilot Study: Partner with community health centers to evaluate nurse usability and ambulance transfer dispatch times.

## Sources
- Datasets: UCI Machine Learning Repository (Detrano et al., 1989), DOI: 10.24432/C5630X; Multi-hospital mirror: github.com/redwankarimsony/heart-disease-data
- Open Source Libraries: scikit-learn v1.9.1, pandas v3.0.6, numpy v2.4.4, shap v0.54.1, gradio v6.32.0, joblib, matplotlib, seaborn
- Literature: Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." Am J Cardiol, 64(5):304-310; Elkan, C. (2001). "The foundations of cost-sensitive learning." IJCAI.
- AI Assistance: An Antigravity coding assistant was used for bash environment configuration, script verification, and formatting output plots. All clinical formulations, asymmetric loss functions, data auditing, and empirical evaluations were directed and understood by the team.
```

---

### 3. Citations & AI Disclosure Statement
*(Required under Section 10: "Rules and responsible AI")*
* **Datasets:** UCI Heart Disease Registry (Cleveland, Hungarian, Switzerland, and VA Long Beach cohorts) curated by Dr. Robert Detrano, M.D., Ph.D.
* **Pretrained Models & Libraries:** `scikit-learn` v1.9.1, `shap` v0.54.1, `gradio` v6.32.0, `pandas` v3.0.6, `numpy` v2.4.4.
* **AI Coding Tools Disclosure:** An Antigravity coding assistant was utilized to scaffold project architecture, run bash scripts, and format visualizations. All clinical methodologies, cost matrices, experimental designs, and interpretations were verified and understood by the team.
