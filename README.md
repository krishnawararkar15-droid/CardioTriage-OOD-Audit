# CardioTriage: Cost-Sensitive Cardiac Risk Screening & Cross-Hospital Generalization Audit
**Track:** Health  
**Team:** Krishna Sunil Wararkar (ST. Paul High School)  

> **Regulatory Disclaimer:** This project is a prototype and is not medical advice.

---

## The problem
Frontline triage nurses at understaffed rural health posts and community clinics evaluate dozens of walk-in patients daily presenting with ambiguous chest pain, shortness of breath, and fatigue, but have no on-site cardiologists, cardiac catheterization suites, or nuclear imaging scanners. When conventional clinical judgment or standard symmetric AI algorithms underestimate borderline cardiac presentations, patients experiencing acute coronary syndromes are mistakenly discharged home with fatal delays in emergency care. Because sending an evolving heart attack patient home (a False Negative) is catastrophic, whereas an unnecessary ambulance transfer (a False Positive) is manageable, clinical cardiac triage fundamentally demands an asymmetric, cost-sensitive safety system that prioritizes sensitivity above raw statistical accuracy.

---

## What we built
We built **CardioTriage**, an asymmetric clinical decision-support web application and multi-hospital generalization audit system designed specifically for bedside rural triage nurses.

* **What goes in:** 13 routine checkup measurements readily obtainable during intake:
  * *Demographics & History:* Age (years), Biological Sex (Male/Female), Chest Pain Type (1: Typical Angina, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic/Silent Ischemia), Fasting Blood Sugar (>120 mg/dl), Exercise-Induced Angina (Yes/No).
  * *Vitals & Exercise Stress:* Resting Blood Pressure (mm Hg), Serum Cholesterol (mg/dl), Maximum Heart Rate Achieved (bpm), ST Depression induced by exercise (`oldpeak` in mm), ST Slope (Upsloping, Flat, Downsloping).
  * *Secondary Diagnostics (optional):* Resting ECG (Normal, ST-T Abnormality, LVH), Fluoroscopy Major Vessels (`ca`: 0–3), and Thallium Nuclear Scan (`thal`: Normal, Fixed Defect, Reversible Defect). If specialist imaging is unavailable at a rural clinic, the system safely imputes training-cohort medians.
* **What comes out:**
  1. **Immediate Clinical Triage Status:** An unambiguous visual badge: `🚨 URGENT: IMMEDIATE CARDIOLOGY ESCALATION` versus `✅ ROUTINE: NON-EMERGENT FOLLOW-UP`.
  2. **Calibrated Cardiac Probability:** Calibrated patient disease risk probability score (`%`).
  3. **Actionable Nursing Protocol:** Immediate emergency protocols (IV access, chewable aspirin, oxygen saturation monitoring, continuous 12-lead ECG telemetry, and regional transfer coordination).
  4. **Model Consensus Audit:** A side-by-side comparison showing whether standard symmetric AI would have dangerously missed the patient as a False Negative and how CardioTriage rescues them.
  5. **Explainable Diagnostic Risk Drivers:** Ranked local clinical factors driving the specific patient's risk score based on TreeSHAP attributions.
* **How someone would use it:** A rural triage nurse inputs the patient's baseline measurements and symptoms into the web interface at intake. In under 2 seconds, the tool stratifies ischemic risk, provides justifiable referral documentation, and alerts the nurse to initiate immediate emergency transfer protocols for borderline presentations.

---

## What we added
Starting from suggested project **4.1.2 Heart disease risk from checkup data**, we expanded beyond the baseline across all three Veritas AI "Make It Your Own" dimensions to produce a rigorous, research-grade audit:

1. **Our Own User & Setting:** Designed specifically around the emergency workflow of a frontline triage nurse at an isolated rural health clinic without an on-site cardiologist, replacing symmetric statistical accuracy with asymmetric clinical safety loss.
2. **Our Own Question (Cost-Sensitive Decision Optimization):** Standard models default to an uncalibrated $50\%$ decision cutoff ($\tau = 0.50$), which treats a dead patient ($FN$) the same as an extra test ($FP$). We mathematically formalized an asymmetric clinical loss function:
   $$\text{Clinical Loss} = 5 \times \text{FN} + 1 \times \text{FP}$$
   By sweeping thresholds across $\tau \in [0.10, 0.80]$ subject to a clinical safety constraint ($\text{Recall} \ge 92\%$), we discovered the optimal operating threshold at **$\tau^* = 0.31$**, completely eliminating False Negatives on the held-out test cohort.
3. **Our Own Data (Multi-Hospital Generalization Audit across $N=920$ patients):** Rather than evaluating solely on the standard Cleveland Clinic dataset, we collected and standardized three external, untouched international cohorts ($617$ out-of-distribution patients):
   * **Hungarian Institute of Cardiology (Budapest):** $N = 294$
   * **University Hospital, Zurich (Switzerland):** $N = 123$
   * **VA Medical Center (Long Beach, California):** $N = 200$
   We evaluated hospital domain shift to discover why standard models collapse in real-world clinic deployments.
4. **Demographic Subgroup Fairness Audit:** Audited diagnostic parity across Biological Sex (Male vs. Female) and Age ($<55$ vs. $\ge 55$), proving our calibrated threshold eliminates historic female under-diagnosis in cardiology.
5. **SHAP Clinical Failure Forensics:** Deployed TreeSHAP to generate global feature importance summaries and individual patient waterfall forensics for True Positives, False Alarms, and Triage Rescues.

---

## How it works

### Data
We ingested four international hospital registries from the official UCI Machine Learning Repository ($920$ total patient records):
* **Cleveland Clinic Foundation (USA):** $303$ patients (prevalence: $45.87\%$ diseased)
* **Hungarian Institute of Cardiology (Budapest):** $294$ patients (prevalence: $36.05\%$ diseased)
* **University Hospital Zurich (Switzerland):** $123$ patients (prevalence: $93.50\%$ diseased)
* **VA Medical Center Long Beach (USA):** $200$ patients (prevalence: $74.50\%$ diseased)

* **Target Binarization:** Angiographic disease status `num` ($0 = <50\%$ diameter stenosis, $1\text{--}4 = >50\%$ stenosis) was converted to binary target $y \in \{0, 1\}$.
* **Physiological Sanity Enforcement:** Clinical recording artifacts (resting BP of $0\text{ mm Hg}$ or serum cholesterol of $0\text{ mg/dl}$, which are physiologically incompatible with living patients) were detected and mapped to `NaN`.
* **Zero-Leakage Partitioning:** Cleveland data was split into an $80\%$ Training Set ($N=242$, $45.87\%$ prevalence) and an untouched $20\%$ Held-Out Test Set ($N=61$, $45.90\%$ prevalence) stratified by disease. `SimpleImputer(strategy='median')` and `StandardScaler` were **fit exclusively on the Cleveland training set**; learned statistics were frozen and used to transform the test set and external hospital cohorts. Zero test data was leaked during training.

### Model
* **Baseline 1 (Majority Classifier):** Predicts the most frequent class (healthy, $y=0$) for every sample.
* **Baseline 2 (Clinical Logistic Regression):** L2-regularized linear classifier ($\tau = 0.50$) on standardized features with odds-ratio analysis.
* **Baseline 3 (Standard Random Forest):** 150-tree ensemble (`max_depth=4`, `min_samples_leaf=2`) operating at the standard symmetric threshold ($\tau = 0.50$).
* **Proposed System (CardioTriage Cost-Optimized RF):** Evaluates risk probabilities using our asymmetric clinical loss formula ($\text{Loss} = 5\times\text{FN} + 1\times\text{FP}$). The optimal cutoff was identified at **$\tau^* = 0.31$**.

---

## Results

### 1. In-Distribution Held-Out Benchmark (Cleveland Test Set, $N=61$)

| Model Pipeline | Threshold ($\tau$) | Accuracy | Sensitivity (Recall) | Specificity | Precision | Missed Cases (FN) | False Alarms (FP) | Clinical Loss ($5\times\text{FN} + \text{FP}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | $0.50$ | $54.1\%$ | $0.0\%$ | $100.0\%$ | $0.0\%$ | $28$ | $0$ | $140.0$ |
| **Standard Logistic Regression** | $0.50$ | $86.9\%$ | $92.9\%$ | $81.8\%$ | $81.2\%$ | $2$ | $6$ | $16.0$ |
| **Standard Random Forest** | $0.50$ | $90.2\%$ | $92.9\%$ | $87.9\%$ | $86.7\%$ | $2$ | $4$ | $14.0$ |
| **CardioTriage (Cost-Optimized)** | $\mathbf{0.31}$ | **$83.6\%$** | $\mathbf{100.0\%}$ | **$69.7\%$** | **$73.7\%$** | $\mathbf{0}$ | **$10$** | $\mathbf{10.0}$ |

* **Beating the Baseline:** The majority baseline claims $54.1\%$ accuracy, but misses **$100\%$ of sick patients** ($FN = 28$, clinical loss $140.0$).
* **Clinical Safety Win:** Conventional Random Forest ($\tau = 0.50$) achieves $90.2\%$ accuracy, but misses $2$ active heart disease cases ($FN = 2$). CardioTriage ($\tau^* = 0.31$) slashes missed cases to **$0$ ($100.0\%$ Sensitivity)** and cuts total clinical risk penalty from $14.0$ to $10.0$ (**$28.6\%$ cost reduction**).

---

### 2. Multi-Hospital Out-of-Distribution (OOD) Stress Test ($N=617$ External Patients)

![OOD Generalization](outputs/ood_generalization_audit.png)

| Hospital Cohort | Domain | Sample Size ($N$) | Std RF ($\tau=0.50$) Recall | CardioTriage ($\tau^*=0.31$) Recall | Missed Cases Drop ($\text{FN}_{\text{std}} \to \text{FN}_{\text{opt}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cleveland (In-Dist Test)** | In-Dist | $N=61$ | $92.9\%$ | $\mathbf{100.0\%}$ | $2 \to \mathbf{0}$ ($100\%$ reduction) |
| **Hungary (Budapest)** | External OOD | $N=294$ | $53.8\%$ | $\mathbf{82.1\%}$ | $49 \to \mathbf{19}$ ($61.2\%$ reduction) |
| **Switzerland (Zurich)** | External OOD | $N=123$ | $53.9\%$ | $\mathbf{82.6\%}$ | $53 \to \mathbf{20}$ ($62.3\%$ reduction) |
| **VA Long Beach (CA)** | External OOD | $N=200$ | $49.0\%$ | $\mathbf{83.2\%}$ | $76 \to \mathbf{25}$ ($67.1\%$ reduction) |

---

### 3. Subgroup Demographic Fairness Audit

![Subgroup Fairness](outputs/subgroup_fairness_audit.png)

* **Sex Parity:** CardioTriage achieved **$100.0\%$ Sensitivity across both Males ($N=41$) and Females ($N=20$)** (Overall accuracy: $78.0\%$ in males, $95.0\%$ in females). This directly fixes the documented clinical failure mode where women presenting with atypical angina are misdiagnosed and discharged by symmetric algorithms.
* **Age Parity:** Achieved **$100.0\%$ Sensitivity across both age brackets** ($<55$ years, $N=28$; and $\ge 55$ years, $N=33$).

---

### 4. SHAP Feature Attribution & Local Clinical Forensics

![SHAP Summary](outputs/shap_global_summary.png)

* **Global Diagnostic Drivers:** TreeSHAP identified fluoroscopy vessel calcification (`ca`), exercise-induced ST depression (`oldpeak`), maximum heart rate achieved (`thalach`), and asymptomatic presentation (`cp`) as the primary predictors of cardiac risk.
* **Forensic Case Studies:**
  * **True Positive ([shap_case_true_positive.png](outputs/shap_case_true_positive.png)):** Confirmed acute patient with severe exercise ST depression ($2.6\text{ mm}$) and fluoroscopy stenosis correctly triaged with $91.3\%$ risk.
  * **False Alarm ([shap_case_false_alarm.png](outputs/shap_case_false_alarm.png)):** Patient with atypical chest pain and elevated cholesterol driving an escalated referral despite absence of vessel stenosis.
  * **Triage Rescue ([shap_case_triage_rescue.png](outputs/shap_case_triage_rescue.png)):** Borderline patient with predicted risk of $42.5\%$. Under standard AI ($\tau=0.50$), this patient is misdiagnosed as healthy ($FN$); under CardioTriage ($\tau^*=0.31$), this patient is successfully caught and escalated.

---

### Where it fails, and why we think so (Raw, Transparent Failure Analysis)

To adhere strictly to the hackathon's honesty criterion, we transparently document our model's four empirical limitations:

1. **The 40% OOD Sensitivity Collapse (Diagnostic Covariate Shift):**
   When the Cleveland-trained model was deployed on foreign clinics, standard AI recall plummeted from $92.9\%$ down to $49.0\%–53.9\%$. Our missingness audit revealed why: **Cleveland Clinic was a tertiary research facility that routinely ordered invasive fluoroscopy (`ca`) and radioactive thallium scans (`thal`)**. In contrast, community hospitals in Hungary and Switzerland almost never ordered these expensive tests ($99\%$ missing `ca` in Hungary, $96\%$ missing in Switzerland). When our model had to infer risk using imputed medians for these key imaging features, confidence collapsed. While CardioTriage's cost-calibrated threshold recovered recall to $>82\%$ (rescuing $114$ patients), $17\%–18\%$ of complex OOD cases were still missed due to unmeasured diagnostic data.
2. **The False Alarm Trade-Off (10 False Positives on Cleveland Test):**
   Achieving $100.0\%$ sensitivity required lowering accuracy from $90.2\%$ down to $83.6\%$, accepting $10$ false alarms ($30.3\%$ false alarm rate, specificity $69.7\%$). In a resource-constrained rural health post, transferring $10$ non-diseased patients incurs ambulance fuel costs and patient anxiety. However, from a medical risk-management perspective, transferring an ambiguous patient for an elective echocardiogram is safe, whereas discharging a patient with an evolving myocardial infarction is fatal.
3. **Severe Referral Bias in Zurich, Switzerland:**
   The Switzerland cohort exhibited an extreme disease prevalence of $93.50\%$ ($115$ out of $123$ patients had confirmed coronary disease), representing high-acuity surgical admissions rather than walk-in triage presentations. Our model missed $20$ patients in this cohort because several presented with normal resting ECGs and atypical symptoms that diverged from typical outpatient patterns.
4. **Physiological Signal Loss in VA Long Beach:**
   In the VA Long Beach cohort, $56$ patients lacked resting blood pressure measurements and $53$ lacked maximum heart rate data. While median imputation allows automated processing, statistical imputation cannot replace live hemodynamic monitoring.

---

## What we'd do next
1. **Conformal Risk Prediction:** Implement split conformal prediction to output mathematically guaranteed confidence intervals for each patient's risk probability, explicitly alerting nurses when prediction uncertainty is too wide for safe autonomous screening.
2. **Adaptive Missing-Feature Training (Pattern Submodels):** Train dedicated ensemble submodels conditioned on partial feature subsets (e.g., vitals-only vs. vitals + ECG) so that rural clinics lacking specialist imaging are evaluated by models trained without those features, rather than relying on median imputation.
3. **Prospective Rural Pilot Study:** Partner with a network of community health workers to evaluate usability, measuring whether CardioTriage reduces specialist transfer dispatch times.

---

## Sources
* **Datasets:**
  * UCI Heart Disease Repository (Detrano et al., 1989): [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (DOI: 10.24432/C5630X)
  * Multi-Hospital Consolidated Mirror: [https://github.com/redwankarimsony/heart-disease-data](https://github.com/redwankarimsony/heart-disease-data)
* **Open Source Libraries:** `scikit-learn` v1.9.1, `pandas` v3.0.6, `numpy` v2.4.4, `shap` v0.54.1, `gradio` v6.32.0, `joblib`, `matplotlib`, `seaborn`.
* **Foundational Literature:**
  * Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* The American Journal of Cardiology, 64(5), 304-310.
  * Elkan, C. (2001). *The foundations of cost-sensitive learning.* International Joint Conference on Artificial Intelligence (IJCAI).
* **AI Tool Usage Disclosure:** An Antigravity coding assistant was used for bash environment configuration, script verification, and formatting output plots. All clinical formulations, asymmetric loss functions, data auditing, and empirical evaluations were directed and understood by the team.
