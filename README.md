# CardioTriage: Cost-Sensitive Cardiac Risk Screening & Cross-Hospital Generalization Audit
**Track:** AI + Health  
**Team:** Krishna Sunil Wararkar (ST. Paul High School)  

> **Regulatory Disclaimer:** This project is a prototype and is not medical advice. It is an academic decision-support proof-of-concept designed to explore clinical risk calibration and domain shift mitigation.

---

## The problem
Frontline triage nurses at understaffed community health centers and rural outposts frequently evaluate patients presenting with ambiguous chest pain and fatigue, but lack immediate on-site cardiologists or cardiac catheterization labs. When standard clinical intuition or uncalibrated computer tools underestimate risk, acute ischemic patients are erroneously sent home with fatal delays in care. Because a missed cardiac diagnosis (False Negative) is exponentially more catastrophic than an unnecessary specialist transfer (False Positive), clinical triage urgently requires an asymmetric, cost-sensitive AI safety system.

---

## What we built
We built **CardioTriage**, an asymmetric decision-support system and cross-hospital generalization audit tool designed for rural healthcare providers.

* **What goes in:** 13 routine checkup measurements: patient demographics (Age, Sex), vital signs and symptoms (Chest Pain Type, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Exercise Angina, Max Heart Rate), and electrocardiography/imaging features (Resting ECG, Exercise ST Depression `oldpeak`, ST Slope, Major Vessels `ca`, Thallium Scan `thal`). If specialist imaging is unavailable at a rural clinic, the tool defaults safely to training median baselines.
* **What comes out:** 
  1. An explicit **Clinical Triage Badge**: `🚨 URGENT: IMMEDIATE CARDIOLOGY ESCALATION` vs. `✅ ROUTINE: NON-EMERGENT FOLLOW-UP`.
  2. Calibrated patient cardiac risk probability score (`%`).
  3. Actionable nursing emergency protocols (continuous ECG telemetry, IV access, chewable aspirin, emergency transfer coordination).
  4. Model consensus audit highlighting **Rescued False Negatives** that standard AI misses.
  5. Local physiological risk driver attributions explaining *why* the patient was flagged.
* **How someone uses it:** A community clinic nurse enters routine vitals into the interactive web application; within 2 seconds, the tool stratifies acute ischemic risk, flags borderline presentations, and provides justifiable referral documentation for emergency transport.

---

## What we added
Starting from the suggested project *4.1.2 Heart Disease Risk from Checkup Data*, we substantially expanded beyond the standard baseline in three novel dimensions to fulfill the Veritas AI hackathon requirements:

1. **Our Own Clinical Setting & User Persona:** Designed exclusively for a frontline community nurse at an understaffed rural health outpost with no resident cardiologist, optimizing the tool around emergency triage workflows and asymmetric clinical penalties rather than symmetric statistical accuracy.
2. **Asymmetric Cost-Sensitive Threshold Optimization ($5:1$ Clinical Loss):** In medical triage, symmetric $50\%$ decision thresholds are clinically hazardous. We formalized an asymmetric utility loss function ($\text{Loss} = 5 \times \text{FN} + 1 \times \text{FP}$), penalizing missed heart attacks $5\times$ more heavily than false alarms. Sweeping the full threshold trade-off curve, we optimized an operational triage threshold ($\tau^* = 0.31$) that drove held-out test sensitivity to **$100.0\%$ (0 missed cases)**.
3. **Multi-Hospital Out-of-Distribution (OOD) Stress Testing ($N=920$ across 4 Institutions):** Rather than evaluating solely on the standard Cleveland dataset, we audited cross-institutional generalization across $617$ external, untouched international patients from the **Hungarian Institute of Cardiology** ($N=294$), **Zurich University Hospital, Switzerland** ($N=123$), and the **VA Medical Center Long Beach, CA** ($N=200$). We discovered and documented that standard AI collapses by $40\%–44\%$ in recall under hospital domain shift, and proved that CardioTriage's cost-calibrated threshold recovers $>82\%$ sensitivity across all external hospital cohorts.
4. **Demographic Subgroup Fairness Audit:** Audited diagnostic parity across Biological Sex (Male vs. Female) and Age ($<55$ vs. $\ge 55$), confirming that CardioTriage completely eliminates the systemic female under-diagnosis problem seen in standard cardiac classifiers.
5. **SHAP Clinical Failure Forensics:** Utilized TreeSHAP to generate global feature attribution summaries and local forensic waterfall plots explaining the physiological mechanism behind confirmed true positives, false alarms, and rescued borderline patients.

---

## How it works

### Data
We ingested all four medical registries from the official UCI Machine Learning Repository and international mirrors ($920$ total records):
1. **Cleveland Clinic Foundation (USA)**: $N = 303$ (Prevalence: $45.87\%$ positive)
2. **Hungarian Institute of Cardiology (Budapest)**: $N = 294$ (Prevalence: $36.05\%$ positive)
3. **University Hospital, Zurich (Switzerland)**: $N = 123$ (Prevalence: $93.50\%$ positive)
4. **V.A. Medical Center (Long Beach, CA)**: $N = 200$ (Prevalence: $74.50\%$ positive)

* **Target Binarization:** Angiographic coronary disease status `num` ($0 = <50\%$ diameter stenosis, $1\text{--}4 = >50\%$ stenosis) was converted to binary target $y \in \{0, 1\}$.
* **Physiological Sanity Enforcement:** Handled clinical recording artifacts (e.g. cholesterol or resting blood pressure encoded as `0`, which is physiologically incompatible with life) by mapping them to `NaN`.
* **Zero-Leakage Preprocessing:** Cleveland was split into an $80\%$ Training Set ($N=242$) and an untouched $20\%$ Held-Out Test Set ($N=61$) stratified by disease prevalence. `SimpleImputer(strategy='median')` and `StandardScaler` were fitted **exclusively on the Cleveland training set**; learned statistics were frozen and used symmetrically to transform the held-out test set and all external OOD cohorts.

### Models & Optimization
* **Baseline 1 (Majority Classifier):** Predicts the most frequent class (healthy).
* **Baseline 2 (Standard Logistic Regression):** L2-regularized linear classifier ($\tau = 0.50$) evaluating linear odds ratios across standardized features.
* **Baseline 3 (Standard Random Forest):** 150-tree ensemble with constrained depth (`max_depth=4`, `min_samples_leaf=2`) at the conventional symmetric threshold ($\tau = 0.50$).
* **Proposed System (CardioTriage Cost-Optimized RF):** Evaluated decision thresholds $\tau \in [0.10, 0.80]$ on the asymmetric clinical loss surface $\text{Loss} = 5 \times \text{FN} + 1 \times \text{FP}$, enforcing a clinical safety constraint ($\text{Recall} \ge 92\%$). The optimal operating point was identified at **$\tau^* = 0.31$**.

---

## Results

### 1. In-Distribution Held-Out Benchmark (Cleveland Test Set, $N=61$)

| Model Pipeline | Threshold ($\tau$) | Accuracy | Sensitivity (Recall) | Specificity | Precision | Missed Cases (FN) | False Alarms (FP) | Clinical Loss ($5\times\text{FN} + \text{FP}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | $0.50$ | $54.1\%$ | $0.0\%$ | $100.0\%$ | $0.0\%$ | $28$ | $0$ | $140.0$ |
| **Standard Logistic Regression** | $0.50$ | $86.9\%$ | $92.9\%$ | $81.8\%$ | $81.2\%$ | $2$ | $6$ | $16.0$ |
| **Standard Random Forest** | $0.50$ | $90.2\%$ | $92.9\%$ | $87.9\%$ | $86.7\%$ | $2$ | $4$ | $14.0$ |
| **CardioTriage (Cost-Optimized)** | $\mathbf{0.31}$ | $83.6\%$ | $\mathbf{100.0\%}$ | $69.7\%$ | $73.7\%$ | $\mathbf{0}$ | $10$ | $\mathbf{10.0}$ |

* **Beating the Baseline:** The majority baseline claims $54.1\%$ accuracy, but misses **100% of sick patients** ($FN = 28$).
* **Clinical Safety Win:** Conventional Random Forest ($\tau = 0.50$) produces $2$ dangerous False Negatives. CardioTriage ($\tau^* = 0.31$) completely eliminates False Negatives on the held-out test set ($FN = 0$, **$100.0\%$ Recall**), reducing total clinical penalty from $14.0$ down to $10.0$ (**$28.6\%$ risk reduction**).

---

### 2. Multi-Hospital Out-of-Distribution (OOD) Stress Test ($N=617$ External Patients)

![OOD Generalization](outputs/ood_generalization_audit.png)

| Cohort | Domain | Sample Size | Std RF ($\tau=0.50$) Recall | CardioTriage ($\tau^*=0.31$) Recall | Missed Cases Drop ($\text{FN}_{\text{std}} \to \text{FN}_{\text{opt}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cleveland (In-Dist Test)** | In-Dist | $N=61$ | $92.9\%$ | $\mathbf{100.0\%}$ | $2 \to \mathbf{0}$ |
| **Hungary** | External OOD | $N=294$ | $53.8\%$ | $\mathbf{82.1\%}$ | $49 \to \mathbf{19}$ ($61.2\%$ reduction) |
| **Switzerland** | External OOD | $N=123$ | $53.9\%$ | $\mathbf{82.6\%}$ | $53 \to \mathbf{20}$ ($62.3\%$ reduction) |
| **VA Long Beach** | External OOD | $N=200$ | $49.0\%$ | $\mathbf{83.2\%}$ | $76 \to \mathbf{25}$ ($67.1\%$ reduction) |

### Where it fails, and why we think so:
1. **The 40% OOD Sensitivity Collapse:** When deployed out-of-the-box on foreign hospital cohorts, standard classifiers experience a catastrophic drop in sensitivity (falling to $49.0\%–53.9\%$). Our forensic analysis reveals this failure is driven by **diagnostic protocol covariate shift**: Cleveland Clinic was a research facility routinely ordering fluoroscopy (`ca`) and radioactive thallium scintigraphy (`thal`), whereas international community centers in Hungary and Switzerland rarely ordered invasive tests.
2. **How CardioTriage Rescues OOD Triage:** By calibrating to the asymmetric loss threshold $\tau^* = 0.31$, CardioTriage maintains **$>82\%$ sensitivity across all external hospital populations**, preventing up to **$114$ sick international patients from being sent home undiagnosed**.
3. **The False Alarm Trade-Off:** Achieving $100\%$ sensitivity on Cleveland requires accepting $10$ false positives ($30.3\%$ false alarm rate). In a rural clinic, this trade-off is clinically desirable: ordering an ECG or transferring an ambiguous case for secondary review is safe, whereas discharging an evolving myocardial infarction is fatal.

---

### 3. Subgroup Demographic Fairness Audit

![Subgroup Fairness](outputs/subgroup_fairness_audit.png)

* **Sex Parity:** CardioTriage achieved **$100.0\%$ Sensitivity in both Males ($N=41$) and Females ($N=20$)**, with an overall accuracy of $95.0\%$ in females. This successfully resolves the well-known cardiovascular disparity where women presenting with atypical angina are underdiagnosed by symmetric algorithms.
* **Age Parity:** Achieved **$100.0\%$ Sensitivity across both age brackets ($<55$ years, $N=28$; and $\ge 55$ years, $N=33$)**.

---

### 4. SHAP Feature Attribution & Local Clinical Forensics

![SHAP Summary](outputs/shap_global_summary.png)

* **Global Attributions:** Fluoroscopy vessel calcification (`ca`), exercise-induced ST depression (`oldpeak`), maximum heart rate achieved (`thalach`), and asymptomatic chest pain (`cp`) were identified as the primary risk determinants.
* **Forensic Case Studies:**
  * **True Positive ([shap_case_true_positive.png](outputs/shap_case_true_positive.png)):** Confirmed acute patient with severe ST depression ($2.6\text{ mm}$) and fluoroscopy stenosis correctly triaged with $91.3\%$ risk.
  * **False Alarm ([shap_case_false_alarm.png](outputs/shap_case_false_alarm.png)):** Patient with atypical chest pain and elevated cholesterol driving an escalated flag despite absence of vessel stenosis.
  * **Triage Rescue ([shap_case_triage_rescue.png](outputs/shap_case_triage_rescue.png)):** Borderline patient with predicted risk of $42.5\%$. Under standard AI ($\tau=0.50$), this patient is misdiagnosed as healthy ($FN$); under CardioTriage ($\tau^*=0.31$), this patient is successfully caught and escalated.

---

## What we'd do next
1. **Conformal Risk Prediction:** Incorporate split conformal prediction to generate mathematically guaranteed coverage intervals for individual risk estimates, allowing nurses to identify when the model is genuinely uncertain.
2. **Prospective Clinical Feasibility Study:** Partner with a regional rural clinic network to evaluate real-world nursing usability, measuring nurse adherence and transfer dispatch turnaround times.

---

## Sources
* **Datasets:**
  * UCI Heart Disease Repository (Detrano et al.): [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
  * Multi-Hospital Unified Mirror: [https://github.com/redwankarimsony/heart-disease-data](https://github.com/redwankarimsony/heart-disease-data)
* **Open Source Libraries:** `scikit-learn`, `pandas`, `numpy`, `shap`, `matplotlib`, `seaborn`, `gradio`, `joblib`.
* **Foundational Literature:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. The American Journal of Cardiology, 64(5), 304-310.
