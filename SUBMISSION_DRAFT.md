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

> **Disclaimer:** This project is a prototype and is not medical advice.

## The problem
Imagine a nurse working alone on a night shift at a small rural clinic. A 58-year-old patient walks in complaining of unusual fatigue, mild chest discomfort, and sweating. The nearest cardiologist and catheterization lab are two hours away by ambulance. If the nurse sends this patient home thinking it is just acid reflux, and they are actually having a heart attack, the delay in care can be fatal. In emergency triage, a missed heart attack (False Negative) is catastrophic, while an unnecessary ambulance transfer (False Positive) is just an inconvenience. Triage urgently needs an AI tool designed around patient safety, where missing a sick person is penalized far more than calling a false alarm.

## What we built
We built CardioTriage, a simple, fast web tool that helps rural nurses quickly evaluate cardiac risk and decide whether a patient needs immediate emergency transfer.
- What goes in: 13 routine checkup measurements that a nurse can collect in a few minutes: patient demographics (Age, Sex), vitals and symptoms (Chest Pain Type, Blood Pressure, Cholesterol, Blood Sugar, Exercise Angina, Max Heart Rate, ST Depression), and specialist tests (ECG, Major Vessels, Thallium Scan). If advanced scans are unavailable at a rural clinic, our tool safely fills them in with median values from our training data so the nurse still gets an answer.
- What comes out: A clear visual triage badge (🚨 URGENT: IMMEDIATE CARDIOLOGY ESCALATION or ✅ ROUTINE: NON-EMERGENT FOLLOW-UP), a risk score (0-100%), a step-by-step emergency checklist for the nurse (IV access, aspirin, ECG telemetry, ambulance dispatch), a model comparison showing if a standard AI would have missed the patient, and the top clinical reasons why they were flagged.
- How someone would use it: The nurse types the patient's vitals into our web screen right at intake. In less than 2 seconds, the tool gives a risk score, flags dangerous borderline cases, and provides clear notes to justify calling an emergency transfer.

## What we added
We started from suggested project 4.1.2: Heart disease risk from checkup data, but expanded it in three major ways:
1. Our own user and setting: Instead of building a generic hospital database model, we designed it specifically for a solo triage nurse in an understaffed rural clinic with no cardiologist on site.
2. Our own question (The 5:1 Safety Rule): Standard AI models use a default 50% cutoff—meaning if a patient has a 49% risk, the computer calls them "healthy." We created a custom clinical penalty formula where missing a heart attack is penalized 5 times more than calling a false alarm (Loss = 5 * FN + 1 * FP). By lowering the decision threshold to 31% (tau* = 0.31), our model caught 100% of sick patients on our held-out test set.
3. Our own data (Testing across 4 different hospitals, 920 patients): We wanted to see what happens when our model is tested in real hospitals around the world. We evaluated 617 untouched external patient records from the Hungarian Institute of Cardiology, Zurich University Hospital in Switzerland, and the VA Medical Center in Long Beach, California.
4. Checking fairness for men and women: Cardiac disease is notoriously under-diagnosed in women. We audited our model across sexes and age groups to ensure it protects female and older patients equally.
5. SHAP explainability: We added visual waterfall charts that explain the biological reasons behind each prediction so the nurse can trust and verify the AI's reasoning.

## How it works
- Data: 920 total patient records from the public UCI Heart Disease dataset across four hospitals: Cleveland Clinic (USA, N=303), Hungarian Institute (Budapest, N=294), Zurich University Hospital (Switzerland, N=123), and VA Medical Center (Long Beach, N=200). Target column num (0-4) was converted to binary (0: healthy, 1: disease). Impossible zeros in cholesterol and blood pressure were replaced with NaN. We split Cleveland into an 80% train set (N=242) and an untouched 20% test set (N=61). Imputers and scalers were fit strictly on the training set to guarantee zero data leakage.
- Model: A 150-tree Random Forest Classifier combined with our asymmetric 5:1 loss threshold optimization (tau* = 0.31). Compared against a Majority Baseline (predicts healthy) and clinical Logistic Regression.

## Results
- Cleveland Held-Out Test Set (N=61):
  * Majority Baseline: 54.1% Accuracy, 0.0% Recall (misses all 28 sick patients, FN=28, penalty=140.0).
  * Standard Logistic Regression (tau=0.50): 86.9% Accuracy, 92.9% Recall (FN=2, FP=6, penalty=16.0).
  * Standard Random Forest (tau=0.50): 90.2% Accuracy, 92.9% Recall (FN=2, FP=4, penalty=14.0).
  * CardioTriage (tau*=0.31): 83.6% Accuracy, 100.0% Recall (0 missed cases, FN=0, FP=10, penalty=10.0, a 28.6% cost reduction).
- Testing on Other Hospitals (N=617 External Patients):
  * Standard models collapse when deployed in new hospitals: Recall drops to 53.8% on Hungary (49 missed), 53.9% on Switzerland (53 missed), and 49.0% on VA Long Beach (76 missed).
  * CardioTriage restores robust safety: Achieves 82.1% Recall on Hungary (19 missed), 82.6% Recall on Switzerland (20 missed), and 83.2% Recall on VA Long Beach (25 missed), rescuing 114 sick patients who would have been sent home undiagnosed.
- Fairness Audit: Achieved 100.0% Recall across both Male (N=41) and Female (N=20) subgroups, and across both age brackets (<55 and >=55).
- Where it fails, and why we think so:
  1. The 40% Hospital Domain Drop: Cleveland Clinic was a research facility routinely ordering fluoroscopy (ca) and thallium scans (thal). European community clinics almost never ordered them (99% missing ca in Hungary, 96% in Switzerland). Missing scans caused standard AI recall to collapse to ~50%. While CardioTriage recovered recall to >82%, 18% of complex cases were still missed due to unmeasured tests.
  2. The 10 False Alarms: Catching 100% of sick patients required accepting 10 false alarms (30.3% false alarm rate). An unnecessary checkup is far better than sending a heart attack patient home, but it still consumes rural clinic resources.
  3. Severe Referral Bias in Switzerland: Zurich cohort had 93.5% prevalence (surgical admissions).
  4. Missing Basic Vitals in Long Beach: 56 patients had no recorded blood pressure and 53 had no recorded heart rate.

## What we'd do next
1. Uncertainty ranges (Conformal Prediction): Add confidence intervals for each risk score to tell the nurse when the data is too ambiguous for AI.
2. Vitals-Only Submodels: Train dedicated secondary models that use only basic vitals (blood pressure, heart rate, chest pain) so clinics without imaging scans don't need to guess missing tests.

## Sources
- Datasets: UCI Heart Disease Dataset (Detrano et al., 1989), DOI: 10.24432/C5630X; Mirror: github.com/redwankarimsony/heart-disease-data
- Tools: Python, pandas, scikit-learn, numpy, gradio, shap, matplotlib, seaborn
- Literature: Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." Am J Cardiol, 64(5):304-310.
- AI Assistance: Antigravity coding assistant was used for bash environment setup, plotting formats, and script checks. All clinical scoping, loss formulas, and failure analyses were directed and understood by our team.
```

---

### 3. Citations & AI Disclosure Statement
*(Required under Section 10: "Rules and responsible AI")*
* **Datasets:** UCI Heart Disease Registry (Cleveland, Hungarian, Switzerland, and VA Long Beach cohorts) curated by Dr. Robert Detrano, M.D., Ph.D.
* **Pretrained Models & Libraries:** `scikit-learn` v1.9.1, `shap` v0.54.1, `gradio` v6.32.0, `pandas` v3.0.6, `numpy` v2.4.4.
* **AI Coding Tools Disclosure:** An Antigravity coding assistant was utilized to scaffold project architecture, run bash scripts, and format visualizations. All clinical methodologies, cost matrices, experimental designs, and interpretations were verified and understood by the team.
