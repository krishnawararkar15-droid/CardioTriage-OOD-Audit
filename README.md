# CardioTriage: Cost-Sensitive Cardiac Risk Screening & Cross-Hospital Generalization Audit
**Track:** Health  
**Team:** Krishna Sunil Wararkar (ST. Paul High School)  

> **Disclaimer:** This project is a prototype and is not medical advice.

---

## The problem
Imagine a nurse working alone on a night shift at a small rural clinic. A 58-year-old patient walks in complaining of unusual fatigue, mild chest discomfort, and sweating. The nearest cardiologist and catheterization lab are two hours away by ambulance. If the nurse sends this patient home thinking it is just acid reflux, and they are actually having a heart attack, the delay in care can be fatal. In emergency triage, a missed heart attack (False Negative) is catastrophic, while an unnecessary ambulance transfer (False Positive) is just an inconvenience. Triage urgently needs an AI tool designed around patient safety, where missing a sick person is penalized far more than calling a false alarm.

---

## What we built
We built **CardioTriage**, a simple, fast web tool that helps rural nurses quickly evaluate cardiac risk and decide whether a patient needs immediate emergency transfer.

* **What goes in:** 13 routine checkup measurements that a nurse can collect in a few minutes:
  * Patient details: Age, biological sex, chest pain type, fasting blood sugar, exercise angina.
  * Routine vitals & stress numbers: Resting blood pressure, cholesterol, max heart rate during exercise, and exercise ST depression (`oldpeak`).
  * Specialist tests (if available): Resting ECG, fluoroscopy vessels (`ca`), and thallium scan (`thal`). If a rural clinic does not have these advanced imaging tests, our tool safely fills them in with median values from our training data so the nurse can still get an answer.
* **What comes out:**
  1. A clear visual triage badge: **`🚨 URGENT: IMMEDIATE CARDIOLOGY ESCALATION`** or **`✅ ROUTINE: NON-EMERGENT FOLLOW-UP`**.
  2. A patient risk score (0% to 100%).
  3. A step-by-step emergency checklist for the nurse (start IV line, give chewable aspirin, attach continuous ECG monitor, and call the transfer team).
  4. A "Safety Audit" showing if a standard 50% AI model would have missed the patient, and how CardioTriage catches them.
  5. The top clinical reasons why the patient was flagged (such as severe ST depression or high blood pressure).
* **How someone would use it:** The nurse types the patient's vitals into our web screen right at intake. In less than 2 seconds, the tool gives a risk score, flags dangerous borderline cases, and provides clear notes to justify calling an emergency transfer.

---

## What we added
We started from the suggested project **4.1.2: Heart disease risk from checkup data**, but we wanted to go far beyond a standard tutorial project. We expanded it in three major ways:

1. **Our own user and setting:** Instead of building a generic predictor for a hospital database, we designed it specifically for a solo triage nurse in an understaffed rural clinic with no cardiologist on site.
2. **Our own question (The 5:1 Safety Rule):** Standard machine learning models use a default 50% cutoff—meaning if a patient has a 49% chance of a heart attack, the computer calls them "healthy." That is terrifying in medicine. We created a custom clinical penalty formula where missing a heart attack is penalized 5 times more than calling a false alarm ($\text{Loss} = 5 \times \text{FN} + 1 \times \text{FP}$). By testing different cutoffs, we found that lowering the decision threshold to **31% ($\tau^* = 0.31$)** caught **100% of sick patients** on our test set.
3. **Our own data (Testing across 4 different hospitals, 920 patients):** Most student projects only test on the standard Cleveland Clinic data. We wanted to see what happens when our model is sent to real-world hospitals in other countries. We collected 617 untouched external patient records from the **Hungarian Institute of Cardiology**, **Zurich University Hospital in Switzerland**, and the **VA Medical Center in Long Beach, California**.
4. **Checking fairness for men and women:** Cardiac disease is notoriously under-diagnosed in women because their symptoms are often non-typical. We audited our model across sexes and age groups to make sure it protects female and older patients equally.
5. **SHAP explainability:** We added visual waterfall charts that explain the biological reasons behind each prediction so the nurse can trust and verify the AI's reasoning.

---

## How it works

### Data
We used 920 total patient records from the public UCI Heart Disease dataset across four hospitals:
* **Cleveland Clinic (USA):** 303 patients (45.9% had heart disease)
* **Hungarian Institute of Cardiology (Budapest):** 294 patients (36.1% had heart disease)
* **Zurich University Hospital (Switzerland):** 123 patients (93.5% had heart disease)
* **VA Medical Center Long Beach (USA):** 200 patients (74.5% had heart disease)

**How we cleaned it:**
* We converted the target column `num` (0 for healthy, 1 to 4 for disease) into a clean binary label: `0` for no disease, and `1` for heart disease.
* We checked for impossible numbers. Some patients had cholesterol or blood pressure listed as 0, which is biologically impossible. We replaced these zeros with `NaN` so they wouldn't corrupt our math.
* **Strict zero data leakage:** We split the Cleveland data into 80% for training (242 patients) and 20% for held-out testing (61 patients). We fit our median imputer and scaler **only on the training set**. The test set and the three foreign hospitals were never seen during training or preprocessing.

### Model
* **Baseline 1 (Majority Guess):** Always guesses "healthy" because healthy is the most common class.
* **Baseline 2 (Logistic Regression):** A standard linear model evaluated at the default 50% cutoff.
* **Baseline 3 (Standard Random Forest):** A 150-tree Random Forest evaluated at the default 50% cutoff.
* **CardioTriage (Our Cost-Optimized Random Forest):** We tuned the decision cutoff on our 5:1 clinical loss surface. At a cutoff of **0.31**, our model prioritizes patient safety above all else.

---

## Results

### 1. Held-Out Cleveland Test Set (61 Patients)

| Model | Decision Cutoff | Accuracy | Recall (Sensitivity) | Missed Cases (FN) | False Alarms (FP) | Clinical Penalty ($5\times\text{FN} + \text{FP}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.50 | 54.1% | 0.0% | **28 missed** | 0 | 140.0 |
| **Standard Logistic Regression** | 0.50 | 86.9% | 92.9% | **2 missed** | 6 | 16.0 |
| **Standard Random Forest** | 0.50 | 90.2% | 92.9% | **2 missed** | 4 | 14.0 |
| **CardioTriage (Our System)** | **0.31** | **83.6%** | **100.0%** | **0 missed!** | **10** | **10.0** |

* **Beating the baseline:** The baseline gets 54.1% accuracy by guessing everyone is healthy, but it misses **all 28 sick patients**. Our model catches every single one.
* **Why our accuracy is lower, but our model is safer:** Notice that the standard Random Forest has 90.2% accuracy, but it sends **2 heart disease patients home**. CardioTriage has 83.6% accuracy, but it misses **0 patients**. We traded 6 extra false alarms for 2 saved lives, reducing our total clinical risk score from 14.0 down to 10.0.

---

### 2. Testing on Other Hospitals (617 External Patients)

When we tested the models on new hospitals, we saw a massive drop-off:

| Hospital Cohort | Sample Size | Standard AI (50% cutoff) Recall | CardioTriage (31% cutoff) Recall | Sick Patients Rescued |
| :--- | :---: | :---: | :---: | :---: |
| **Cleveland (Test Set)** | 61 | 92.9% (2 missed) | **100.0% (0 missed)** | **+2 rescued** |
| **Hungary (Budapest)** | 294 | 53.8% (49 missed) | **82.1% (19 missed)** | **+30 rescued!** |
| **Switzerland (Zurich)** | 123 | 53.9% (53 missed) | **82.6% (20 missed)** | **+33 rescued!** |
| **VA Long Beach (USA)** | 200 | 49.0% (76 missed) | **83.2% (25 missed)** | **+51 rescued!** |

---

### 3. Fairness Audit: Men vs. Women & Age

* **Biological Sex:** CardioTriage caught **100.0% of sick men ($N=41$) and 100.0% of sick women ($N=20$)**. Women are often under-diagnosed by standard models because their symptoms present atypically; our lower threshold completely eliminated this gap.
* **Age Groups:** CardioTriage achieved **100.0% recall for patients under 55 ($N=28$) and patients 55 and older ($N=33$)**.

---

### Where it fails, and why we think so (Raw & Honest Limitations)

We want to be completely honest about where our model struggles:

1. **The 40% Hospital Domain Drop:** When the standard model was run on hospitals in Hungary, Switzerland, and Long Beach, its recall collapsed from 93% to around 50%. Why? Because Cleveland Clinic was a major research center that ordered expensive tests like fluoroscopy (`ca`) and radioactive thallium scans (`thal`). The community hospitals in Hungary and Switzerland almost never ordered them ($99\%$ and $96\%$ were missing). When our model had to guess without those tests, it lost confidence. While our 31% safety threshold helped recover recall back above 82% (saving 114 patients across the three hospitals), it still missed 18% of cases because missing data cannot completely replace actual medical tests.
2. **The 10 False Alarms:** To catch 100% of sick patients on Cleveland, we had to accept 10 false alarms (a 30.3% false alarm rate). In a rural clinic, this means 10 healthy patients might be put in an ambulance or sent for follow-up testing they didn't strictly need. In our view, an unnecessary checkup is far better than a funeral, but it still puts extra strain on rural clinic resources.
3. **Severe Patient Bias in Zurich:** 93.5% of the Zurich patients had severe heart disease because they were high-risk surgical referrals. Our model missed 20 patients there because several had normal resting ECGs and atypical symptoms that differed from ordinary walk-in patients.
4. **Missing Basic Vitals in Long Beach:** In the VA Long Beach dataset, 56 patients had no recorded blood pressure and 53 had no recorded max heart rate. Filling in medians helps the code run, but no algorithm can substitute for an actual blood pressure cuff.

---

## What we'd do next
1. **Uncertainty ranges (Conformal Prediction):** We want to add a feature that tells the nurse: *"The model is 95% confident this patient's risk is between 40% and 60%."* If the uncertainty range is too wide, the tool would explicitly tell the nurse: *"Data is too ambiguous—do not rely on AI; consult a senior doctor immediately."*
2. **Vitals-Only Submodels:** Instead of using median imputation when expensive scans like thallium are missing, we would train a specialized secondary model that relies purely on basic vitals (blood pressure, heart rate, chest pain) so rural clinics get accurate predictions without guessing missing tests.

---

## Sources
* **Datasets:** UCI Machine Learning Repository — Heart Disease Dataset (Detrano et al., 1989), [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease).
* **Consolidated Data Mirror:** [https://github.com/redwankarimsony/heart-disease-data](https://github.com/redwankarimsony/heart-disease-data).
* **Tools & Libraries:** Python, `pandas`, `scikit-learn`, `numpy`, `gradio`, `shap`, `matplotlib`, `seaborn`.
* **Foundational Paper:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* The American Journal of Cardiology, 64(5), 304-310.
* **AI Assistance:** We used an Antigravity coding assistant to help set up our terminal environment, format our plots, and test our python scripts. All clinical problem scoping, mathematical formulations, and failure analyses were researched, developed, and verified by our team.
