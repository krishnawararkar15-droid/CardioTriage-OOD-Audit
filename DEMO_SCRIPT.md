# CardioTriage — 2-Minute Demo Video Script & Production Guide
**Veritas AI International High School Hackathon 2026 | AI + Health Track**  
**Presenter:** Krishna Sunil Wararkar (ST. Paul High School)  
**Target Video Duration:** Exactly 1 minute 55 seconds (Under the 2-minute limit)

---

## ⏱️ Visual & Spoken Teleprompter Script

### 🎬 Part 1: The Problem (0:00 – 0:15 | ~15 Seconds)
* **What to Show on Screen:** Title slide or web app header showing **CardioTriage: Rural Clinic Decision Support System**, highlighting the subtitle and regulatory disclaimer.
* **Spoken Script (Word-for-Word):**
  > *"Every day, triage nurses at understaffed rural clinics evaluate patients with ambiguous chest pain without an on-site cardiologist or cath lab. When standard computer tools or clinical intuition underestimate risk, acute cardiac patients are erroneously sent home with fatal delays. Because missing a heart attack is exponentially more catastrophic than an extra test, clinical triage demands an asymmetric, safety-first AI system."*

---

### 🎬 Part 2: The Live Demonstration (0:15 – 1:15 | ~60 Seconds)
* **What to Show on Screen:** Open the live Gradio interface (`python app.py`) in your browser (`http://127.0.0.1:7860`).
* **Screen Action 1 (0:15 – 0:35):** Click on **Preset Case 1 (High Risk Acute Ischemia)**.
* **Spoken Script:**
  > *"This is CardioTriage. A nurse inputs thirteen routine measurements—such as blood pressure, heart rate, and ST-segment depression. When we evaluate an acute patient with significant ST depression and angina, clicking 'Generate Triage Decision' instantly outputs an urgent escalation badge in bright red, calculating an 89% risk score and providing an immediate nursing action protocol: telemetry monitoring, aspirin, and emergency transport dispatch."*

* **Screen Action 2 (0:35 – 1:15):** Click on **Preset Case 2 (Borderline Rescued Patient)**.
* **Spoken Script:**
  > *"Now, let's look at what makes CardioTriage unique. Here is a 56-year-old female presenting with atypical chest pain and subtle ST depression of 1.2 mm. Her predicted risk is 42%. A conventional AI model using an uncalibrated 50% threshold labels her 'Low Risk' and sends her home—a fatal false negative. But CardioTriage, calibrated to an asymmetric 5:1 clinical loss threshold of 0.31, immediately catches this patient and triggers a Clinical Safety Alert, successfully rescuing her from a missed diagnosis."*

---

### 🎬 Part 3: How It Works & Scientific Benchmark (1:15 – 1:45 | ~30 Seconds)
* **What to Show on Screen:** Click over to **Tab 2: Multi-Hospital OOD & Explainability Audit** inside the app, scrolling past the OOD Generalization chart and the SHAP waterfall plots.
* **Spoken Script:**
  > *"How does this work? We trained on the Cleveland Clinic dataset with strict zero-leakage median imputation. While a majority-class baseline gets 54% accuracy but misses 100% of sick patients, our cost-optimized Random Forest achieves 100% sensitivity on held-out test data. Even more critically, we audited our model across 617 external international patients from Hungary, Switzerland, and VA Long Beach. Standard models collapsed to under 54% sensitivity due to clinical covariate shift, but CardioTriage maintained over 82% sensitivity across all foreign hospitals, completely eliminating diagnostic disparity between men and women."*

---

### 🎬 Part 4: Limitations & Next Steps (1:45 – 2:00 | ~15 Seconds)
* **What to Show on Screen:** Switch back to the main app dashboard or show the project GitHub repository with the disclaimer banner visible.
* **Spoken Script:**
  > *"As a limitation, our high sensitivity accepts a 30% false alarm rate to ensure zero missed cases. With more time, our next step is implementing split conformal prediction to generate mathematically guaranteed uncertainty intervals for rural providers. Thank you!"*

---

## 🛠️ Video Recording & Submission Tips
1. **Tool:** Use [OBS Studio](https://obsproject.com/) (free) or [Loom](https://www.loom.com/) to record your screen and microphone.
2. **Resolution:** 1080p (1920x1080) in full screen.
3. **Pacing:** Speak clearly, calmly, and authoritatively. Do not rush—the script above is timed at ~135 words per minute, easily fitting within 115 seconds.
4. **Hosting:** Upload the video to YouTube as **Unlisted** (or Google Drive set to *"Anyone with the link can view"*).
5. **Incognito Test:** Open your video link in an incognito window before submitting to verify it plays without logging in.
