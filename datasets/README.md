# Datasets Used in the Thesis Project

This repository contains the datasets that form the foundation of my MSc thesis work on **Quantum Machine Learning** applied to healthcare prediction tasks.  

---

## 🫀 Heart Failure Prediction Dataset

- **Source:** [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- **File:** `heart.csv`  
- **Description:**  
  This dataset includes medical records of patients that can be used to predict heart disease.  
  It contains **11 clinical features** (such as age, cholesterol, blood pressure, maximum heart rate, etc.) along with a binary target variable:  
  - `HeartDisease = 1` → patient has heart disease  
  - `HeartDisease = 0` → patient does not have heart disease  

- **Columns Overview:**  
  - `Age` – Age of the patient  
  - `Sex` – Gender (M/F)  
  - `ChestPainType` – Chest pain type (e.g., ATA, NAP, ASY)  
  - `RestingBP` – Resting blood pressure  
  - `Cholesterol` – Serum cholesterol (mg/dl)  
  - `FastingBS` – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
  - `RestingECG` – Resting electrocardiogram results  
  - `MaxHR` – Maximum heart rate achieved  
  - `ExerciseAngina` – Exercise induced angina (Y/N)  
  - `Oldpeak` – ST depression induced by exercise  
  - `ST_Slope` – Slope of the peak exercise ST segment  
  - `HeartDisease` – Target variable  

---

## 🧠 Stroke Prediction Dataset

- **Source:** [Kaggle - Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- **File:** `stroke.csv`  
- **Description:**  
  This dataset contains information about patients that can be used to predict the likelihood of a stroke.  
  The dataset includes **10 features** related to demographics, lifestyle, and health history, with a binary target variable indicating the occurrence of stroke:  
  - `stroke = 1` → patient experienced a stroke  
  - `stroke = 0` → patient did not experience a stroke  

- **Columns Overview:**  
  - `id` – Patient ID  
  - `gender` – Gender of the patient  
  - `age` – Age of the patient  
  - `hypertension` – 0 = no, 1 = yes  
  - `heart_disease` – 0 = no, 1 = yes  
  - `ever_married` – Marital status  
  - `work_type` – Type of work (children, govt_job, never_worked, private, self-employed)  
  - `Residence_type` – Urban or rural  
  - `avg_glucose_level` – Average glucose level in blood  
  - `bmi` – Body Mass Index  
  - `smoking_status` – Current smoking status  
  - `stroke` – Target variable  

---

## 📌 Notes

- Both datasets originate from **Kaggle** and were preprocessed to suit the prediction tasks in this thesis.
- They are intended only for **educational and research purposes**.
- Any preprocessing or feature engineering steps applied in this project are documented within the thesis and accompanying code.  

---