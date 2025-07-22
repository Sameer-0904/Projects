# 🩺 Diabetes Prediction Web App

An elegant and responsive **Streamlit web application** that uses a **machine learning model (SVM)** to predict the risk of diabetes based on key health parameters. Built with a clean UI, real-time prediction, and detailed health recommendations.

---

## 🚀 Live Demo
🌐 [Try the App on Streamlit](https://predict-diabetes-risk.streamlit.app/)

---

## 📊 About the Project

This project uses the **Pima Indians Diabetes Dataset** to train a **Support Vector Machine (SVM)** model and deploys it using **Streamlit**. Users can input various health metrics and get an immediate risk prediction.

### 🧠 Machine Learning Model
- **Algorithm**: Support Vector Machine (SVM)
- **Dataset**: Pima Indians Diabetes Database
- **Libraries Used**: `numpy`, `scikit-learn`, `pickle`, `streamlit`

---

## 🧾 Input Parameters

| Feature                     | Description                               | Typical Range       |
|----------------------------|-------------------------------------------|---------------------|
| **Pregnancies**            | Number of pregnancies                     | 0 – 20              |
| **Glucose**                | Glucose level (mg/dL)                     | 0 – 200             |
| **Blood Pressure**         | Diastolic BP (mmHg)                       | 0 – 140             |
| **Skin Thickness**         | Skin fold thickness (mm)                  | 0 – 100             |
| **Insulin**                | Serum insulin (μU/mL)                     | 0 – 900             |
| **BMI**                    | Body Mass Index (kg/m²)                   | 10.0 – 67.0         |
| **Diabetes Pedigree**      | Diabetes pedigree function                | 0.078 – 2.420       |
| **Age**                    | Age in years                              | 1 – 120             |

---

## 🧪 Sample Test Inputs

### 🔴 Diabetic Person
```python
[5, 160, 85, 32, 180, 35.5, 0.75, 50]
```
### 🟢 Non-Diabetic Person
```python
[1, 95, 70, 20, 85, 22.5, 0.2, 25]
```

## ⚠️ Disclaimer
This app is intended for educational and awareness purposes only.
Always consult a healthcare professional for actual diagnosis and treatment.
