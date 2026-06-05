# heart-disease-ml-model
# ❤️ Heart Failure Prediction

A machine learning project that predicts the likelihood of heart failure in patients based on clinical features. Built using Python and Logistic Regression with hyperparameter tuning.

---

## 📌 Project Overview

Heart failure is a critical medical condition that affects millions of people worldwide. Early prediction can significantly improve patient outcomes. This project uses a supervised machine learning approach to classify whether a patient is at risk of heart failure based on key health indicators.

---

## 📊 Dataset

- **Source:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Records:** 918 patients
- **Features:** 11 clinical features + 1 target variable (HeartDisease)

### Features Used:
| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Gender (M/F) |
| ChestPainType | Type of chest pain (ASY, ATA, NAP, TA) |
| Cholesterol | Serum cholesterol in mm/dl |
| FastingBS | Fasting blood sugar |
| RestingECG | Resting electrocardiogram results |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of the peak exercise ST segment |

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — shape, dtypes, null values
2. **EDA (Univariate & Multivariate Analysis)** — KDE plots, countplots, heatmaps
3. **Outlier Detection & Treatment** — IQR capping method
4. **Feature Encoding** — Manual label encoding for categorical features
5. **Multicollinearity Check** — VIF (Variance Inflation Factor)
6. **Model Building** — Compared Logistic Regression, Decision Tree, SVC
7. **Hyperparameter Tuning** — GridSearchCV on Logistic Regression
8. **Final Model** — Logistic Regression (C=1, penalty='l2')

---

## 🤖 Models Compared

| Model | Accuracy |
|---|---|
| Decision Tree | Lower |
| SVC | Medium |
| **Logistic Regression** | **Highest ✅** |

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Statsmodels (VIF)
- Google Colab

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Run all cells sequentially
3. The dataset is automatically downloaded via `kagglehub`

```bash
# Install dependencies if needed
pip install kagglehub scikit-learn pandas matplotlib seaborn plotly statsmodels
```

---

## 📁 Project Structure

```
heart-failure-prediction/
│
├── Heart_Failure_Prediction.ipynb   ← Main notebook
├── README.md                        ← Project documentation
└── requirements.txt                 ← Dependencies
```

---

## 📈 Key Findings

- Patients with **ASY (Asymptomatic) chest pain** are at higher risk
- **ST_Slope (Flat)** is strongly associated with heart disease
- **Logistic Regression** outperformed other models after hyperparameter tuning
- Dataset had **no missing values** but required outlier treatment for RestingBP, Cholesterol, and Oldpeak

- ## 💖 App Working

- <img width="547" height="607" alt="image" src="https://github.com/user-attachments/assets/1932ec96-5bb9-4db6-948f-1090b2c130d5" />


---

## 👤 Author

**[Ayush Kumar Jha]**
- GitHub: [@Ayush-kumar-jha953](https://github.com/Ayush-kumar-jha953)





