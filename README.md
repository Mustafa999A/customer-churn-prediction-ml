# Customer Churn Prediction using Machine Learning

## Project Overview

This project predicts whether a customer will churn using Machine Learning techniques.

The workflow includes:

- Data Cleaning
- Baseline Model (Logistic Regression)
- Leakage Detection and Fixing
- Random Forest Model Training
- Hyperparameter Tuning
- Model Deployment using Streamlit

---

## Model Performance

Final Model: Random Forest  
Accuracy: ~93–94%

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Files in Project

model_training.ipynb → Model training workflow  
streamlit_app.py → Web app interface  
churn_rf_model.pkl → Trained model  
feature_columns.pkl → Feature list  
requirements.txt → Dependencies  

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
