import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Model & Features
# =========================

model = pickle.load(open("churn_rf_model.pkl", "rb"))

feature_columns = pickle.load(
    open("feature_columns.pkl","rb")
)

# =========================
# UI
# =========================

st.title("Customer Churn Prediction App")

st.write(
"Enter customer details to predict churn."
)

# =========================
# Inputs
# =========================

account_length = st.number_input("Account Length", 0)

vmail_message = st.number_input("VMail Message", 0)

day_mins = st.number_input("Day Minutes", 0.0)

eve_mins = st.number_input("Evening Minutes", 0.0)

night_mins = st.number_input("Night Minutes", 0.0)

intl_mins = st.number_input("International Minutes", 0.0)

custserv_calls = st.number_input("Customer Service Calls", 0)

day_calls = st.number_input("Day Calls", 0)

eve_calls = st.number_input("Evening Calls", 0)

night_calls = st.number_input("Night Calls", 0)

intl_calls = st.number_input("International Calls", 0)

intl_plan = st.selectbox(
    "International Plan",
    ["No","Yes"]
)

vmail_plan = st.selectbox(
    "Voice Mail Plan",
    ["No","Yes"]
)

# Convert Yes/No to 0/1

intl_plan = 1 if intl_plan == "Yes" else 0

vmail_plan = 1 if vmail_plan == "Yes" else 0

# =========================
# Create Input Dictionary
# =========================

input_dict = {
    'Account Length': account_length,
    'VMail Message': vmail_message,
    'Day Mins': day_mins,
    'Eve Mins': eve_mins,
    'Night Mins': night_mins,
    'Intl Mins': intl_mins,
    'CustServ Calls': custserv_calls,
    'Day Calls': day_calls,
    'Eve Calls': eve_calls,
    'Night Calls': night_calls,
    'Intl Calls': intl_calls,
    'Intl Plan': intl_plan,
    'VMail Plan': vmail_plan
}

# Convert to DataFrame

input_data = pd.DataFrame([input_dict])

# Ensure correct column order

input_data = input_data[feature_columns]

# =========================
# Prediction
# =========================

if st.button("Predict Churn"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:

        st.error(
            f"Customer is likely to Churn "
            f"(Probability: {probability[0][1]*100:.2f}%)"
        )

    else:

        st.success(
            f"Customer is likely to Stay "
            f"(Probability: {probability[0][0]*100:.2f}%)"
        )
