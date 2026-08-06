import streamlit as st
import pandas as pd
import joblib
model = joblib.load("best_model.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex","Housing","Saving accounts","Checking account"]}
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk is good or bad")
age = st.number_input("Age",min_value=18,max_value=89,value=30)
sex = st.selectbox("Sex",["male","female"])
housing = st.selectbox("Housing",["own","free","rent"])
Job = st.number_input("Job (0-3)",min_value=0,max_value=3,value=1)
saving_accounts = st.selectbox("Saving accounts",["little","moderate","rich","quite rich"])
checking_account = st.selectbox("Checking account",["little","moderate","rich"])
credit_amount = st.number_input("Credit amount",min_value=0,value=1000)
duration = st.number_input("Duration (in months)",min_value=1,value=12)

input_df = pd.DataFrame({
    "Age":[age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [Job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    
    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")