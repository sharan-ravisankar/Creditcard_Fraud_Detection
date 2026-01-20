#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app predicts whether a transaction is **Fraud** or **Legit** "
    "based on a trained Machine Learning model."
)

st.divider()


st.subheader("Enter Transaction Details")


time = st.number_input(
    "Time (seconds since first transaction)",
    min_value=0.0,
    value=100000.0
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)



input_data = {
    "Time": time,
    "Amount": amount
}


for i in range(1, 29):
    input_data[f"V{i}"] = 0.0


df = pd.DataFrame([input_data])


if st.button("Predict Fraud"):

    
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

 
    fraud_probability = model.predict_proba(df)[:, 1][0]

    
    threshold = 0.8

    if fraud_probability >= threshold:
        prediction = "Fraud 🚨"
        st.error(prediction)
    else:
        prediction = "Legit ✅"
        st.success(prediction)

    st.subheader("Prediction Result")
    st.write(f"**Fraud Probability:** {fraud_probability:.4f}")


# In[ ]:




