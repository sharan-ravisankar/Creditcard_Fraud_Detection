#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import pickle

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=100000.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

if st.button("Predict Fraud"):

    # 1. Create input with EXACT column order
    data = {}

    data["Time"] = time

    for i in range(1, 29):
        data[f"V{i}"] = 0.0   # assume normal behaviour

    data["Amount"] = amount

    df = pd.DataFrame([data])

    # 2. Scale ONLY Time and Amount
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    # 3. Predict probability
    fraud_probability = model.predict_proba(df)[:, 1][0]

    threshold = 0.8

    if fraud_probability >= threshold:
        st.error("Fraud 🚨")
    else:
        st.success("Legit ✅")

    st.write(f"**Fraud Probability:** {fraud_probability:.4f}")


# In[ ]:




