
import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("iri_model.pkl")
features = joblib.load("iri_features.pkl")

# App UI
st.title("ΔIRI Prediction Tool")
st.write("Enter pavement & climate data to predict the change in IRI (ΔIRI).")

inputs = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0, format="%.4f")
    inputs.append(val)

if st.button("Predict ΔIRI"):
    df = pd.DataFrame([inputs], columns=features)
    result = model.predict(df)[0]
    st.success(f"Predicted ΔIRI: {round(result, 4)}")
