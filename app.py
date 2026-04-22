import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler
# Make sure these files exist in the same folder!
try:
    model = joblib.load('concrete_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    st.set_page_config(page_title="Concrete Strength AI", layout="centered")

    st.title("🏗️ Concrete Compressive Strength Predictor")
    st.markdown("---")
    st.write("Adjust the mix parameters below to see the predicted strength.")

    # 2. Input Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cementitious Materials")
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 300.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 0.0)
        ash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 0.0)
        water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0)

    with col2:
        st.subheader("Aggregates & Time")
        superp = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 0.0)
        coarse = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 1000.0)
        fine = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 800.0)
        age = st.slider("Curing Age (Days)", 1, 365, 28)

    # 3. Prediction Button
    st.markdown("---")
    if st.button("Calculate Compressive Strength"):
        # Prepare data
        features = np.array([[cement, slag, ash, water, superp, coarse, fine, age]])
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)
        
        # Results Display
        st.balloons()
        st.metric(label="Predicted Strength", value=f"{prediction[0]:.2f} MPa")
        
        # Simple Logic Interpretation
        if prediction[0] > 40:
            st.success("High Strength Concrete detected.")
        elif prediction[0] > 20:
            st.info("Standard Strength Concrete.")
        else:
            st.warning("Low Strength/Early Stage Concrete.")

except FileNotFoundError:
    st.error("Error: 'concrete_model.pkl' or 'scaler.pkl' not found. Please run the training script first!")
