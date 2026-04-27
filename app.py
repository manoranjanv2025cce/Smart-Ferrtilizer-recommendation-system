import streamlit as st
import pandas as pd
import joblib

# --------------------------
# LOAD MODEL & ENCODERS
# --------------------------
model = joblib.load("fertilizer_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Fertilizer Recommendation", layout="centered")

st.title("🌱 Smart Fertilizer Recommendation System")
st.write("Enter soil and crop details to get the best fertilizer recommendation.")

# --------------------------
# USER INPUTS
# --------------------------
nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=50.0)

# Crop input (dropdown from encoder)
crop_options = encoders['Crop Type'].classes_
crop = st.selectbox("Select Crop", crop_options)

# --------------------------
# PREDICTION BUTTON
# --------------------------
if st.button("Predict Fertilizer"):

    try:
        # Encode crop
        input_data = {
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Temparature': temperature,   # match dataset spelling
            'Humidity': humidity,
            'Moisture': moisture
        }

        if 'Crop type' in encoders:
            input_data['Crop type'] = encoders['Crop type'].transform([crop])[0]
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Prediction
        pred = model.predict(input_df)

        fertilizer = encoders['Fertilizer Name'].inverse_transform(pred)[0]

        st.success(f"🌾 Recommended Fertilizer: **{fertilizer}**")

    except Exception as e:
        st.error(f"Error: {e}")

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")