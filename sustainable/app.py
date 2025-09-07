import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model and the feature names from your training data
with open('your_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# --- Title and Introduction ---
st.title("🌱 Smart Fertilizer Recommender")
st.write("Enter your field's conditions to get an optimal fertilizer recommendation.")

# --- User Input Widgets ---
st.header("Field and Crop Information")

# You'll need to know the unique categories from your original data for these
crops = ['Maize', 'Rice', 'Wheat']  # Replace with the crops in your dataset
states = ['Assam', 'Gujarat', 'Maharashtra'] # Replace with the states in your dataset

area = st.number_input("Area (in Hectares):", min_value=1.0, value=10.0, step=1.0)
pesticide = st.number_input("Pesticide Used (in kg):", min_value=0.0, value=50.0, step=1.0)
rainfall = st.number_input("Rainfall (in mm):", min_value=0.0, value=1200.0, step=1.0)
temperature = st.number_input("Temperature (in Celsius):", min_value=0.0, value=25.0, step=1.0)
selected_crop = st.selectbox("Crop Type:", crops)
selected_state = st.selectbox("State:", states)

# --- Button to Trigger Prediction ---
if st.button("Get Recommendation"):
    
    # Create a DataFrame from the user inputs
    input_data = {
        'Area': area,
        'Pesticide': pesticide,
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Crop': selected_crop,
        'State': selected_state
    }
    input_df = pd.DataFrame([input_data])
    
    # Perform one-hot encoding
    input_df = pd.get_dummies(input_df, columns=['Crop', 'State'], drop_first=True)
    
    # Reindex the DataFrame to match the model's feature order and fill missing columns with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # ----------------------------------------------------
    # Find the optimal fertilizer amount
    # ----------------------------------------------------
    best_yield = -1
    recommended_fertilizer = 0

    # Test fertilizer amounts from 0 to 200 kg
    for fertilizer_amount in np.arange(0, 201, 1):
        test_df = input_df.copy()
        test_df['Fertilizer'] = fertilizer_amount

        # Predict the yield
        predicted_yield = model.predict(test_df)[0]

        if predicted_yield > best_yield:
            best_yield = predicted_yield
            recommended_fertilizer = fertilizer_amount

    # --- Display the Results ---
    st.success("Recommendation Complete! 🚀")
    st.write(f"**Optimal Fertilizer Amount:** {recommended_fertilizer:.2f} kg")
    st.write(f"**Predicted Yield at this amount:** {best_yield:.2f} tons per hectare")