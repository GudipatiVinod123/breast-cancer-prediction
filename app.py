import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Breast cancer Prediction App")
st.write("This app predicts whether a person has Breast cancer based on their health parameters.")
# Load the trained model
model=joblib.load("breast_cancer_model.pkl")
model_scaler=joblib.load("scaler.pkl")

#message
radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=30.0,help="Range: 0.0 - 30.0")
st.caption("Range: 0.0 - 30.0")
texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0,help="Range: 0.0 - 40.0")
st.caption("Range: 0.0 - 40.0")
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0,help="Range: 0.0 - 200.0")
st.caption("Range: 0.0 - 200.0")
area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0)
st.caption("Range: 0.0 - 2500.0")
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")                  
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
radius_se = st.number_input("Radius SE", min_value=0.0, max_value=5.0)
st.caption("Range: 0.0 - 5.0")
texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0)
st.caption("Range: 0.0 - 5.0")
perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=50.0)
st.caption("Range: 0.0 - 50.0")
area_se = st.number_input("Area SE", min_value=0.0, max_value=600.0)
st.caption("Range: 0.0 - 600.0")
smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")

radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=40.0)
st.caption("Range: 0.0 - 40.0")
texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0)
st.caption("Range: 0.0 - 50.0")
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0)
st.caption("Range: 0.0 - 300.0")
area_worst = st.number_input("Area Worst", min_value=0.0, max_value=5000.0)
st.caption("Range: 0.0 - 5000.0")
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=1.0)
st.caption("Range: 0.0 - 1.0")

#prediction
if st.button("Predict"):
    
    input_data = np.array([[ 
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])

    scaling = model_scaler.transform(input_data)
    prediction = model.predict(scaling)

    if prediction[0] == 1:
        st.success("The person is having breast cancer.")
    else:
        st.success("The person is not having breast cancer.")
