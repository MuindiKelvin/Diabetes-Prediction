import pandas as pd
import pickle
import streamlit as st

# Load the trained model from the pickle file
model_file_path = 'diabetes_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app code
st.title("Diabetes Prediction Model")

# Create input fields for user input
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin Level", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, value=30, step=1)

# Create a feature vector from user inputs
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree],
    'Age': [age]
})

# Make predictions using the loaded model
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display the prediction results
st.subheader("Prediction")
if prediction[0] == 1:
    st.write("The model predicts that the patient has diabetes.")
else:
    st.write("The model predicts that the patient does not have diabetes.")

# Display the prediction probabilities
st.subheader("Prediction Probabilities")
st.write("Probability of No Diabetes:", prediction_proba[0][0])
st.write("Probability of Diabetes:", prediction_proba[0][1])
