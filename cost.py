import streamlit as st
import pickle
import numpy as np


def load_model():
    with open(r"C:\Users\atber\Downloads\saved_steps2.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data['model']
le_region = data['le_region']
le_sex = data['le_sex']
le_smoker = data['le_smoker']

def show_predict_page():
    st.title("Medical Insurance Cost Predictor")

    st.write("""### We need some information to predict your medical insurance cost""")
    
    gender = ('male', 'female')
    region = ('northeast', 'southeast', 'northwest', 'southwest')
    smoker = ('yes', 'no')
    
    gen = st.selectbox("Gender", gender)
    reg = st.selectbox("Region", region)
    smoker = st.selectbox("Do you smoker", smoker)
    age = st.slider('Select your age:', 0, 100, 25)
    bmi = st.slider('Select your BMI:', 0,70,20)
    children = st.slider('Select # of children you have:', 0,20, 1)
    
    ok = st.button("Calculate Cost")
    if ok:
       X = np.array([[gen, reg, smoker, age, bmi, children]])
        
        # Transform the categorical features
       X[:, 0] = le_sex.transform(X[:, 0])
       X[:, 1] = le_region.transform(X[:, 1])
       X[:, 2] = le_smoker.transform(X[:, 2])
        
        # Convert to float for the model
       X = X.astype(float)

        # Predict the cost
       cost = regressor_loaded.predict(X)
       st.subheader(f"The estimated cost is ${cost[0]:.2f}")

show_predict_page()
