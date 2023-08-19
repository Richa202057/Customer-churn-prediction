# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# load the model from disk
import joblib

model = joblib.load("model.sav")

# Import python scripts
from preprocessing import preprocess


def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application description
    st.sidebar.markdown("""
    :dart: This Streamlit app is designed to predict customer churn in a fictional telecommunications use case.
    
    """)
    # Setting Application sidebar default
    image = Image.open('img.webp')

    st.sidebar.image(image)


    st.info("Input data below")
    # Based on our optimal features selection
    st.subheader("Demographic data")
    seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
    dependents = st.selectbox('Dependent:', ('Yes', 'No'))
    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72,value=0)
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.selectbox('PaymentMethod', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150,value=0)
    totalcharges = st.number_input('The total amount charged to the customer', min_value=0, max_value=10000,value=0)

    st.subheader("Services signed up for")
    mutliplelines = st.selectbox("Does the customer have multiple lines", ('Yes', 'No', 'No phone service'))
    phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
    internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
    onlinesecurity = st.selectbox("Does the customer have online security", ('Yes', 'No', 'No internet service'))
    onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No', 'No internet service'))
    techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No', 'No internet service'))
    streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No', 'No internet service'))
    streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No', 'No internet service'))
    data = {
            'SeniorCitizen': seniorcitizen,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneservice,
            'MultipleLines': mutliplelines,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'Contract': contract,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges
         }
    features_df = pd.DataFrame.from_dict([data])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_df)

    # Preprocess inputs
    preprocess_df = preprocess(features_df, 'Online')

    prediction = model.predict(preprocess_df)

    if st.button('Predict'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')




if __name__ == '__main__':
    main()



