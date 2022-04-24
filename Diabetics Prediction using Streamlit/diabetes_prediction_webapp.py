# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:55:13 2022

@author: ASUS
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open("C:/Users/ASUS/Documents/ML/Deploy ML model/trained_model.sav",'rb'))

#creating a function for prediction
def diabetic_prediction(input_data):
    
    #changing the input data as numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    #reshaping the array as we predict for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic"
  
    
def main():
    
    #giving a title 
    st.title("Diabetes Prediction Application")
    
    #getting the input data
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies=st.text_input("Number of Pregnancies: ")
    Glucose=st.text_input("Glucose value: ")
    BloodPressure=st.text_input("Blood Pressure value: ")
    SkinThickness=st.text_input("Skin Thickness ")
    Insulin=st.text_input("Insulin value: ")
    BMI=st.text_input("BMI value: ")
    DiabetesPedigreeFunction=st.text_input("DiabetesPedigreeFunction value: ")
    Age=st.text_input("Age of the person: ")
    
    
    #code for prediction
    
    diagnosis= ""
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis=diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__=="__main__":
    main()