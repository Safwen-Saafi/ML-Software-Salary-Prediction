import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
"United States of America",                                                  
"Germany",
"United Kingdom of Great Britain and Northern Ireland",
"Canada",                                                   
"India",                                                   
"France",                                                   
"Netherlands",                                             
"Australia",                                                
"Brazil",                                               
"Spain",                                                   
"Sweden",                                                  
"Italy",                                                    
"Poland",                                                  
"Switzerland",                                              
"Denmark",                                                 
"Norway",                                                  
"Palestine",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3) #3 is the default value

    ok = st.button("Calculate Salary") #if we click on the button this is true otherwise it is false
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)  #Salary is anumpy array with only one value
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")