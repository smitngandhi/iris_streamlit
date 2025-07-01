import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Prediction

This app predicts iris flower type""")

st.sidebar.header("User Input Features")

def user_input():
    sepal_length = st.sidebar.slider("Sepal Length" , 4.3 , 7.9 , 5.4)
    sepal_width = st.sidebar.slider("Sepal Width" , 2.0 , 4.4 , 3.4)
    petal_length = st.sidebar.slider("Petal Length" , 1.0 , 6.9 , 1.3)
    petal_width = st.sidebar.slider("Petal Width" , 0.1 , 2.5 , 0.2)
    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width 
    }

    dataframe = pd.DataFrame(data , index = [0])
    return dataframe

df = user_input()

st.subheader("User input parameters")

st.write(df)

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X ,y)

prediction  = clf.predict(df)
prediction_probability = clf.predict_proba(df)

st.subheader("Class Labels and their corresponding names")
st.write(iris.target_names)

st.subheader("Predicted_class")
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_probability)