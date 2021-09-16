import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
#Iris Flower Prediction Web-App

This web application is used to predict the category of an iris flower using Machine Learning
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

# model = pkl.load(open('model_pickle', 'rb'))
#
# prediction = model.predict(df)
# prediction_prob = model.predict_proba(df)

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
X=iris.data
Y=iris.target

model=RandomForestClassifier()
model.fit(X,Y)

prediction = model.predict(df)
prediction_proba = model.predict(df)

st.subheader('Class labels and their index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)