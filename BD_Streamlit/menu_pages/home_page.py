import streamlit as st

def home_page():
    st.title("Welcome to the Anomaly Detection App")
    st.write("This application allows you to explore and analyze data from sensors using autoencoder models for detecting potential anomalies.")
    st.write("Autoencoders are neural networks used for unsupervised learning that attempt to reproduce the input in the output, reducing the dimension of the internal representation. This makes them effective in identifying deviations or anomalies in the data, as they are able to capture the most relevant features of the dataset.")
    st.write("You can use the Sidebar to navigate between different pages.")