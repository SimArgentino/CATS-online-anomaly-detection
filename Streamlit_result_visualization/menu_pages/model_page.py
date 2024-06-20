import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit_option_menu
from streamlit_option_menu import option_menu

# Function for the home page
def model_page():

    # Introduction to the Dataset
    st.title("Dataset Used")
    st.write("The Controlled Anomalies Time Series (CATS) Dataset is a dataset designed for the evaluation and benchmarking of time series anomaly detection algorithms. This dataset was created to provide a controlled context in which anomalies are deliberately introduced, allowing researchers and developers to systematically test and compare the performance of their algorithms.")
    
    # Description of the Dataset Structure
    st.title("Dataset Structure")

    st.write(f"### Time Series")
    st.write("The dataset consists of various time series, each representing a sequence of data ordered over time.")
    st.write("Each time series can represent a different phenomenon, such as sensor values, financial data, environmental measurements, etc.")

    st.write(f"### Controlled Anomalies")
    st.write("Anomalies are deliberately and controllably inserted into the dataset.")
    st.write("The types of anomalies can vary and include:")
    st.markdown("- Point anomalies: individual data points that significantly deviate from normal behavior.")
    st.markdown("- Collective anomalies: groups of data points that together represent abnormal behavior.")
    st.markdown("- Contextual anomalies: data points that are anomalous only in a specific temporal context.")

    st.write(f"### Labeling")
    st.write("Each data point in the time series is labeled to indicate whether it is anomalous or normal.")
    st.write("The labels allow for verifying the accuracy of anomaly detection algorithms.")

    st.write(f"### Parameter Variation")
    st.write("The time series can vary in terms of length, sampling frequency, and complexity of normal behavior.")
    st.write("Parameter variation helps test the robustness of algorithms under different conditions.")

    # Model Description
    st.title("Model Used")
    st.write("Autoencoders are a specific architecture of artificial neural networks primarily used for unsupervised learning of data. Their main function is to learn a compressed representation of the data while preserving essential information. This technique is very useful in various applications, including dimensionality reduction, denoising, data compression, and generating new data.")
    st.write("An autoencoder consists of two main components: the encoder and the decoder. The encoder compresses the original input into a lower-dimensional representation. This neural network reduces the dimensionality of the input, producing a compact and meaningful representation known as the code. The decoder, on the other hand, attempts to reconstruct the original input from the encoded representation. In other words, it takes the lower-dimensional code and tries to restore the original input as faithfully as possible.")
    st.write("The operation of autoencoders can be summarized in several key phases. Initially, the input is passed through the encoder, transforming it into a code. This code is then passed through the decoder, which attempts to reconstruct the original input. The goal is for the reconstructed output to be as close as possible to the original input. The loss function, such as the Mean Squared Error (MSE), measures the difference between the original input and the reconstructed output. During the training phase, the parameters of the encoder and decoder are updated to minimize this loss function, using techniques like backpropagation and optimization algorithms like Stochastic Gradient Descent (SGD) or Adam.")
    st.write("Autoencoders find applications in many fields. In dimensionality reduction, they are used to compress high-dimensional data into low-dimensional representations while maintaining essential information, offering a nonlinear alternative to techniques like PCA (Principal Component Analysis). In denoising, they are employed to remove noise from data, improving its quality. Additionally, they can generate new data similar to the input, making them useful in creative and simulation applications, such as generating images and music. Finally, they are effective in anomaly detection, identifying significant differences between the original input and the reconstructed output, and are used in various contexts like system monitoring, cybersecurity, and fraud detection.")

    # Project
    st.title("Performance Evaluation")
    st.write("We trained several models by modifying both architecture and hyperparameters.")
    st.write("Below are the different performances of each model.")

    # Description of the Architectures
    st.markdown("### Architecture")
    st.write("We analyzed the variation in performance with the increase in the number of network layers.")
    st.write("Model 1:")
    code = '''def build_autoencoder(k):
    model = models.Sequential()

    # Encoder layer part
    model.add(layers.Input(shape=(k,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(8, activation='relu'))  # bottleneck layer
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    # Decoder layer part
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(k, activation='sigmoid'))

    return model'''
    st.code(code, language='python')

    st.write("Model 2:")
    code = '''def build_autoencoder(k):
    model = models.Sequential()

    # Encoder layer part
    model.add(keras.Input(shape=(k,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))  # bottleneck layer

    # Decoder layer part
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(k, activation='tanh'))
    
    return model'''
    st.code(code, language='python')
