import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models

import streamlit_option_menu
from streamlit_option_menu import option_menu
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def build_autoencoder(model_type):
    if model_type == "Model 1":
        model = models.Sequential()

        # Encoder layer part
        model.add(layers.Input(shape=(17,)))
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
        model.add(layers.Dense(17, activation='sigmoid'))

    if model_type == "Model 2":
        model = models.Sequential()

        # Encoder layer part
        model.add(keras.Input(shape=(17,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))  # bottleneck layer

        # Decoder layer part
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(17, activation='tanh'))
    
    return model

def model_build(model_type, optimizer, epoch, percentile):
    # Model loading
    if model_type == "Model 1":
        autoencoder = build_autoencoder("Model 1")
        if optimizer == "Adam" and epoch == "30" and percentile == "90":
            autoencoder.load_weights("models\m1_adam_e30.h5")
            threshold = 0.712901719027131
        if optimizer == "Adam" and epoch == "30" and percentile == "95":
            autoencoder.load_weights("models\m1_adam_e30.h5")
            threshold = 0.8540586741485884
        if optimizer == "Adam" and epoch == "60" and percentile == "90":
            autoencoder.load_weights("models\m1_adam_e60.h5")
            threshold = 0.7195667774220919
        if optimizer == "Adam" and epoch == "60" and percentile == "95":
            autoencoder.load_weights("models\m1_adam_e60.h5")
            threshold = 0.8690659052845591
        if optimizer == "RMSprop" and epoch == "30" and percentile == "90":
            autoencoder.load_weights("models\m1_rms_e30.h5")
            threshold = 0.738507739374793
        if optimizer == "RMSprop" and epoch == "30" and percentile == "95":
            autoencoder.load_weights("models\m1_rms_e30.h5")
            threshold = 0.880710482063847
        if optimizer == "RMSprop" and epoch == "60" and percentile == "90":
            autoencoder.load_weights("models\m1_rms_e60.h5")
            threshold = 0.729910393617794
        if optimizer == "RMSprop" and epoch == "60" and percentile == "95":
            autoencoder.load_weights("models\m1_rms_e60.h5")
            threshold = 0.8784755999776444

    if model_type == "Model 2":
        autoencoder = build_autoencoder("Model 2")
        if optimizer == "Adam" and epoch == "30" and percentile == "90":
            autoencoder.load_weights("models\m2_adam_e30.h5")
            threshold = 0.22526050543400736
        if optimizer == "Adam" and epoch == "30" and percentile == "95":
            autoencoder.load_weights("models\m2_adam_e30.h5")
            threshold = 0.37730381179387495
        if optimizer == "Adam" and epoch == "60" and percentile == "90":
            autoencoder.load_weights("models\m2_adam_e60.h5")
            threshold = 0.22454969771467106
        if optimizer == "Adam" and epoch == "60" and percentile == "95":
            autoencoder.load_weights("models\m2_adam_e60.h5")
            threshold = 0.37595087812035954
        if optimizer == "RMSprop" and epoch == "30" and percentile == "90":
            autoencoder.load_weights("models\m2_rms_e30.h5")
            threshold = 0.22683427400617032
        if optimizer == "RMSprop" and epoch == "30" and percentile == "95":
            autoencoder.load_weights("models\m2_rms_e30.h5")
            threshold = 0.37864416533085626
        if optimizer == "RMSprop" and epoch == "60" and percentile == "90":
            autoencoder.load_weights("models\m2_rms_e60.h5")
            threshold = 0.2267485741862487
        if optimizer == "RMSprop" and epoch == "60" and percentile == "95":
            autoencoder.load_weights("models\m2_rms_e60.h5")
            threshold = 0.38110223655544756
    
    return autoencoder, threshold

# Function to display anomalies
def execution_page():
    # Page title
    st.title("Execution of the selected model")

    # Model selection
    st.sidebar.title("Model Selection:")

    # Model sidebar
    with st.sidebar:
        model_type = option_menu(
            menu_title = "Model",
            options = ["Model 1", 
                       "Model 2"],
            icons = ["","star-fill"],
            menu_icon = "pc-display",
            default_index = 0,
        )
    
    # Parameter selection
    st.sidebar.title("Parameter Selection:")
        
    # Optimizer sidebar
    with st.sidebar:
        optimizer = option_menu(
            menu_title = "Optimizer",
            options = ["Adam", 
                       "RMSprop"],
            icons = ["","star-fill"],
            menu_icon = "speedometer",
            default_index = 0,
        )
    
    # Epoch sidebar
    with st.sidebar:
        epoch = option_menu(
            menu_title = "Epoch",
            options = ["30", 
                       "60"],
            icons = ["star-fill"],
            menu_icon = "speedometer",
            default_index = 0,
        )
    
    # Percentile sidebar
    with st.sidebar:
        percentile = option_menu(
            menu_title = "Percentile",
            options = ["90", 
                       "95"],
            icons = ["star-fill"],
            menu_icon = "speedometer",
            default_index = 0,
        )
    
    # Upload CSV file for analysis
    file = st.file_uploader("Upload CSV file of sensor data")

    if file is not None:
        # Read data
        df = pd.read_csv(file, parse_dates=['timestamp'], index_col=0)

        predictions = []
        ground_truths = []

        # Select time range to analyze
        start_data = df.first_valid_index()
        end_data = df.last_valid_index()
        date_format = '%Y-%m-%d %H:%M:%S'

        range = st.slider(
            "Select range to analyze",
            min_value = start_data,
            max_value = end_data,
            step = df.index[1]-df.index[0],
            value = (datetime.strptime(str(start_data), date_format), datetime.strptime(str(end_data), date_format)),
            format = "YY/MM/DD - hh:mm:ss"
        )

        # Select time range
        df2 = df.loc[range[0]:range[1]]
        df3 = df2.drop(["y"], axis=1)
        st.dataframe(df3)

        if st.button('Show anomalies'):
            # Progress Bar setup
            progress_bar = st.progress(0, text="Creating model...")

            autoencoder, threshold = model_build(model_type, optimizer, epoch, percentile)

            progress_bar.progress(10, text="Predicting...")

            # Prediction
            preds = autoencoder.predict(df3)
            mse = np.mean(np.power(df3 - preds, 2), axis=1)

            # Convert probabilities to binary classes (0 or 1) using a threshold of 0.5
            binary_preds = mse > threshold

            # Extend lists with new predictions and ground truth
            predictions.extend(binary_preds)
            ground_truths.extend(df["y"])

            progress_bar.progress(20, text="Generating results...")
            
            anomalies = binary_preds[binary_preds == 1]
            df4 = df2[df2.index.isin(anomalies.index)]

            num_anomalies = len(anomalies)
            if num_anomalies > 0:
                st.markdown(f"<span style='color:red'>Found {num_anomalies} anomalies.</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green'>No anomalies found.</span>", unsafe_allow_html=True)

            # Plot AIMP
            progress_bar.progress(30, text="Creating AIMP plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y="aimp", 
                          title='Sensor anomalies: aimp')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="aimp", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig) 
            
            # Plot AMUD and ARND
            progress_bar.progress(40, text="Creating AMUD and ARND plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["amud", "arnd"], 
                          title='Sensor anomalies: amud, arnd')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="amud", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="arnd", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)

            # Plot ASIN1 and ASIN2
            progress_bar.progress(40, text="Creating ASIN1 and ASIN2 plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["asin1", "asin2"], 
                          title='Sensor anomalies: asin1, asin2')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="asin1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="asin2", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)

            # Plot ADBR and ADFL
            progress_bar.progress(50, text="Creating ADBR and ADFL plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["adbr", "adfl"],
                          title='Sensor anomalies: adbr, adfl')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="adbr", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="adfl", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)

            # Plot BED1 and BED2
            progress_bar.progress(60, text="Creating BED1 and BED2 plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["bed1", "bed2"], 
                          title='Sensor anomalies: bed1, bed2')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bed1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bed2", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)

            # Plot BFO1 and BFO2
            progress_bar.progress(70, text="Creating BFO1 and BFO2 plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["bfo1", "bfo2"], 
                          title='Sensor anomalies: bfo1, bfo2')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bfo1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bfo2", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)
            

            # Plot BSO1, BSO2 and BSO3
            progress_bar.progress(80, text="Creating BSO1, BSO2 and BSO3 plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["bso1", "bso2", "bso3"], 
                          title='Sensor anomalies: bso1, bso2, bso3')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bso1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bso2", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="bso3", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)
            

            # Plot CED1, CFO1 and CSO1
            progress_bar.progress(90, text="Creating CED1, CFO1 and CSO1 plots...")
            fig = px.line(df3, 
                          x=df3.index, 
                          y=["ced1", "cfo1", "cso1"],
                          title='Sensor anomalies: ced1, cfo1, cso1')
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="ced1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="cfo1", 
                                     color_discrete_sequence=["red"]).data[0])
            fig.add_trace(px.scatter(df4, x=df4.index, 
                                     y="cso1", 
                                     color_discrete_sequence=["red"]).data[0])
            st.plotly_chart(fig)

            # Update progress to 100% at the end
            progress_bar.progress(100)

            # Hide progress bar after execution
            progress_bar.empty()
