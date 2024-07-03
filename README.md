<h1 align="center" id="title">CATS-online-anomaly-detection</h1>

<p align="center"><img src="https://socialify.git.ci/SimArgentino/CATS-online-anomaly-detection/image?description=1&amp;font=Bitter&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Brick%20Wall&amp;tab=readme-ov-file%3Flanguage%3D1&amp;theme=Light" alt="project-image"></p>

<h2>📜 Project Description:</h2>

The goal of this project is to create a notebook for anomaly detection and to simulate online detection using Kafka.

The dataset used is the Controlled Anomalies Time Series (CATS) dataset. The notebook is designed to swap out the dataset to perform anomaly detection easily. The neural network used is an Autoencoder.

For this project, 16 models were trained to evaluate the results in different scenarios by assessing the following characteristics:

    Optimizer: Adam-RMSprop
    Epochs: 30-60
    Threshold selection percentile
    Two different autoencoder models

The results are viewable on Streamlit, where you can also upload a dataset and choose your model to perform anomaly detection. The interface allows you to select the model section for anomaly detection in real-time and visualize the results.

<h2>📊 Project Graph: </h2>
  
  ![ProjectGraph](https://github.com/SimArgentino/CATS-online-anomaly-detection/assets/93777986/617f1036-a83a-4aff-9444-5c033e9b20ea)



<h2>🛠️ Installation Steps:</h2>
<p>To view our results you can run the Streamlit app:</p>

<p>1. Install requirements</p>

```
pip install -r ./CATS-online-anomaly-detection/Streamlit_result_visualization/requirements.txt
```

<p>2. Run the streamlit app</p>

```
streamlit run ./CATS-online-anomaly-detection/Streamlit_result_visualization/app.py
```

<h2>🫵 Build your model </h2>
You can swap the model by changing the model section in the notebook.
After you save the model, move it in the following path:    

```
CATS-online-anomaly-detection/Streamlit_result_visualization/models/
```

<h2>💻 Built with</h2>

Technologies used in the project:

*   Kafka
*   Tensorflow
*   Streamlit
