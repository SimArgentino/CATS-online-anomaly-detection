# Import streamlit and pandas
import streamlit as st

# Function for the home page
def colab_page(): 
    st.title("Google Colab Code")

    st.write("Installation of required libraries")
    st.markdown("- Cmake: build management tool")
    st.markdown("- tensorflow: machine learning library")
    st.markdown("- tensorflow-io: TensorFlow extension for I/O integration")
    st.markdown("- kafka-python: client for Apache Kafka")
    code = '''
!pip install Cmake
!pip install tensorflow==2.10.0 tensorflow-io==0.27.0
!pip install tensorflow-io
!pip install kafka-python
'''
    st.code(code, language='python')


    st.write("Importing the necessary libraries for the project")
    code = '''
import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
'''
    st.code(code, language='python')


    st.write("Installation of gdown, a tool for downloading files from Google Drive")
    code = '''
!pip install gdown
'''
    st.code(code, language='python')


    st.write("Dataset download and initial processing")
    code = '''
!gdown 1lBYbdZSI-8jKoMp8m8CLKOtcyF3Id02u
!unzip data_parquet.zip
'''
    

    st.write("Show Tensorflow Version")
    code = '''
!pip show tensorflow
'''
    st.code(code, language='python')


    code = '''
import numpy as np
import pandas as pd
import os

df = pd.read_parquet("/content/data.parquet")
columns_to_scale = ['aimp', 'amud', 'arnd', 'asin1', 'asin2', 'adbr', 'adfl', 'bed1', 'bed2', 'bfo1', 'bfo2', 'bso1', 'bso2', 'bso3', 'ced1', 'cfo1', 'cso1']

#Data Pre-processing:
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

df_train=df[0:400000]
df_val=df[400001:500001]
df_test = df[1000000:1300000]
df.head()
'''
    st.code(code, language='python')

    
    st.write("Kafka setup")
    code = '''
import os
from datetime import datetime
import time
import threading
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
'''
    st.code(code, language='python')


    code = '''
#Download and extract Kafka:

!curl -sSOL https://dlcdn.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz
!tar -xzf kafka_2.13-3.7.0.tgz
'''
    st.code(code, language='python')


    code = '''
# Start Zookeeper in daemon (background) mode using the specified configuration file
!./kafka_2.13-3.7.0/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-3.7.0/config/zookeeper.properties

# Start the Kafka server in daemon (background) mode using the specified configuration file
!./kafka_2.13-3.7.0/bin/kafka-server-start.sh -daemon ./kafka_2.13-3.7.0/config/server.properties

# Print a message to inform the user that the system is waiting for 10 seconds
!echo "Waiting for 10 secs until kafka and zookeeper services are up and running"

# Pause the script execution for 10 seconds to allow services to start up properly
!sleep 10
'''
    st.code(code, language='python')


    code = '''
# List all processes and filter the output to show only those related to Kafka
!ps -ef | grep kafka
'''
    st.code(code, language='python')


    st.write("Creating and describing Kafka topics with the following specifications:")
    st.markdown("- cats-train: partitions=1, replication-factor=1")
    st.markdown("- cats-test: partitions=2, replication-factor=1")
    code = '''
# Create a Kafka topic named 'cats-train' with a replication factor of 1 and 1 partition
# Replication Factor: This defines the number of copies of the data across the Kafka cluster.
# Partitions: Partitions are the basic unit of parallelism in Kafka. Each topic can be split into multiple partitions, and each partition can be hosted on different brokers.
!./kafka_2.13-3.7.0/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic cats-train

# Create a Kafka topic named 'cats-test' with a replication factor of 1 and 2 partitions
!./kafka_2.13-3.7.0/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 2 --topic cats-test
'''
    st.code(code, language='python')


    code = '''
# Describe the Kafka topic 'cats-train' to get detailed information about it
!./kafka_2.13-3.7.0/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic cats-train

# Describe the Kafka topic 'cats-test' to get detailed information about it
!./kafka_2.13-3.7.0/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic cats-test
'''
    st.code(code, language='python')


    st.write("Splitting the dataset into Train, Validation and Test Set")
    code = '''
#Divisione in Train, Validation e Test

x_train_df = df_train.drop(["y"], axis=1)
y_train_df = df_train["y"]


x_val_df = df_val.drop(["y"], axis=1)
y_val_df = df_val["y"]

x_test_df = df_test.drop(["y"], axis=1)
y_test_df = df_test["y"]

x_test = list(filter(None, x_test_df.to_csv(index=False).split("\n")[1:]))
y_test = list(filter(None, y_test_df.to_csv(index=False).split("\n")[1:]))


NUM_COLUMNS = len(x_test_df.columns)
'''
    st.code(code, language='python')


    code = '''
def error_callback(exc):
    raise Exception('Error while sending data to Kafka: {0}'.format(str(exc)))


# Define a function to send messages to a specified Kafka topic.
def write_to_kafka(topic_name, items):
    # Initialize a counter to keep track of the number of messages sent.
    count = 0

    # Create a Kafka producer that connects to a Kafka server at '127.0.0.1:9092'.
    producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])

    # Iterate over each (message, key) pair in the `items` iterable.
    for message, key in items:
        # Send the message to the specified Kafka topic with the key and value encoded in UTF-8.
        # Attach the error callback to handle any errors that occur during sending.
        producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(error_callback)

        # Increment the count of messages sent.
        count += 1

    # Ensure that all buffered messages are sent to the Kafka topic before proceeding.
    producer.flush()


    print("Wrote {0} messages into topic: {1}".format(count, topic_name))


write_to_kafka("cats-test", zip(x_test, y_test))
'''
    st.code(code, language='python')


    st.write("Creating the model architecture")
    code = '''
def build_autoencoder(k):

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
    return model
'''
    st.code(code, language='python')


    code = '''
# Compile the Model
input_shape = 17
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

autoencoder.summary()
'''
    st.code(code, language='python')


    st.write("Training the created model")
    code = '''
import matplotlib.pyplot as plt
import pandas as pd

# Model training
history = autoencoder.fit(x_train_df, x_train_df, epochs=30, batch_size=64, validation_split=0.3)

# Extract loss and accuracy
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history.get('accuracy', [])  # If there is no accuracy, leave the list empty
val_accuracy = history.history.get('val_accuracy', [])  # If there is no validation accuracy, leave the list empty
epochs = range(1, len(loss) + 1)

# Save results to a CSV file
results_df = pd.DataFrame({
    'Epoch': epochs,
    'Training Loss': loss,
    'Validation Loss': val_loss,
    'Training Accuracy': accuracy if accuracy else [None] * len(epochs),  # Fill with None if empty
    'Validation Accuracy': val_accuracy if val_accuracy else [None] * len(epochs)  # Fill with None if empty
})

results_df.to_csv('training_results.csv', index=False)
print("Training results saved in training_results.csv")

# Plot Loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Accuracy
if accuracy and val_accuracy:  # Plot only if there are accuracy data
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
'''
    st.code(code, language='python')


    st.write("Performance evaluation")
    code = '''
#Calcola l'errore di ricostruzione sui dati di validazione

validation_data = x_val_df
reconstructions = autoencoder.predict(validation_data)
reconstruction_errors = np.mean((validation_data - reconstructions)**2, axis=1)
threshold = np.percentile(reconstruction_errors, 95)
'''
    st.code(code, language='python')


    code = '''print(x_val_df)'''
    st.code(code, language='python')

    code = '''print(threshold)'''
    st.code(code, language='python')

    code = '''BATCH_SIZE=64'''
    st.code(code, language='python')


    code = '''
# Create a KafkaGroupIODataset using TensorFlow I/O (tfio) experimental streaming API.

test_ds = tfio.experimental.streaming.KafkaGroupIODataset( #This specific dataset is designed to consume data from Kafka topics in a streaming fashion, enabling real-time data processing within TensorFlow workflows.

    topics=["cats-test"],  # Specify the Kafka topic(s) from which data will be consumed.
    group_id="testcg",  # Set the consumer group ID for coordinating message consumption.
    servers="127.0.0.1:9092",  # Define the Kafka broker(s) address for establishing connection.
    stream_timeout=10000,  # Set the stream timeout in milliseconds
    configuration=[
        "session.timeout.ms=7000",  # Configure session timeout for Kafka consumer in milliseconds.
        "max.poll.interval.ms=8000",  # Configure maximum poll interval for Kafka consumer in milliseconds.
        "auto.offset.reset=earliest"  # Set the consumer to read from the earliest available offset on startup.
    ],
)


# Define a function to decode Kafka messages and keys into structured data.
def decode_kafka_test_item(raw_message, raw_key):
    # Decode the raw_message (CSV format) into a TensorFlow tensor with default values of 0.0 for each column.
    message = tf.io.decode_csv(raw_message, [[0.0] for i in range(NUM_COLUMNS)])

    # Convert the raw_key from string format to a numerical TensorFlow tensor.
    key = tf.strings.to_number(raw_key)

    # Return a tuple of decoded message and key.
    return (message, key)

# Apply the decode_kafka_test_item function to each element in test_ds using the map transformation.
test_ds = test_ds.map(decode_kafka_test_item)
test_ds = test_ds.batch(BATCH_SIZE)
'''
    st.code(code, language='python')


    st.write("Inference phase")
    code = '''
predictions = []
ground_truths = []

# Iterate over the dataset and perform inference
for batch in test_ds:
    inputs, labels = batch
    # Perform inference
    preds = autoencoder.predict(inputs)
    mse = np.mean(np.power(inputs - preds, 2), axis=1)
    # Convert probabilities to binary classes (0 or 1) using a threshold of 0.5
    binary_preds = mse > threshold

    # Extend the lists with new predictions and ground truth
    predictions.extend(binary_preds)
    ground_truths.extend(labels.numpy())

# Save the results to a CSV file
# Assuming predictions are single-dimensional vectors
df = pd.DataFrame({
    'Prediction': [pred[0] if isinstance(pred, (list, np.ndarray)) else pred for pred in predictions],
    'Ground Truth': ground_truths
})

df.to_csv('predictions_vs_ground_truth.csv', index=False)

print("Predictions and ground truth saved to predictions_vs_ground_truth.csv")
'''
    st.code(code, language='python')


    st.write("Visualizing the final results")
    code = '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Calculate metrics
accuracy = accuracy_score(ground_truths, predictions)
precision = precision_score(ground_truths, predictions)
recall = recall_score(ground_truths, predictions)
f1 = f1_score(ground_truths, predictions)
roc_auc = roc_auc_score(ground_truths, predictions)

# Calculate confusion matrix
conf_matrix = confusion_matrix(ground_truths, predictions)

# Save metrics to a CSV file
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, roc_auc]
})

metrics_df.to_csv('evaluation_metrics.csv', index=False)
print("Evaluation metrics saved to evaluation_metrics.csv")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(ground_truths, predictions)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()
'''
    st.code(code, language='python')
