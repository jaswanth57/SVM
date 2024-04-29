from flask import Flask, request, jsonify
from scapy.all import sniff, IP
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# Define the feature extraction function
def extract_features(packet):
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    protocol = packet.proto
    payload_size = len(packet[IP].payload)
    return {'src_ip': src_ip, 'dst_ip': dst_ip, 'protocol': protocol, 'payload_size': payload_size}

# Function to preprocess IP addresses
def preprocess_features(features):
    features['src_ip'] = sum(bytearray(features['src_ip'], 'utf-8'))
    features['dst_ip'] = sum(bytearray(features['dst_ip'], 'utf-8'))
    return [features['src_ip'], features['dst_ip'], features['payload_size']]

# Define the machine learning model
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Adjust according to number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Training the model with dummy data
def train_model(model):
    data = {
        'src_ip': ['192.168.1.1', '192.168.1.2', '10.0.0.1'],
        'dst_ip': ['192.168.1.2', '192.168.1.1', '10.0.0.2'],
        'protocol': ['TCP', 'UDP', 'TCP'],
        'payload_size': [1500, 1300, 1200]
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['protocol'] = le.fit_transform(df['protocol'])
    X = df[['src_ip', 'dst_ip', 'payload_size']].applymap(lambda x: sum(bytearray(x, 'utf-8')) if isinstance(x, str) else x)
    y = df['protocol']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train, epochs=10)
    
# API endpoint for predicting packet data
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = {'src_ip': data['src_ip'], 'dst_ip': data['dst_ip'], 'protocol': data['protocol'], 'payload_size': data['payload_size']}
    preprocessed_features = preprocess_features(features)
    preprocessed_features = np.array([preprocessed_features])
    prediction = model.predict([preprocessed_features])
    predicted_protocol = le.inverse_transform([prediction.argmax()])
    return jsonify({'predicted_protocol': predicted_protocol[0]})


# Initialize the model and label encoder globally
model = build_model()
train_model(model)
le = LabelEncoder()  # Label encoder for protocol types
le.fit(['TCP', 'UDP'])  # Fit label encoder with your expected classes

if __name__ == "__main__":
    app.run(debug=True)