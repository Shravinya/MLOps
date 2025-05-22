# utils.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import cv2

# Preprocess image to match model input
def preprocess_image(image):
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = image[..., :3]

    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# Load model
def load_model_custom(path):
    return tf_load_model(path)

# Prediction
def predict_disease(model, processed):
    pred = model.predict(processed)[0]
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return {
        "class": classes[np.argmax(pred)],
        "confidence": float(np.max(pred))
    }

# Dummy segmentation and ROI metrics (you can replace this with real model output)
def segment_anomalies(image):
    output = np.zeros_like(image)
    output = cv2.circle(output, (output.shape[1]//2, output.shape[0]//2), 60, (255, 0, 0), 5)
    metrics = {
        "anomaly_area": "Estimated ~5.2%",
        "region_confidence": "High"
    }
    return output, metrics

# Report generator
def generate_report(metadata, prediction, roi_metrics):
    return {
        "PatientID": metadata["PatientID"],
        "ScanType": metadata["ScanType"],
        "Diagnosis": prediction["class"],
        "Confidence": f"{prediction['confidence']*100:.2f}%",
        "RegionAnalysis": roi_metrics
    }
