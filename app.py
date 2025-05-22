# app.py
import streamlit as st
import numpy as np
import pydicom
from PIL import Image
from utils import preprocess_image, load_model_custom, predict_disease, generate_report, segment_anomalies

st.set_page_config(page_title="AI Medical Diagnostics", layout="wide")
st.title("🧠 AI-Powered Brain Tumor Diagnostic Tool")

uploaded_file = st.file_uploader("Upload MRI/X-ray/CT image (DICOM, PNG, JPG):", type=["dcm", "png", "jpg", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".dcm"):
            dcm = pydicom.dcmread(uploaded_file)
            image = dcm.pixel_array
            metadata = {"PatientID": getattr(dcm, "PatientID", "Unknown"), "ScanType": getattr(dcm, "Modality", "DICOM")}
        else:
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            metadata = {"PatientID": "Unknown", "ScanType": "Image"}

        st.image(image, caption="🖼️ Uploaded Scan", use_column_width=True)
        processed = preprocess_image(image)
        model = load_model_custom("brain_tumor_model.keras")
        prediction = predict_disease(model, processed)

        segmentation, roi_metrics = segment_anomalies(image)
        st.image(segmentation, caption="📍 Detected Abnormal Regions", use_column_width=True)

        report = generate_report(metadata, prediction, roi_metrics)
        st.subheader("📝 Report Summary")
        st.json(report)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
