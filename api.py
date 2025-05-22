# api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import pydicom
import numpy as np
from PIL import Image
import io
import os
import uuid
import shutil

from utils import (
    preprocess_image,
    load_model_custom,
    predict_disease,
    segment_anomalies,
    generate_report
)

app = FastAPI()
MODEL_PATH = "brain_tumor_model.keras"
model = load_model_custom(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".dcm"):
            dcm = pydicom.dcmread(io.BytesIO(contents))
            image = dcm.pixel_array
            metadata = {
                "PatientID": getattr(dcm, "PatientID", "Unknown"),
                "ScanType": getattr(dcm, "Modality", "DICOM")
            }
        else:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            image = np.array(pil_img)
            metadata = {"PatientID": "Unknown", "ScanType": "Image"}

        processed = preprocess_image(image)
        prediction = predict_disease(model, processed)
        segmentation, roi_metrics = segment_anomalies(image)
        report = generate_report(metadata, prediction, roi_metrics)
        return JSONResponse(content=report)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate_report")
async def generate_patient_report(
    patient_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    scan_date: str = Form(...),
    conditions: List[str] = Form(...),
    confidences: List[float] = Form(...),
    severities: List[str] = Form(...),
    annotated_image: UploadFile = File(None),
    scan_comparison: str = Form(None)
):
    image_path = None
    if annotated_image:
        os.makedirs("static", exist_ok=True)
        ext = annotated_image.filename.split(".")[-1]
        image_path = f"static/{uuid.uuid4()}.{ext}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(annotated_image.file, f)

    patient_info = {
        "name": name,
        "age": age,
        "gender": gender,
        "scan_date": scan_date
    }

    predictions = [
        {"condition": cond, "confidence": conf, "severity": sev}
        for cond, conf, sev in zip(conditions, confidences, severities)
    ]

    json_path, pdf_path = generate_report(
        patient_id=patient_id,
        patient_info=patient_info,
        predictions=predictions,
        annotated_image_path=image_path,
        scan_comparison=scan_comparison
    )

    return {
        "message": "Report generated successfully",
        "json_report": json_path,
        "pdf_report": pdf_path
    }
