from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import json

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# Define class names (must match your dataset classes)
classNames = [
    'LicensePlate',
    'With Helmet',
    'Without Helmet',
    'noSeatbelt',
    'number plate',
    'rider',
    'seatbelt',
    'triple ride',
    'with helmet',
    'without helmet'
]

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO inference
        results = model(img)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                det = {
                    "class_id": cls,
                    "class_name": classNames[cls],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                }

                # Run OCR if it's a number plate
                if classNames[cls].lower() in ["licenseplate", "number plate"]:
                    crop = img[y1:y2, x1:x2]
                    ocr_result = reader.readtext(crop)
                    text = " ".join([res[1] for res in ocr_result])
                    det["plate_text"] = text

                detections.append(det)

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

