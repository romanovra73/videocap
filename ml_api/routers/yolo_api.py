from fastapi import APIRouter, HTTPException, Request, Depends
from ultralytics import YOLO
import cv2
from models import Img_Encoded
import numpy as np

router = APIRouter(
    prefix="/img_detector",
    tags=["Детектор изображений"]
)

detector = YOLO('models/yolov8n.pt')

async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

@router.post("/person_detected")
async def person_detected(data: bytes = Depends(parse_body)):
    img_data = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
    detections = detector(img)[0]
    for box in detections.boxes:
        if box.cls[0]==0 and box.conf[0]>0.5:
            return {"success": True, "data": True, "error":""}
    return {"success": True, "data": False, "error":""}
    
@router.post("/nperson_detected")
async def nperson_detected(data: bytes = Depends(parse_body)):
    img_data = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
    detections = detector(img)[0]
    n = 0
    for box in detections.boxes:
        if box.cls[0]==0 and box.conf[0]>0.5:
            n += 1
    return {"success": True, "data": n, "error":""}