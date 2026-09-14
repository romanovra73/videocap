import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms, datasets
from PIL import Image
from fastapi import APIRouter, HTTPException, Request, Depends
from models import Img_Encoded
from ultralytics import YOLO

torch.backends.nnpack.enabled = False

router = APIRouter(
    prefix="/pool_detect",
    tags=["Детектор игры на биллиарде"]
)

async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

def get_classifier():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=256, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(torch.load("weights/pool_classifier.pt"))
    model.eval()
    return model

data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
classifier = get_classifier()

detector = YOLO('weights/yolo_pool_detect.pt');

@router.post("")
async def pool_detected(data: bytes = Depends(parse_body)):
    img_data = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
    X = Image.fromarray(img)
    X = data_transforms(X).unsqueeze(0)
    output = classifier(X)
    proba = output.item()
    return {"success":True, "data":proba, "error" : ""}

@router.post("/npersons")
async def npersons_detected(data: bytes = Depends(parse_body)):
    img_data = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
    det = detector(img)[0]
    result = 0
    for box in det.boxes:
        if box.cls[0].item()>0:
            continue
        if box.conf[0].item()<0.5:
            print('Person confidence:',box.conf[0])
            continue
        result += 1
    print('Persons:',result)
    return {"success":True, "data":result, "error" : ""}
