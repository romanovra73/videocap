import sys
sys.path.append('classes')
sys.path.append('libs')
import track
import database
import person_detection
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime,timedelta
import time
import torch
import torchvision.models as torch_models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

router = APIRouter(
    prefix = "/ch1_tracker",
    tags = ["Трекер канала 1"]
)

print('Tracker is starting...')
maha_weight = 0.75
max_ttl = 3

yolo_detector = YOLO('models/yolov8n.pt')
print('Yolo model is loaded')

db = database.get_postgres_connection()

if db==False:
    sys.exit()
cur = db.cursor()
print('Database is connected')

resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
encoder.eval()
transform_pipeline = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print('Encoder is loaded')


query = '''SELECT * FROM video_raw.face_detect_big WHERE datetime>%s ORDER BY datetime LIMIT 1'''
del_query = '''DELETE FROM video_raw.face_detect_big WHERE datetime=%s'''

@router.get("")
async def init_tracking(background_tasks: BackgroundTasks):
    background_tasks.add_task(init_tracking_process)
    return {"success": True, "error": ""}

def init_tracking_process():
    tracks = {}
    next_time = track.Track.get_last_track_time(db)
    ID=1
    tracks = {}
    fl = True
    while(fl):
        cur.execute(query, (next_time,))
        data = cur.fetchone()
        if data is not None:
            next_time = data[0]
            img_data = np.frombuffer(data[1], dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
            detections = person_detection.detection.process_image(yolo_detector, img, next_time)
            if len(detections)==0:
                cur.execute(del_query, (next_time,))
                db.commit()
                track.Track.check_tracks_ttl(db, tracks, next_time, max_ttl)
                continue
 
            for det in detections:
                x1, y1, x2, y2 = det.get_xyxy()
                crop = img[y1:y2, x1:x2]
                det.crop = crop
                pil_crop = Image.fromarray(crop,mode='RGB')
                pil_crop = transform_pipeline(pil_crop).unsqueeze(0)
                with torch.no_grad():
                    token = encoder(pil_crop).flatten()
                det.token = token
            if len(tracks):
                track.Track.run_matching(tracks, detections, db)
            for det in detections:
                if not det.track_id:
                    tracks[ID] = track.Track(det, max_x=img.shape[1], max_y=img.shape[0])
                    ID+=1
            track.Track.check_tracks_ttl(db, tracks, next_time, max_ttl)
        else:
            track.Track.check_tracks_ttl(db, tracks, datetime.now(), max_ttl)
            if len(tracks):
                time.sleep(3)
            else:
                break

