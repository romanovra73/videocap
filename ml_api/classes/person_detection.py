import torch
import numpy as np

class detection:
    def __init__(self, time, box):
        self.time = time
        self.box = box
        self.token = torch.empty(512, dtype=torch.float)
        self.track_id = ''
        self.crop = np.empty((50,200))

    def get_xyxy(self):
        a1,b1,a2,b2 = self.box[0].astype(np.int32)
        x1 = min(a1, a2)
        y1 = min(b1, b2)
        x2 = max(a1, a2)
        y2 = max(b1, b2)
        return np.array([x1,y1,x2,y2]).astype(np.int32)

    def get_xyrh(self):
        x1, y1, x2, y2 = self.box[0]
        result = np.array([
    	    (x1+x2)/2,
    	    (y1+y2)/2,
    	    abs(x1-x2)/abs(y1-y2),
    	    abs(y1-y2)
        ])
        return result

    @staticmethod
    def process_image(detector, img, time, min_height=200):
        new_detections = detector(img)[0]
        res = []
        for box in new_detections.boxes:
            if box.cls[0]>0 or box.conf[0]<=0.5:  ## пропускаем не людей и низкую вероятность
                continue
            coord = box.xyxy.numpy().astype(np.int32)
            if abs(coord[0][1]-coord[0][3])<min_height:
                continue
            det = detection(time, coord)
            res.append(det)
        return res
