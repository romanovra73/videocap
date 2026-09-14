import numpy as np
import scipy
import scipy.linalg
import cv2
import torch
import torch.nn.functional as torchF
from datetime import datetime, timedelta
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, detection, max_x=1980, max_y=1020):
        x1, y1, x2, y2 = detection.get_xyxy()
        self.detected = detection.time
        self.associated = detection.time
        self.states = np.array([[(x1+x2)/2, (y1+y2)/2, abs(x1-x2)/abs(y1-y2), abs(y1-y2), 0, 0, 0, 0]])
        self.tokens = detection.token.unsqueeze(0)
        self._std_weight_position = 1./20
        self._std_weight_velocity = 1. /160
        self.filter_H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0]])
        self.filter_F = np.identity(8)
        xp = 2 * self._std_weight_position * abs(y1-y2)
        vp = 10 * self._std_weight_velocity * abs(y1-y2)
        self.filter_P = np.array([[xp**2, 0, 0, 0, 0, 0, 0, 0],
                                  [0, xp**2, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 2000, 0, 0, 0, 0, 0],
                                  [0, 0, 0, xp**2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, vp**2, 0, 0, 0],
                                  [0, 0, 0, 0, 0, vp**2, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 2000, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 2000]])
        self.filter_R = np.square(np.diag(np.array([xp,xp,1e-1,xp])))
        self.crops = []
        self.max_x = max_x
        self.max_y = max_y

    def get_Q_matrix(self):
        h = abs(self.states[-1][3])
        Q = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
            self._std_weight_velocity *h,
            self._std_weight_velocity *h,
            1e-5,
            self._std_weight_velocity *h,
        ]
        return(np.diag(np.square(np.array(Q))))

    def get_R_matrix(self):
        h = abs(self.states[-1][3])
        R = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        return(np.diag(np.square(np.array(R))))

    def predict_state(self, time):
        ms = (time - self.associated).total_seconds()*1000
        for i in range(4):
            self.filter_F[i][4+i] = ms
        predicted_state = np.dot(self.filter_F, self.states[-1])
        x1,y1,x2,y2 = self.xyrh_to_xyxy(predicted_state)
        fl = False
        if x1<0:
            fl = True
            x1=0
        if y1<0:
            fl = True
            y1=0
        if x2>self.max_x:
            fl = True
            x2 = self.max_x
        if y2>self.max_y:
            fl = True
            y2 = self.max_y
        if fl:
            predicted_state[0] = (x1+x2)/2
            predicted_state[1] = (y1+y2)/2
            predicted_state[2] = abs(x1-x2)/abs(y1-y2)
            predicted_state[3] = abs(y1-y2)
        predicted_P = np.linalg.multi_dot((self.filter_F, self.filter_P, self.filter_F.T)) + self.get_Q_matrix()

        return predicted_state, predicted_P

    def update(self, detection):
        predicted_state, predicted_P = self.predict_state(detection.time)
        proj_state = np.dot(self.filter_H, predicted_state)
        proj_P = np.linalg.multi_dot((self.filter_H, predicted_P, self.filter_H.T)) + self.get_R_matrix()

        cho_factor, lower = scipy.linalg.cho_factor(proj_P, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((cho_factor, lower), np.dot(predicted_P, self.filter_H.T).T, check_finite=False).T

        innovation = detection.get_xyrh() - proj_state
        new_state = predicted_state + np.dot(innovation, kalman_gain.T)
        new_P = predicted_P - np.linalg.multi_dot((kalman_gain, proj_P, kalman_gain.T))

        self.states = np.append(self.states, np.expand_dims(new_state, axis=0), axis=0)
        self.filter_P = new_P
        self.associated = detection.time
        self.tokens = torch.cat((self.tokens, detection.token.unsqueeze(0)), dim=0)

    def maha_distance(self, detection, predicted_box):
        print('Calc Maha distance:')
        mean = np.dot(self.filter_H, predicted_box)#np.average(self.states, axis=0))
        cov = np.linalg.multi_dot((self.filter_H, self.filter_P, self.filter_H.T)) + self.filter_R

        cf = np.linalg.cholesky(cov)
        measurement = detection.get_xyrh()
        d = measurement - mean
        z = scipy.linalg.solve_triangular(cf, d.T, lower=True, check_finite=False, overwrite_b=True)
        result = np.sum(z*z, axis=0)
        print(result)
        if (result>50):
            result = 1e+5
        return result

    def get_token_distance(self, token):
        print('Calc token distance:', self.tokens.shape, token.shape)
        if self.tokens.shape[0]==0:
            return 1
        cossim = torchF.cosine_similarity(self.tokens, token, dim=1)
        result = 1 - cossim.max().item()
        print(result)
        return result

    def add_crop(self, crop):
        enc, frame = cv2.imencode(".jpg", crop)
        if enc:
            self.crops.append(frame)
            return True
        return False

    def get_crops(self):
        return self.crops

    def get_xyxy(self, idx=-1):
        x,y,r,h = self.states[idx][0:4]
        w = r*h
        res = np.array([
            x - w/2,
            y - h/2,
            x + w/2,
            y + h/2
        ])
        return res.astype(np.int32)

    def xyxy_to_xyrh(self, xyxy):
        result = np.array([
            (xyxy[0] + xyxy[2]) / 2,
            (xyxy[1] + xyxy[3]) / 2,
            abs(xyxy[0] - xyxy[2]) / abs(xyxy[1] - xyxy[3]),
            abs(xyxy[1] - xyxy[3])
        ])
        return result

    def xyrh_to_xyxy(self, xyrh):
        width = xyrh[2] * xyrh[3]
        result = np.array([
            xyrh[0] - width / 2,
            xyrh[1] - xyrh[3] / 2,
            xyrh[0] + width / 2,
            xyrh[1] + xyrh[3] / 2,
        ])
        return result.astype(np.int32)

    def save(self, db):
        if len(self.states)<3:
            return True
        db_cur = db.cursor()
        query = '''INSERT INTO
    			cv.svrd_ch1_tracks (date, time_start, time_end, incoming, start_x, start_y, end_x, end_y)
    		   VALUES(
    			%s, %s, %s, %s, %s, %s, %s, %s
    		   ) RETURNING id'''
        db_cur.execute(query, (
    	    datetime.date(self.detected),
    	    datetime.time(self.detected),
    	    datetime.time(self.associated),
    	    True if self.states[-1][1]>800 and self.states[-1][5]>0 else False,
    	    int(self.states[0][0]),
    	    int(self.states[0][1]),
    	    int(self.states[-1][0]),
    	    int(self.states[-1][1])
    	))
        ins_id = db_cur.fetchone()
        if ins_id is None:
            db_cur.close()
            return False
        if len(self.crops):
            query = '''INSERT INTO
    			    cv.svrd_ch1_tracks_crops (track_id, date, crop_id, frame)
    			VALUES(
    			    %s, %s, %s, %s
    			)'''
            i = 0
            for crop in self.crops:
                bytes = crop.tobytes()
                db_cur.execute(query, (
    	    		    ins_id, 
    	    		    datetime.date(self.detected),
    	    		    i, 
    	    		    bytes
    	    	))
                i += 1
        db.commit()
        db_cur.close()
        return True

##----------------------------Static methods-----------------------------

    @staticmethod
    def get_tracks_list(tracks, associated, updated_tracks=[]):
        result = []
        for id in tracks.keys():
            if tracks[id].associated==associated and id not in updated_tracks:
                result.append(id)
        return result

    @staticmethod
    def get_maha_matching(tracks, list_tracks, list_detections, maha_weight=0.75):
        costs = []
        for det in list_detections:
            cols = []
            for id in list_tracks:
                new_state, new_P = tracks[id].predict_state(det.time)
                maha_dist = tracks[id].maha_distance(det, new_state)
                token_dist = tracks[id].get_token_distance(det.token)
                cols.append(maha_weight * maha_dist + (1 - maha_weight) * token_dist)
            costs.append(cols)
        costs_matrix = np.array(costs)
        if costs_matrix.shape[0]==1:
            a = (0,)
            b = (np.argmin(costs_matrix, axis=1)[0],)
            return costs_matrix, [a,b]
        if costs_matrix.shape[1]==1:
            b = (0,)
            a = (np.argmin(costs_matrix, axis=0)[0],)
            return costs_matrix, [a,b]
        return costs_matrix, linear_sum_assignment(costs_matrix)

    @staticmethod
    def iou(detection, track, threshold=0.35):
        a1,b1,a2,b2 = detection.get_xyxy()
        x1,y1,x2,y2 = track.get_xyxy()
        if a1>=x2 or a2<=x1 or b1>=y2 or b2<=y1:
            return 0
        intersection = (min(a2,x2)-max(a1,x1))*(min(b2,y2)-max(b1,y1))
        union = (a2-a1)*(b2-b1) + (x2-x1)*(y2-y1) - intersection
        result = intersection/union
        if result<threshold:
            return 1e+5
        return 1-result

    @staticmethod
    def get_iou_matching(tracks, list_tracks, list_detections, threshold=0.35):
        costs = []
        for det in list_detections:
            cols = []
            for id in list_tracks:
                cols.append(Track.iou(det, tracks[id], threshold))
            costs.append(cols)
        costs_matrix = np.array(costs)
        if costs_matrix.shape[0]==1:
            a = (0,)
            b = (np.argmin(costs_matrix, axis=1)[0],)
            return costs_matrix, [a,b]
        if costs_matrix.shape[1]==1:
            b = (0,)
            a = (np.argmin(costs_matrix, axis=0)[0],)
            return costs_matrix, [a,b]
        return costs_matrix, linear_sum_assignment(costs_matrix)

    @staticmethod
    def run_matching(tracks, detections, db, max_time=5, maha_weight=0.75, max_distance=200, iou_threshold=0.35):
        time_associated = []
        for id in tracks:
            ta = tracks[id].associated
            if ta not in time_associated:
                time_associated.append(ta)
        time_associated.sort(reverse=True)
        max_associated_time = time_associated[0]
        unmatched=[]
        updated_tracks = []
        for det in detections:
            if not det.track_id:
                unmatched.append(det)
        for ta in time_associated:
            if len(unmatched)==0:
                break
            if (unmatched[0].time-ta).total_seconds() < max_time:
                T = Track.get_tracks_list(tracks, ta, updated_tracks)
                cost_matrix, indices = Track.get_maha_matching(tracks, T, unmatched, maha_weight)
                for idx in range(len(indices[0])):
                    row = indices[0][idx]
                    col = indices[1][idx]
                    if cost_matrix[row][col]>max_distance or unmatched[row].track_id:
                        continue
                    track_id = T[col]
                    unmatched[row].track_id = track_id
                    tracks[track_id].update(unmatched[row])
                    updated_tracks.append(track_id)
                    x,y,r,h = unmatched[row].get_xyrh()
                    if (y>=600 and y<=750) or len(tracks[track_id].crops)==0:
                        tracks[track_id].add_crop(unmatched[row].crop)
                unmatched = []
                for det in detections:
                    if not det.track_id:
                        unmatched.append(det)
            else:
                Track.check_tracks_ttl(db, tracks, detections[0].time)
                break
        if len(unmatched)>0:
            T = Track.get_tracks_list(tracks, max_associated_time, updated_tracks)
            if len(T):
                cost_matrix, indices = Track.get_iou_matching(tracks, T, unmatched, iou_threshold)
                for idx in range(len(indices[0])):
                    row = indices[0][idx]
                    col = indices[1][idx]
                    if cost_matrix[row][col]>max_distance or unmatched[row].track_id:
                        continue
                    track_id = T[col]
                    unmatched[row].track_id = track_id
                    tracks[track_id].update(unmatched[row])
                    updated_tracks.append(track_id)
                    x,y,r,h = unmatched[row].get_xyrh()
                    if (y>=600 and y<=750) or len(tracks[track_id].crops)==0:
                        tracks[track_id].add_crop(unmatched[row].crop)

    @staticmethod
    def check_tracks_ttl(db, tracks, check_time, max_ttl=5):
        finished_tracks = []
        for id in tracks.keys():
            if tracks[id].associated + timedelta(seconds=max_ttl) < check_time:
                tracks[id].save(db)
                finished_tracks.append(id)
        for id in finished_tracks:
            del tracks[id]

    @staticmethod
    def get_last_track_time(db):
        db_cur = db.cursor()
        query = '''SELECT
    			date + MAX(time_end)
    		   FROM
    			cv.svrd_ch1_tracks
    		   WHERE
    			date=(SELECT MAX(date) FROM cv.svrd_ch1_tracks)
    		   GROUP BY date'''
        db_cur.execute(query)
        data = db_cur.fetchone()
        db_cur.close()
        return data[0]

    @staticmethod
    def get_next_track_id(db):
        db_cur = db.cursor()
        query = '''SELECT MAX(id)+1 AS new_id FROM svrd_ch1_tracks'''
        db_cur.execute(query)
        data = db_cur.fetchone()
        id = data[0]
        del data
        db_cur.close()
        if id is None:
            id = 1
        return id