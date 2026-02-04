"""
Tracking Module
Implements ByteTrack for multi-object tracking.
"""

import numpy as np
from collections import deque
import scipy.linalg
from scipy.optimize import linear_sum_assignment

# Parameters
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
TRACK_BUFFER = 60 # Number of frames to keep lost tracks
MATCH_THRESH = 0.8
ASPECT_RATIO_THRESH = 1.6
MIN_BOX_AREA = 10

class KalmanFilter:
    """
    Available Kalman Filter implementation for bounding box tracking.
    State: [x, y, a, h, vx, vy, va, vh]
    x, y: Center position
    a: Aspect ratio (w/h)
    h: Height
    """
    def __init__(self):
        ndim, dt = 4, 1.
        
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        
        return new_mean, new_covariance

class Track:
    _count = 0
    
    def __init__(self, tlwh, score):
        # tlwh: top-left x, top-left y, width, height
        self.track_id = 0
        self.is_activated = False
        self.state = 0 # 0: New, 1: Tracked, 2: Lost, 3: Removed
        
        self.score = score
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        
        # Kalman Filter
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(self.tlwh_to_xyah(tlwh))
        
    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
        
    @property
    def tlbr(self):
        """Get current position in tlbr format (x1, y1, x2, y2)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
        
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to state representation (center x, center y, aspect ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
        
    def activate(self, kalman_filter, frame_id):
        """Start a new track."""
        self.track_id = self.next_id()
        self.kf = kalman_filter
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = 1 # Tracked
        self.is_activated = True
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track."""
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = 1 # Tracked
        self.is_activated = True
        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()
            
    def predict(self):
        """Predict next state using Kalman Filter."""
        if self.state != 1: # Only predict for tracked objects if needed, but usually we predict all
            pass
            
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        
    def update(self, new_track, frame_id):
        """Update state with new measurement."""
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = new_track.score
        
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = 1 # Tracked
        self.is_activated = True

    def mark_lost(self):
        self.state = 2 # Lost
        
    def mark_removed(self):
        self.state = 3 # Removed

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

class ByteTracker:
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.det_thresh = TRACK_HIGH_THRESH + 0.1
        self.buffer_size = int(frame_rate / 30.0 * TRACK_BUFFER)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
    def update(self, output_results, img_info, img_size):
        """
        Update tracks with new detections.
        
        output_results: tensor/array [[x1, y1, x2, y2, score, class], ...]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Separate detections into high and low confidence
        if len(output_results) == 0:
            detections = []
            detections_second = []
        else:
            # Assume Output is [x1, y1, x2, y2, score]
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            
            remain_inds = scores > TRACK_HIGH_THRESH
            inds_low = scores > TRACK_LOW_THRESH
            inds_high = scores < TRACK_HIGH_THRESH
            
            inds_second = np.logical_and(inds_low, inds_high)
            
            # High confidence
            dets_first = bboxes[remain_inds]
            scores_first = scores[remain_inds]
            detections = [Track(Track.tlbr_to_tlwh(det), s) for det, s in zip(dets_first, scores_first)]
            
            # Low confidence
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            detections_second = [Track(Track.tlbr_to_tlwh(det), s) for det, s in zip(dets_second, scores_second)]

        # 2. Predict current tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        strack_pool = join_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict using Kalman Filter
        for track in strack_pool:
            track.predict()

        # 3. Association with High Confidence Detections (IoU)
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=MATCH_THRESH)
        
        for itracked, ldet in matches:
            track = strack_pool[itracked]
            det = detections[ldet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # 4. Association with Low Confidence Detections (Leftover tracks)
        # Only for tracks that were already tracked (not lost) and unmatched
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, ldet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[ldet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # 5. Deal with Unmatched Tracks
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 2:
                track.mark_lost()
                lost_stracks.append(track)
                
        # 6. Deal with New Tracks (Unmatched High Conf Detections)
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            
        # 7. Post-processing state
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = join_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = join_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Remove dead tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                self.removed_stracks.append(track)
        
        # Clean lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 2]
        
        return self.tracked_stracks

    def predict(self):
        """
        Predict only (for frames where detection is skipped).
        Updates Kalman Filter predictions but does not change track states.
        """
        self.frame_id += 1
        
        # Join tracked and lost (to keep predicting lost ones just in case)
        # But for visualization we typically only care about tracked ones
        
        # Predict tracked tracks
        for track in self.tracked_stracks:
            track.predict()
            
        # Predict lost tracks (optional, but good if we want to recover them later)
        for track in self.lost_stracks:
            track.predict()
            
        return self.tracked_stracks

# --- Helpers ---

def join_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(pairs[0]), list(pairs[1])
    for a, b in zip(dupa, dupb):
        timea = stracksa[a].frame_id - stracksa[a].start_frame
        timeb = stracksb[b].frame_id - stracksb[b].start_frame
        if timea > timeb:
            dupb.append(b)
        else:
            dupa.append(a)
    res_a = [t for i, t in enumerate(stracksa) if not i in dupa]
    res_b = [t for i, t in enumerate(stracksb) if not i in dupb]
    return res_a, res_b

def iou_distance(atracks, btracks):
    if (len(atracks) == 0 and len(btracks) == 0):
        return np.zeros((0,0))
    if len(atracks) == 0:
        return np.zeros((0, len(btracks)))
        
    ious = np.zeros((len(atracks), len(btracks)))
    for i, t in enumerate(atracks):
        t_tlbr = t.tlbr
        for j, b in enumerate(btracks):
            b_tlbr = b.tlbr
            
            # IoU calc
            xA = max(t_tlbr[0], b_tlbr[0])
            yA = max(t_tlbr[1], b_tlbr[1])
            xB = min(t_tlbr[2], b_tlbr[2])
            yB = min(t_tlbr[3], b_tlbr[3])
            
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (t_tlbr[2] - t_tlbr[0] + 1) * (t_tlbr[3] - t_tlbr[1] + 1)
            boxBArea = (b_tlbr[2] - b_tlbr[0] + 1) * (b_tlbr[3] - b_tlbr[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            ious[i, j] = 1 - iou # Distance
            
    return ious

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
    matches, unmatched_a, unmatched_b = [], [], []
    
    # Scipy linear assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] > thresh:
            unmatched_a.append(row)
            unmatched_b.append(col)
        else:
            matches.append((row, col))
            
    # Add unmatched rows and cols that weren't in optimal mapping
    for row in range(cost_matrix.shape[0]):
        if row not in row_ind:
            unmatched_a.append(row)
            
    for col in range(cost_matrix.shape[1]):
        if col not in col_ind:
            unmatched_b.append(col)
            
    return matches, unmatched_a, unmatched_b

# Helper for Track
Track.tlbr_to_tlwh = lambda tlbr: np.asarray([tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]])
