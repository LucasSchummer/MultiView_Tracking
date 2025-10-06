import supervision as sv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.detector import Detector
from modules.annotator import Annotator


class Tracker():

    def __init__(self, tracker="ByteTrack", detector="yolo11s", tile_mode="simple", tile_size=0, min_ov_ratio=.2, iou_thresh=.5, scale_factor=1, do_labels=True):
        
        self.detector = Detector(detector, tile_mode, tile_size, min_ov_ratio, iou_thresh)
        self.annotator = Annotator(scale_factor, do_labels)

        if tracker == "ByteTrack":
            self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8) # Default parameters
        else:
            self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=.7, max_cosine_distance=.3)


    def __call__(self, frame):

        detections, labels = self.detector.detect(frame, conf_thresh=.3)
        detections, labels = self.update(detections, labels, frame)

        return self.annotator(frame, detections, labels)


    def update(self, detections, labels, frame=None):
        
        # Update ByteTrack
        if isinstance(self.tracker, sv.ByteTrack):

            detections = self.tracker.update_with_detections(detections)
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        # Update DeepSort
        else:
            
            raw_detections = []
            for xyxy, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                raw_detections.append(([x1, y1, w, h], float(conf), int(cls)))

            # Run DeepSORT update
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)


            boxes, ids, classes, confs = [], [], [], []
            for track in tracks:

                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                tlbr = track.to_tlbr()  # [x1, y1, x2, y2]
                boxes.append(tlbr)
                ids.append(track.track_id)
                classes.append(track.det_class)
                confs.append(track.det_conf)

            if len(boxes) == 0:
                detections = sv.Detections.empty()
                labels = []
            else:
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    confidence=np.array(confs),
                    class_id=np.array(classes),
                    tracker_id=np.array(ids, dtype=int)
                )

                labels = [f"#{track_id}" for track_id in detections.tracker_id]

        return detections, labels
    
