import cv2 as cv
import os
import glob

def frame_iterator(path):
    """Yield frames one by one from either a folder or a video."""
    
    if os.path.isdir(path):
        # --- Folder mode ---
        image_paths = sorted(glob.glob(os.path.join(path, "*.*")))
        for p in image_paths:
            frame = cv.imread(p)
            if frame is not None:
                yield frame
            else:
                print(f"Warning: could not read {p}")

    elif os.path.isfile(path):
        # --- Video mode ---
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

        cap.release()
    else:
        raise ValueError(f"Invalid input path: {path}")
    

import supervision as sv
detections = sv.Detections.empty()
