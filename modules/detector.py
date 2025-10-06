import cv2 as cv
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

from modules.annotator import Annotator

class Detector():

    def __init__(self, model="yolo11s.pt", tile_mode="simple", tile_size=0, min_ov_ratio=.2, iou_thresh=.5, scale_factor=1, do_labels=True):
        
        self.tiler = Tiler(tile_mode, tile_size, min_ov_ratio)
        self.model = YOLO(f'models/{model}.pt')
        self.iou_thresh = iou_thresh
        
        self.annotator = Annotator(scale_factor, do_labels)


    def __call__(self, image, conf_thresh=.3):
        
        # Image can be path or frame
        img = cv.imread(image) if type(image) == str else image

        detections, labels = self.detect(img, conf_thresh)

        return self.annotator(img, detections, labels)
    

    def detect(self, image, conf_thresh):

        # Image can be path or frame
        img = cv.imread(image) if type(image) == str else image

        tiles = self.tiler.tile_image(img)

        all_detections = []
        for tile, x_off, y_off in tiles:
            
            results = self.model(tile, conf=conf_thresh, verbose=False)[0]
            if results.boxes is not None:
                
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Map to original coordinates
                mapped = np.stack([
                    boxes[:, 0] + x_off,
                    boxes[:, 1] + y_off,
                    boxes[:, 2] + x_off,
                    boxes[:, 3] + y_off,
                    confs,
                    classes
                ], axis=1)
                
                all_detections.append(mapped)

        detections = self.merge_detections(all_detections)

        labels = [
            f"{self.model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]


        return detections, labels
    

    def merge_detections(self, detections):

        # dets is a list of arrays from each tile: each of shape (n_i, 6)
        dets = [d for d in detections if d is not None and len(d) > 0]

        if not dets:
            return sv.Detections.empty()

        all_boxes = np.concatenate(dets, axis=0)

        # Convert to torch tensor (batch of 1 image)
        pred_tensor = torch.tensor(all_boxes, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N, 6)

        # Apply NMS
        nms_result = non_max_suppression(pred_tensor, iou_thres=self.iou_thresh)

        # Extract the result for this image (since batch=1)
        merged = nms_result[0].numpy() if len(nms_result[0]) > 0 else []

        if len(merged) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=merged[:, 0:4],
                confidence=merged[:, 4],
                class_id=merged[:, 5].astype(int),
            )

        return detections
    


class Tiler():

    def __init__(self, tile_mode, tile_size, min_ov_ratio):
        
        self.tile_mode = tile_mode
        self.tile_size = tile_size
        self.min_ov_ratio = min_ov_ratio


    def tile_image(self, img):

        tiles = []
        h, w = img.shape[:2]

        if self.tile_mode == "tile":

            # Compute the number of tiles fitting and adding one to adapt stride by increasing overlap_ratio
            n_tiles_x, stride_x = self.get_n_tiles(w, tile_size)
            n_tiles_y, stride_y = self.get_n_tiles(h, tile_size)

            for y in range(0, h - tile_size + 1, stride_y):
                for x in range(0, w - tile_size + 1, stride_x):
                    tile = img[y:y+tile_size, x:x+tile_size]
                    tiles.append((tile, x, y))

        elif self.tile_mode == "line":

            tile_size = min(h, w)
            landscape = w > h

            # Number of tiles (only in the tiling direction)
            if landscape:
                
                n_tiles, stride = self.get_n_tiles(w, h)
                for i in range(n_tiles):
                    x = i * stride
                    tile = img[0:tile_size, x:x+tile_size]
                    tiles.append((tile, x, 0))
            else:
                
                n_tiles, stride = self.get_n_tiles(h, w)
                for i in range(n_tiles):
                    y = i * stride
                    tile = img[y:y+tile_size, 0:tile_size]
                    tiles.append((tile, 0, y))

        else:

            tiles.append((img, 0, 0))
        
        return tiles

    
    def get_n_tiles(self, tot_pix, tile_size):

        n_tiles = 1
        while True:
            if n_tiles*tile_size < tot_pix:
                n_tiles += 1
                continue
            overlap_ratio = ((tile_size * n_tiles) - tot_pix) / (n_tiles - 1) / tile_size if n_tiles > 1 else tile_size
            if overlap_ratio < self.min_ov_ratio:
                n_tiles += 1
                continue

            break

        stride = int(tile_size - ((tile_size * n_tiles) - tot_pix) / (n_tiles - 1)) if n_tiles > 1 else tile_size

        return n_tiles, stride
