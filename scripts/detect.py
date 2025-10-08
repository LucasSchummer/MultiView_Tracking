import os
import cv2 as cv
from tqdm import tqdm
from modules.utils import frame_iterator, get_total_frames
from modules.detector import Detector

def run_detection(args):

    input = args.input
    output = args.output
    out_format = args.out_format
    out_fps = args.out_fps
    model = args.detector
    tile_mode = args.tile_mode
    tile_size = args.tile_size
    min_ov_ratio = args.min_ov_ratio
    iou_thresh = args.iou_thresh
    do_labels = args.do_labels

    if os.path.isdir(input):
        filename = input.split("/")[-1]
    else:
        filename = input.split("/")[-1].split(".")[0]

    first_frame = next(frame_iterator(input))
    height, width = first_frame.shape[:2]
    scale_factor = min(width, height) / 1000

    model = Detector(model, tile_mode, tile_size, min_ov_ratio, iou_thresh, scale_factor, do_labels)

    os.makedirs(output, exist_ok=True)

    if out_format == "mp4": out = cv.VideoWriter(f"{output}/YOLO_{filename}.mp4", cv.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))

    tot_frames = get_total_frames(input)
    for i, frame in enumerate(tqdm(frame_iterator(input), total=tot_frames, desc="Processing frames", unit="frame", colour="green")):

        annotated_frame = model(frame)

        if out_format == 'mp4':
            out.write(annotated_frame)
        else:
            cv.imwrite(f"{output}/YOLO_{filename}_{i}.{out_format}", annotated_frame)

    
    if out_format == "mp4": out.release()


