import glob
import os
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm
from modules.utils import frame_iterator 
from modules.detector import Detector

def main(input, output, out_format, out_fps, model, tile_mode, tile_size, min_ov_ratio, iou_thresh, do_labels):

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


    for i, frame in enumerate(tqdm(frame_iterator(input))):

        annotated_frame = model(frame)

        if out_format == 'mp4':
            out.write(annotated_frame)
        else:
            cv.imwrite(f"{output}/YOLO_{filename}_{i}.{out_format}", annotated_frame)

    
    if out_format == "mp4": out.release()



def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Detect objects on a video or set of frames")
    parser.add_argument("--input", required=True,
                        help="Path to the video or frame folder")
    parser.add_argument("--model", default="yolo11s",
                        help="YOLO model to use")
    parser.add_argument("--output", required=True,
                        help="Path to the output folder")
    parser.add_argument("--output_format", default="jpg", choices=["jpg", "png", "tiff", "mp4"],
                        help="Output format for detections (images or video)")
    parser.add_argument("--output_fps", default=30, type=int,
                        help="Framerate of the output (only for video)")
    parser.add_argument("--tile_mode", default="simple", choices=["simple", "line", 'tile'],
                        help="Tiling pattern to use")
    parser.add_argument("--tile_size", default=0, type=int,
                        help="Tile size to use (only for mode \"tile\")")
    parser.add_argument("--min_ov_ratio", default=0.2, type=float,
                        help="Minium overlap ratio between adjacent tiles")
    parser.add_argument("--iou_thresh", default=0.5, type=float,
                        help="IoU threshold to filter detections (only for modes \"line\" and \"tile\")")
    parser.add_argument("--labels", dest="do_labels", action="store_true",
                        help="Include labels on annotations (default: True)")
    parser.add_argument("--no_labels", dest="do_labels", action="store_false",
                        help="Do not include labels on annotations")
    
    parser.set_defaults(do_labels=True)
    

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args.input, args.output, args.output_format, args.output_fps, args.model, args.tile_mode, args.tile_size, args.min_ov_ratio, args.iou_thresh, args.do_labels)

# python detect.py --input "data/stitched_sets/Parc" --output "data/yolo/Parc" 
# python detect.py --input "data/videos/castagnoles.mp4" --model fishes --output "data/yolo/castagnoles" --output_format mp4
# python detect.py --input data/stitched_sets/Mairie --output data/yolo/Mairie --tile_mode line --min_ov_ratio 0.3 --no_labels
