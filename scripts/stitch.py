import glob
import os
import argparse
import cv2 as cv
from tqdm import tqdm
from modules.stitcher import Stitcher

def run_stitching(args):

    frame_folder = args.frame_folder
    output = args.output
    out_format = args.out_format
    ref_frame  = args.ref_frame
    detector = args.detector
    warper_type = args.warper

    folders = os.listdir(frame_folder)
    frames = [glob.glob(f'{frame_folder}/{folder}/*.jpg') for folder in folders]

    os.makedirs(output, exist_ok=True)

    if ref_frame:
        ref_images = glob.glob(f"{ref_frame}/*")
    else:
        ref_images = frames[0]

    # Estimate the stitching parameters on the ref frame and refine parameters with the first frame
    stitcher = Stitcher(detector, warper_type).fit(ref_images, refining_img_paths=frames[0])

    # Stitch each frame if the parameters have been estimated
    if stitcher.ready:

        for i, frame in enumerate(tqdm(frames, total=len(frames), desc="Processing frames", unit="frame", colour="green")):
            stitched_frame = stitcher.stitch(frame)
            cv.imwrite(f"{output}/{folders[i]}.{out_format}", stitched_frame)

        print("Frames stitched successfully")

