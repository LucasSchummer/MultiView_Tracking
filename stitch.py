import glob
import os
import argparse
import cv2 as cv
from modules.stitcher import Stitcher
from matplotlib import pyplot as plt

def main(frame_folder, output, out_format, ref_frame, warper_type):

    folders = os.listdir(frame_folder)
    frames = [glob.glob(f'{frame_folder}/{folder}/*.jpg') for folder in folders]

    os.makedirs(output, exist_ok=True)

    if ref_frame:
        ref_images = glob.glob(f"{ref_frame}/*")
    else:
        ref_images = frames[0]

    stitcher = Stitcher(warper_type).fit(ref_images)

    # Stitch each frame if the parameters have been estimated
    if stitcher.ready:

        for i, frame in enumerate(frames):
            stitched_frame = stitcher.stitch(frame)
            cv.imwrite(f"{output}/{folders[i]}.{out_format}", stitched_frame)

        print("Frames stitched successfully")

def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Stitch a set of frames.")
    parser.add_argument("--frame_folder", required=True,
                        help="Path to the frame folder")
    parser.add_argument("--output", required=True,
                        help="Path to the output folder")
    parser.add_argument("--output_format", default="jpg", choices=["jpg", "png", "tiff"],
                        help="Format of the output images")
    parser.add_argument("--ref_frame", default=None,
                        help="Path to the folder containing the reference frame to use to estimate stitching parameters")
    parser.add_argument("--detector", default="orb", choices=["orb", "sift", "brisk", "akaze"],
                        help="Keypoints detector to use")
    parser.add_argument("--warper", default="spherical", choices=["spherical", "cylindrical", "plane", "affine", "fisheye", "stereographic"],
                        help="Warper type")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    main(args.frame_folder, args.output, args.output_format, args.ref_frame, args.warper)

# python stitch.py --frame_folder "data/frame_sets/Test" --output "data/stitched_sets/Test" --ref_frame "../Prototype Raspberry/rasp_data/frame_sets/Ref40" --output_format tiff