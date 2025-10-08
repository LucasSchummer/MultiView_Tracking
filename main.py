from scripts.stitch import run_stitching
from scripts.detect import run_detection
from scripts.track import run_tracking
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-purpose computer vision toolkit for multi-view settings")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----------------------------
    # Stitch subcommand
    # ----------------------------
    parser_stitch = subparsers.add_parser("stitch", help="Stitch a set of frames")
    parser_stitch.add_argument("--frame_folder", required=True,
                        help="Path to the frame folder")
    parser_stitch.add_argument("--output", required=True,
                        help="Path to the output folder")
    parser_stitch.add_argument("--out_format", default="jpg", choices=["jpg", "png", "tiff"],
                        help="Format of the output images")
    parser_stitch.add_argument("--ref_frame", default=None,
                        help="Path to the folder containing the reference frame to use to estimate stitching parameters")
    parser_stitch.add_argument("--detector", default="orb", choices=["orb", "sift", "brisk", "akaze"],
                        help="Keypoints detector to use")
    parser_stitch.add_argument("--warper", default="spherical", choices=["spherical", "cylindrical", "plane", "affine", "fisheye", "stereographic"],
                        help="Warper type")
    parser_stitch.set_defaults(func=run_stitching)

    # ----------------------------
    # Detect subcommand (placeholder)
    # ----------------------------
    parser_detect = subparsers.add_parser("detect", help="Run object detection on a set of frames or video")
    parser_detect.add_argument("--input", required=True,
                        help="Path to the video or frame folder")
    parser_detect.add_argument("--detector", default="yolo11s",
                        help="YOLO model to use")
    parser_detect.add_argument("--output", required=True,
                        help="Path to the output folder")
    parser_detect.add_argument("--out_format", default="jpg", choices=["jpg", "png", "tiff", "mp4"],
                        help="Output format for detections (images or video)")
    parser_detect.add_argument("--out_fps", default=30, type=int,
                        help="Framerate of the output (only for video)")
    parser_detect.add_argument("--tile_mode", default="simple", choices=["simple", "line", 'tile'],
                        help="Tiling pattern to use")
    parser_detect.add_argument("--tile_size", default=0, type=int,
                        help="Tile size to use (only for mode \"tile\")")
    parser_detect.add_argument("--min_ov_ratio", default=0.2, type=float,
                        help="Minium overlap ratio between adjacent tiles")
    parser_detect.add_argument("--iou_thresh", default=0.5, type=float,
                        help="IoU threshold to filter detections (only for modes \"line\" and \"tile\")")
    parser_detect.add_argument("--labels", dest="do_labels", action="store_true",
                        help="Include labels on annotations (default: True)")
    parser_detect.add_argument("--no_labels", dest="do_labels", action="store_false",
                        help="Do not include labels on annotations")
    parser_detect.set_defaults(do_labels=True)
    parser_detect.set_defaults(func=run_detection)

    # ----------------------------
    # Track subcommand (placeholder)
    # ----------------------------
    parser_track = subparsers.add_parser("track", help="Run object tracking")
    parser_track.add_argument("--input", required=True,
                        help="Path to the video or frame folder")
    parser_track.add_argument("--tracker", default="ByteTrack", choices=["ByteTrack", "DeepSort"],
                        help="Tracker to use")
    parser_track.add_argument("--detector", default="yolo11s",
                        help="YOLO model to use")
    parser_track.add_argument("--output", required=True,
                        help="Path to the output folder")
    parser_track.add_argument("--out_format", default="jpg", choices=["jpg", "png", "tiff", "mp4"],
                        help="Output format for detections (images or video)")
    parser_track.add_argument("--out_fps", default=30, type=int,
                        help="Framerate of the output (only for video)")
    parser_track.add_argument("--tile_mode", default="simple", choices=["simple", "line", 'tile'],
                        help="Tiling pattern to use")
    parser_track.add_argument("--tile_size", default=0, type=int,
                        help="Tile size to use (only for mode \"tile\")")
    parser_track.add_argument("--min_ov_ratio", default=0.2, type=float,
                        help="Minium overlap ratio between adjacent tiles")
    parser_track.add_argument("--iou_thresh", default=0.5, type=float,
                        help="IoU threshold to filter detections (only for modes \"line\" and \"tile\")")
    parser_track.add_argument("--labels", dest="do_labels", action="store_true",
                        help="Include labels on annotations (default: True)")
    parser_track.add_argument("--no_labels", dest="do_labels", action="store_false",
                        help="Do not include labels on annotations")
    parser_track.set_defaults(do_labels=True)
    parser_track.set_defaults(func=run_tracking)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# python main.py stitch --frame_folder "data/frame_sets/Test" --output "data/stitched_sets/Test" --ref_frame "../Prototype Raspberry/rasp_data/frame_sets/Ref40" --out_format png
# python main.py stitch --frame_folder "data/frame_sets/Mairie" --output "data/stitched_sets/Mairie" 
# python main.py detect --input "data/videos/castagnoles.mp4" --detector fishes --output "data/yolo" --out_format mp4
# python main.py detect --input data/stitched_sets/Mairie --output data/yolo/Mairie --out_format tiff --tile_mode tile --tile_size 1500
# python main.py track --input "data/stitched_sets/Parc"  --output "data/tracking" --out_format mp4 --out_fps 3 --tile_mode line