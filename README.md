# Multi-View Tracking

A modular Python toolkit for **multi-view computer vision**, featuring:
- 🪄 **Stitching** of multiple image views  
- 🔍 **Object detection** with YOLO  
- 🎯 **Object tracking** with ByteTrack or DeepSort 

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/LucasSchummer/MultiView_Tracking.git
cd MultiView_Tracking

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## ⚙️ Compatibility

💡 The code has been tested with **Python 3.10**.  
Using other versions may cause unexpected issues.

---

## 🚀 Usage

### 🧵 Stitching
Stitch a set of images from multiple viewpoints into a single panorama.

```bash
python main.py stitch \
    --frame_folder path/to/frames \
    --output path/to/output

```

⚠️ Your frame folder should only contain folders, one per frame. Each individual subfolder should contain the different views of a given scene

**Optional arguments :**
```bash
    --ref_frame : Reference frame folder for parameter estimation.
    --out_format : Output format ('jpg', 'png', 'tiff'). Default : 'jpg'
    --detector : Keypoint detector ('orb', 'sift', 'brisk', 'akaze'). Default : 'orb'
    --warper : Warper type ('spherical', 'cylindrical', 'plane', 'affine', 'fisheye', 'stereographic'). Default: 'spherical'
```

### 🎯 Detection
Run object detection on a video or folder of frames.

```bash
python main.py detect \
    --input path/to/video_or_frames \
    --output path/to/output \
```

**Optional arguments :**
```bash
    --out_format : Output format ('jpg', 'png', 'tiff', 'mp4'). Default : 'jpg'
    --out_fps : Output framerate (if mp4). Default : 30
    --detector : YOLO model to use (yoloxx or 'fishes'). Default : yolo11s
    --tile_mode : Tiling pattern for each frame ('simple', 'line' or 'tile'). Default : 'simple'
    --tile_size : Tile size to use (px) (Only for mode 'tile'). 
    --min_ov_ratio : Minimum overlap ratio between adacent tiles. Default : 0.2
    --iou_thresh : IoU threshold to filter detections (only for modes 'line' and 'tile'). Default : 0.5
    --labels : Include prediction labels on the output. Default : True
    --no_labels : Do not include prediction labels on the output. Default : False
    
```

### 🧍 Tracking
Track detected objects across frames using ByteTrack or DeepSort.

```bash
python main.py track \
    --input path/to/video_or_frames \
    --output path/to/output \
```

**Optional arguments :**
```bash
    --tracker : Tracker to use ('ByteTrack', 'DeepSort'). Default : 'ByteTrack'
    --out_format : Output format ('jpg', 'png', 'tiff', 'mp4'). Default : 'jpg'
    --out_fps : Output framerate (if mp4). Default : 30
    --detector : YOLO model to use (yoloxx or 'fishes'). Default : yolo11s
    --tile_mode : Tiling pattern for each frame ('simple', 'line' or 'tile'). Default : 'simple'
    --tile_size : Tile size to use (px) (Only for mode 'tile'). 
    --min_ov_ratio : Minimum overlap ratio between adacent tiles. Default : 0.2
    --iou_thresh : IoU threshold to filter detections (only for modes 'line' and 'tile'). Default : 0.5
    --labels : Include prediction labels on the output. Default : True
    --no_labels : Do not include prediction labels on the output. Default : False
    
```

---

### 🛠️ Example Workflow

#### 1. **Stitching**

**Stitch frames** from multiple cameras into panoramas :
   ```bash
   python main.py stitch --frame_folder data/ex_frame_folder --output stitched/ex
   ```

**Stitch frames** using a reference frame to estimate parameters :
   ```bash
   python main.py stitch --frame_folder data/ex_frame_folder --output stitched/ex --ref_frame data/ref_frame_folder
   ```

**Stitch frames** using a different keypoint detector and warper :
   ```bash
   python main.py stitch --frame_folder data/ex_frame_folder --output stitched/ex --detector sift --warper cylindrical
   ```

#### 2. **Object detection**

**Detect objects** on frames :
   ```bash
   python main.py detect --input stitched/ex --output detections/ex
   ```

**Detect objects** with a custom YOLO model (stored in 'models/') :
   ```bash
   python main.py detect --input stitched/ex --output detections/ex --detector fishes
   ```

**Detect objects** using tiling :
   ```bash
   python main.py detect --input stitched/ex --output detections/ex --tile_mode tile --tile_size 1000
   ```

**Detect objects** and output a video :
   ```bash
   python main.py detect --input stitched/ex --output detections/ex --out_format mp4 --out_fps 3
   ```

#### 3. **Multi-object Tracking**

**Track objects** across frames :
   ```bash
   python main.py track --input stitched/ex --output tracking/ex
   ```

**Track objects** on a video :
   ```bash
   python main.py track --input stitched/ex.mp4 --output tracking/ex --out_format mp4
   ```

**Track objects** with DeepSort :
   ```bash
   python main.py track --input stitched/ex --output tracking/ex --tracker DeepSort
   ```

---

📄 For detailed parameter options, run:
```bash
python main.py --help
```
or:
```bash
python main.py <command> --help
```
(e.g., `python main.py stitch --help`)