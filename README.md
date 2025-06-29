# Player Tracking ML Project

This project implements a multi-object tracking system for identifying and tracking individual players across video frames in sports footage.

## Features

- **YOLO Detection**: Uses pre-trained YOLOv11 model for player detection
- **Persistent ID Tracking**: Assigns unique, persistent IDs to each player
- **GPU Acceleration**: Supports CUDA for faster processing
- **Video Output**: Generates tracked video with bounding boxes and player IDs

## Project Structure

```
player_tracking_ml/
├── main.py              # Main tracking logic
├── tracker.py           # CentroidTracker implementation
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Place your YOLO model (best.pt) here
├── input/              # Place your input video (15sec_input_720p.mp4) here
└── output/             # Output video will be saved here
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Files
- Place your YOLO model file (`best.pt`) in the `models/` directory
- Place your input video (`15sec_input_720p.mp4`) in the `input/` directory

### 3. Run the Tracking
```bash
python main.py
```

The processed video with player tracking will be saved as `output/tracked_output.mp4`.

## How It Works

1. **Detection**: YOLO model detects players in each frame
2. **Tracking**: CentroidTracker assigns and maintains unique IDs for each player
3. **Visualization**: Bounding boxes with persistent IDs are drawn on each frame
4. **Output**: Final video with tracking information is saved

## Technical Details

- **Tracker**: Uses centroid-based tracking with distance matching
- **ID Persistence**: Players maintain their IDs across frames and re-appearances
- **GPU Support**: Automatically detects and uses CUDA if available
- **Performance**: Optimized for real-time processing with GPU acceleration

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- YOLO model file (`best.pt`)
- Input video file (`15sec_input_720p.mp4`) 