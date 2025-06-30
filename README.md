# Player Re-identification & Tracking – Assessment Submission

##  Project Goal
The objective of this assessment was to perform **player re-identification and tracking** from a football video using a custom YOLO model (`best.pt`). The aim was to ensure each player retains a consistent ID throughout the video. The solution was built using Python, OpenCV, and PyTorch, leveraging GPU acceleration (CUDA) for faster performance.

---

##  What I Tried

### 1. **Centroid Tracker (Basic)**
I started with a simple **Centroid Tracker**, which keeps track of object IDs by comparing the distance between detected player centroids across frames.  
This method is:
- Lightweight
- Easy to implement
- But fragile when players move quickly or occlude each other.

### 2. **IoU-Based Tracker**
Then, I attempted to integrate an **IoU (Intersection over Union)** tracker to improve stability. This compared the overlap between bounding boxes across frames to associate detections.  
However:
- It worked in some cases but still had ID switches when players overlapped or moved erratically.
- Integration was messy and less reliable than expected.

### 3. **Combining Centroid + IoU** *(Idea)*
I briefly explored combining both approaches — using centroid distance for fallback when IoU failed — but I didn’t have enough time to refine this hybrid approach due to the deadline.

### 4. **Deep SORT / Re-ID** *(The approach was in the branch `reid`, working one was in `main` branch)*
Finally, I tried integrating **Deep SORT** for re-identification using embeddings. However, due to compatibility issues between YOLOv8’s tensor structure and Deep SORT’s expected input format, the implementation broke down.  
Despite troubleshooting, the re-ID integration remained buggy and I couldn’t fix it before the deadline.

---

##  Final Working Solution

I went back to a reliable and minimal approach:  
- Uses YOLOv11 model (`best.pt`) to detect class `2` (players)
- Calculates centroids of detected bounding boxes
- Tracks objects based on proximity of centroids across frames
- Draws bounding boxes and labels with unique IDs



---
## ⚡ CUDA Acceleration

This project checks for a CUDA-enabled GPU and automatically runs on GPU if available. This significantly speeds up inference, making it suitable for real-time or near real-time applications. Run the below command to install your CUDA Toolkit to use NVIDIA drivers (takes around 2.8 gb but worth running)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

## Requirements

- Python 3.8+
- CUDA-compatible GPU (highly recommended, for faster processing)
- YOLO model file (`models/best.pt`)
- Input video file (`input/15sec_input_720p.mp4`)

## Installation

```bash
# Clone the repo
git clone https://github.com/your_username/player_tracking_ml.git
cd player_tracking_ml

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Program

```bash
python main.py
```

## Project Structure

```
player_tracking_ml/
├── input/
│   └── 15sec_input_720p.mp4
├── models/
│   └── best.pt
├── output/
│   └── tracked_output.mp4  # (generated after running)
├── tracker.py
├── main.py
├── requirements.txt
└── README.md
```


## Limitations
- The tracking is not 100% accurate (some ID jumps or losses when players overlap).
- No advanced re-identification logic (like Deep SORT embeddings).
- Bounding box consistency can be shaky when players move fast or overlap.

---

## Note
Although I come from a web development background, this assessment helped me stretch into new ML territories. I used a combination of research, experimentation, and support from open-source resources and LLMs to build this.