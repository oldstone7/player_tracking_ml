from ultralytics import YOLO
import cv2
import os
import torch

# === Settings ===
PLAYER_CLASS_ID = 2  # try 1 or 2
CONFIDENCE_THRESHOLD = 0.4
DEBUG = False

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load model ===
model_path = os.path.join('models', 'best.pt')
model = YOLO(model_path).to(device)

# === Load video ===
video_path = os.path.join('input', '15sec_input_720p.mp4')
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Output setup ===
os.makedirs('output', exist_ok=True)
out = cv2.VideoWriter('output/clean_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Inference ===
    results = model(frame)
    detections = results[0].boxes

    for box in detections:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id == PLAYER_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if DEBUG and frame_count < 5:
        print(f"Frame {frame_count}: drew clean boxes.")

cap.release()
out.release()
print("✅ Clean output saved to 'output/clean_output.mp4'")
