from ultralytics import YOLO
import cv2
import os
import torch
from tracker import CentroidTracker

CONFIDENCE_THRESHOLD = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = os.path.join('models', 'best.pt')
model = YOLO(model_path).to(device)

video_path = os.path.join('input', '15sec_input_720p.mp4')
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs('output', exist_ok=True)
out = cv2.VideoWriter('output/tracked_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
tracker = CentroidTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes

    boxes = []
    for box, cls, conf in zip(detections.xyxy, detections.cls, detections.conf):
        if int(cls) == 2 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(boxes)

    for object_id, centroid in tracked_objects.items():
        for (x1, y1, x2, y2) in boxes:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {object_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

    out.write(frame)
    frame_count += 1
    print(f"Processed frame {frame_count}")

cap.release()
out.release()
print("✅ Finished processing. Output saved to 'output/tracked_output.mp4'")
