import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

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

tracker = Tracker(distance_function="iou", distance_threshold=0.3)  # Lowered threshold

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    print(f"Frame {frame_count} Classes: {classes}")

    norfair_detections = []
    for (box, cls) in zip(detections, classes):
        if int(cls) in [1, 2]:  # Adjusted to test 1 or 2 as players
            x1, y1, x2, y2 = map(int, box[:4])
            norfair_detections.append(Detection(points=np.array([[x1, y1], [x2, y2]])))

    tracked_objects = tracker.update(detections=norfair_detections)

    for obj in tracked_objects:
        x1, y1 = map(int, obj.estimate[0])
        x2, y2 = map(int, obj.estimate[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    print(f"Processed frame {frame_count} - Active IDs: {[obj.id for obj in tracked_objects]} - Detections: {len(norfair_detections)}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Finished processing. Output saved to 'output/tracked_output.mp4'")