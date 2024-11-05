from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11m_custom_v4.pt")
video_path = "Brown1.mp4"
cap = cv2.VideoCapture(0)
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, conf=0.85)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
            annotated_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            cv2.imshow("YOLO11 Tracking", annotated_frame)
        else:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO11 Tracking", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
