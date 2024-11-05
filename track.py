from ultralytics import YOLO

model = YOLO("Models/yolo11m_custom_v4.pt")

results = model.track(source="Assets/falling_1.mp4", show=True, save=True, conf=0.9, tracker="bytetrack.yaml")