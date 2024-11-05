from ultralytics import YOLO

model = YOLO("yolo11m_custom_v4.pt")

results = model.predict(source="Brown1.mp4", show=True, save=True)