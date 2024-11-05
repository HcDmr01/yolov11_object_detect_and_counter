from ultralytics import YOLO

model = YOLO("Models/yolo11n.pt")

train_results = model.train(
    data="data_custom.yaml",
    epochs=80,
    imgsz=640,
    device=0
)

metrics = model.val()