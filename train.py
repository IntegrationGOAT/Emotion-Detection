from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
data=".", 
    epochs = 20,
    imgsz = 64,
    batch = 32
)

