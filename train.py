from ultralytics import settings
from ultralytics import YOLO


settings.update({'datasets_dir': './data'})

model = YOLO('yolov8n.pt')

results = model.train(data='WiderPerson.yaml', epochs=100, imgsz=640)