from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='WiderPerson2.yaml',
        epochs=400, 
        imgsz=1024, 
        batch=2,
        lr0=5e-3,
        save_period=10,
        project='WiderPerson',
        )