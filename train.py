from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='WiderPerson.yaml',
        epochs=400, 
        imgsz=1000, 
        batch=2,
        save_period=10,
        project='WiderPerson',
        )