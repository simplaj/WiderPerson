from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='WiderPerson.yaml',
        epochs=400, 
        imgsz=640, 
        batch=4,
        project='WiderPerson',
        )