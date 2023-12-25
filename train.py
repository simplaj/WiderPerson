from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='WiderPerson.yaml',
        epochs=100, 
        imgsz=(1000), 
        batch=4,
        save_period=10,
        project='WiderPerson',
        )