from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='WiderPerson2.yaml',
        epochs=200, 
        imgsz=640, 
        batch=4,
        save_period=10,
        project='WiderPerson',
        )