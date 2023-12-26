from ultralytics import YOLO

# 加载模型
# model = YOLO('yolov8n.pt')  # 加载官方模型
model = YOLO('/home/tzh/Project/WiderPerson/WiderPerson/train3/weights/epoch80.pt')  # 加载自定义模型
# 验证模型
metrics = model.val(save_json=True, split='test', imgsz=1000)  # 无需参数，数据集和设置通过模型属性记住