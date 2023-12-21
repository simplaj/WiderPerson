from ultralytics import YOLO

# 加载模型
# model = YOLO('yolov8n.pt')  # 加载官方模型
model = YOLO('/home/tzh/Project/WiderPerson/WiderPerson/train18/weights/best.pt')  # 加载自定义模型

# 验证模型
metrics = model.val(save_json=True, split='test')  # 无需参数，数据集和设置通过模型属性记住
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 包含每个类别map50-95的列表