from ultralytics import YOLO

model = YOLO("yolov8n.pt") # 加载一个预训练的YOLOv8n模型

# results = model.predict(source="images.jpg")

results = model.predict(source="images.jpg", stream=True)    #source替换为视频地址

for result in results:      #视频中的每一幅图
    # Detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    # result.boxes.xywh   # box with xywh format, (N, 4)
    # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    # result.boxes.conf   # confidence score, (N, 1)
    # result.boxes.cls    # cls, (N, 1)
    num_of_box = result.__len__()   #一幅图中有多少个框
    for i in range(num_of_box):
        xyxyn = result.boxes.xyxyn[i].tolist()
        print(xyxyn)
        #ToDo：在界面视频中，画出框xyxyn
        

        cls = result.boxes.cls[i].tolist()
        print(cls)
        #ToDo：在界面中，显示目标类别cls

        conf = result.boxes.conf[i].tolist()
        print(conf)
        #ToDo：在界面中，显示置信度conf

        #ToDo：判断鼠标坐标（归一化坐标）是否位于xyxyn之内，是则执行操作
        