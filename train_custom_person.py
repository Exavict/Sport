from ultralytics import YOLO


if __name__ == '__main__':
    # 加载预训练模型pt文件
    model = YOLO('./runs/pose/train/weights/best.pt')
    # yaml绝对路径
    results = model.train(data='/Users/shirley/PycharmProjects/yolov8_pose/datasets/custom_person/person.yaml',  # 确定路径
                          epochs=1,  # 训练轮数: 30    50    60   70   100
                          imgsz=(640, 420),  # 图像大小   800  900  1000  1100
                          device='CPU',  # 'CPU', 0 -> GPU
                          val=True,  # 训练期间是否验证
                          optimizer='auto',  # 优化器算法  'SGD'   'Adam'  'RMSProp'
                          batch=8  # 加载批次
                          )