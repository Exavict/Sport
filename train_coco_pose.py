from ultralytics import YOLO


if __name__ == '__main__':
    # 加载预训练的YOLO-pose模型（推荐用于训练）
    # yolov8n.pt: Nano最
    model = YOLO('./pretrained_models/yolov8n-pose.pt')
    # yaml绝对路径
    results = model.train(data='/Users/shirley/PycharmProjects/yolov8_pose/datasets/coco-pose/coco-pose.yaml',  # 确定路径
                          epochs=1,  # 训练轮数: 30    50    60   70   100
                          imgsz=(640, 420),  # 图像大小   800  900  1000  1100
                          device='CPU',  # 'CPU', 0 -> GPU
                          val=True,  # 训练期间是否验证
                          optimizer='auto',  # 优化器算法  'SGD'   'Adam'  'RMSProp'
                          batch=8  # 加载批次
                          )