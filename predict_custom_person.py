from ultralytics import YOLO
import os

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # pt文件路径
    pth_path = os.path.join(current_dir, 'runs/pose/train2/weights/best.pt')
    # Load a model
    pose_model = YOLO(pth_path)
    image_path = './test_imgs/000000000070.jpg'
    results = pose_model.predict(image_path, imgsz=600, save=True)
    print(f'预测结果 --> ', results)