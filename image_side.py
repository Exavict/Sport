import math

from ultralytics import YOLO
import os


def calculate_back_angle(key_points):
#背部角度计算
    left_point_idx=5
    right_point_idx=6
    left_points = [key_points.xyn[0][left_point_idx][0],key_points.xyn[0][left_point_idx][1]]
    right_points = [key_points.xyn[0][right_point_idx][0],key_points.xyn[0][right_point_idx][1]]
    slope = math.atan2(left_points[1]-right_points[1], left_points[0]-right_points[0])
    angle = math.degrees(slope)

    if angle > 180:
        angle_diff = 360 - angle
    return angle

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # pt文件路径
    pth_path = os.path.join(current_dir, './pretrained_models/yolov8n-pose.pt')

    # 加载模型
    pose_model = YOLO(pth_path)

    # 测试图像路径
    image_path = './test_imgs/001.jpg'


    # 执行预测并保存图像到默认目录
    results = pose_model.predict(image_path, imgsz=600, save=True)
    save_dir = results[0].save_dir

    # 将预测结果保存到 txt 文件
    results_file_path = os.path.join(save_dir, 'results.txt')

    with open(results_file_path, 'w') as f:
        for result in results:
            f.write(f"Keypoints: {result.keypoints}\n")
            f.write("\n")

    degree=calculate_back_angle(result.keypoints)

    print(degree.__abs__())
