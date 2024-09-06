import math

from ultralytics import YOLO
import os


def calculate_shoulder_angle(key_points):
#肩部角度计算
    left_point_idx=5
    right_point_idx=6
    left_points = [key_points.xyn[0][left_point_idx][0],key_points.xyn[0][left_point_idx][1]]
    right_points = [key_points.xyn[0][right_point_idx][0],key_points.xyn[0][right_point_idx][1]]
    slope = math.atan2(left_points[1]-right_points[1], left_points[0]-right_points[0])
    angle = math.degrees(slope)

    if angle > 180:
        angle_diff = 360 - angle
    return angle


def calculate_angle_leg(key_points):
    left_points_idx = [11, 13, 15]
    right_points_idx = [12, 14, 16]
    #key_points：result.key_points
    def _calculate_angle(line1, line2):
        # Calculate the slope of two straight lines
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # Convert radians to angles
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360-angle_diff

        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
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

    degree=calculate_shoulder_angle(result.keypoints)
    degree2=180-calculate_angle_leg(result.keypoints)
    print(degree.__abs__(),degree2)
