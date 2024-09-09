import math

from ultralytics import YOLO
import os


def calculate_angle_shoulder(key_points):
#肩部角度计算
    left_point_idx=5
    right_point_idx=6
    left_points = [key_points.xyn[0][left_point_idx][0],key_points.xyn[0][left_point_idx][1]]
    right_points = [key_points.xyn[0][right_point_idx][0],key_points.xyn[0][right_point_idx][1]]
    slope = math.atan2(left_points[1]-right_points[1], left_points[0]-right_points[0])
    angle = math.degrees(slope)

    if angle > 180:
        angle_diff = 360 - angle
    return angle.__abs__()

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
    return 180-angle

def calculate_angle_back(key_points):
#背部角度计算
    left1_point_idx=5
    left2_point_idx=11
    left1_points = [key_points.xyn[0][left1_point_idx][0],key_points.xyn[0][left1_point_idx][1]]
    left2_points = [key_points.xyn[0][left2_point_idx][0],key_points.xyn[0][left2_point_idx][1]]
    slope1 = math.atan2(left1_points[1]-left2_points[1], left1_points[0]-left2_points[0])
    angle1 = math.degrees(slope1)
    #未识别点剔除
    if left1_points==[0,0]:
        angle1=0
    if left1_points==[0,0]:
        angle1=0

    right1_point_idx = 6
    right2_point_idx = 12
    right1_points = [key_points.xyn[0][right1_point_idx][0], key_points.xyn[0][right1_point_idx][1]]
    right2_points = [key_points.xyn[0][right2_point_idx][0], key_points.xyn[0][right2_point_idx][1]]
    slope2 = math.atan2(right1_points[1] - right2_points[1], right1_points[0] - right2_points[0])
    angle2 = math.degrees(slope2)
    #未识别点剔除
    if right1_points==[0,0]:
        angle2=0
    if right2_points==[0,0]:
        angle2=0

    #未识别角度剔除
    if angle1==0:
        return  abs(angle2+90)
    if angle2==0:
        return  abs(angle1+90)

    return abs((angle1+angle2)/2+90)


def calculate_angle_neck(key_points):
    left_points_idx = [3, 5, 11]
    right_points_idx = [4, 6, 12]
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


    #排除未识别点
    for x in left_points:
        if x==[0,0]:
            angle_left=0
    for x in right_points:
        if x==[0,0]:
            angle_right=0


    if angle_left==0:
        return angle_right
    if angle_right==0:
        return angle_left
    angle = (angle_left + angle_right) / 2

    return angle

def image(model_path,image_path_front,image_path_side):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # pt文件路径
    pth_path = os.path.join(current_dir, model_path)

    # 加载模型
    pose_model = YOLO(pth_path)

    # 执行预测并保存图像到默认目录
    results_front = pose_model.predict(image_path_front, imgsz=600, save=True)
    results_side = pose_model.predict(image_path_side, imgsz=600, save=True)
    save_dir_front = results_front[0].save_dir
    save_dir_side = results_front[0].save_dir

    # 将预测结果保存到 txt 文件
    results_file_path_front = os.path.join(save_dir_front, 'results_front.txt')
    results_file_path_side = os.path.join(save_dir_side, 'results_side.txt')

    with open(results_file_path_front, 'w') as f:
        for result_front in results_front:
            f.write(f"Keypoints: {result_front.keypoints}\n")
            f.write("\n")

    with open(results_file_path_side, 'w') as f:
        for result_side in results_side:
            f.write(f"Keypoints: {result_side.keypoints}\n")
            f.write("\n")

    degree_shoulder= calculate_angle_shoulder(result_front.keypoints)
    degree_leg= calculate_angle_leg(result_front.keypoints)


    degree_back= calculate_angle_back(result_side.keypoints)
    degree_neck= calculate_angle_neck(result_side.keypoints)

    degree_list=[degree_shoulder,degree_leg,degree_back,degree_neck]

    return degree_list