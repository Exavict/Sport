import os
import cv2
import numpy as np
import math
import json
import datetime
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy
from Inference import LSTM
from config import *
import time

SPORTS_TYPE = ['squat', 'situp', 'pushup','pull-up']

sport_list = {
    'situp': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]]
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]]
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [17, 15], [15, 13]]
    },
   "pull-up":{
        'left_points_idx': [4, 6, 8],
        'right_points_idx': [3, 5, 7],
        'maintaining': 50,
        'relaxing': 100,
        'concerned_key_points_idx': [3, 4, 5, 6, 7, 8],
        'concerned_skeletons_idx': [[8, 6], [6, 4], [5, 3], [3, 7]]
    },


}

#角度计算
def calculate_angle(key_points, left_points_idx, right_points_idx):
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
            angle_diff = 360 - angle_diff

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

#视频帧标注，视频关键点绘制
def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
#pose_result: 包含姿态估计结果的对象，其中包括原始图像 (orig_img) 和关键点信息 (keypoints)。
#plot_size_redio: 用于调整绘制关键点和线条的大小比例。
#show_points: 如果提供，将只绘制这些指定的关键点。默认是 None，意味着绘制所有关键点。
#show_skeleton: 如果提供，将只绘制这些指定的骨架线。默认是 None，意味着绘制所有骨架线
    class _Annotator(Annotator):

        def kpts(self, kpts, shape=(640, 640), radius=5, line_thickness=2, kpt_line=True):
            """Plot keypoints on the image.

            Args:
                kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
                shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
                radius (int, optional): Radius of the drawn keypoints. Default is 5.
                kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                           for human pose. Default is True.
                line_thickness (int, optional): thickness of the kpt_line. Default is 2.

            Note: `kpt_line=True` currently only supports human pose plotting.
            """
            """
            在图像上绘制关键点。

            参数:
                kpts (tensor): 预测的关键点，形状为 [17, 3]。每个关键点包含 (x, y, 置信度)。
                shape (tuple): 图像的形状，表示为元组 (h, w)，其中 h 是高度，w 是宽度。
                radius (int, 可选): 绘制关键点的半径。默认值为 5。
                kpt_line (bool, 可选): 如果为 True，函数将绘制连接关键点的线条，用于人体姿态。默认值为 True。
                line_thickness (int, 可选): 线条的厚度。默认值为 2。

            注意: `kpt_line=True` 目前仅支持人体姿态绘制。
            """
            #画点
            if self.pil:
                # Convert to numpy first:pil->numpy
                self.im = np.asarray(self.im).copy()
            nkpt, ndim = kpts.shape
            is_pose = nkpt == 17 and ndim == 3
            kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
            colors = Colors()
            for i, k in enumerate(kpts):
                if show_points is not None:
                    if i not in show_points:
                        continue
                color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)),
                               int(radius * plot_size_redio), color_k, -1, lineType=cv2.LINE_AA)

            #画线
            if kpt_line:  # 如果启用了关键点连接线绘制
                ndim = kpts.shape[-1]  # 获取关键点数组的最后一个维度（维度数）
                for i, sk in enumerate(self.skeleton):  # 遍历每一对骨架连接
                    if show_skeleton is not None:  # 如果用户指定了要显示的骨架线
                        if sk not in show_skeleton:  # 如果当前骨架线不在指定的显示范围内，则跳过
                            continue
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))  # 获取第一个关键点的位置
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))  # 获取第二个关键点的位置
                    if ndim == 3:  # 如果关键点数据包含置信度（第三个维度）
                        conf1 = kpts[(sk[0] - 1), 2]  # 获取第一个关键点的置信度
                        conf2 = kpts[(sk[1] - 1), 2]  # 获取第二个关键点的置信度
                        if conf1 < 0.5 or conf2 < 0.5:  # 如果任一关键点的置信度低于0.5，则跳过这条线的绘制
                            continue
                    # 如果关键点的位置无效（在图像边界或外部），则跳过该线的绘制
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    # 使用 OpenCV 绘制骨架线
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             thickness=int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
                    # 将骨架线绘制在图像上，颜色为对应的 limb_color，线条厚度根据 plot_size_redio 进行调整

            if self.pil:  # 如果图像是 PIL 格式
                # 将处理后的 NumPy 数组图像转换回 PIL 图像，并更新绘制内容
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))  # 创建 _Annotator 实例并传入原始图像的副本
    if pose_result.keypoints is not None:  # 如果有关键点数据
        for k in reversed(pose_result.keypoints.data):  # 遍历每个关键点集（反向顺序）
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)  # 调用 kpts 方法在图像上绘制关键点和骨架线

    return annotator.result()  # 返回绘制后的图像结果

#视频文本：种类，次数，帧率
def put_text(frame, exercise, count, during_time, redio):
    #矩形框
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(163 * redio)),
        (55, 104, 0), -1
    )
    #运动种类文本绘制
    if exercise in sport_list.keys():
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    #未检测到为'No Object'
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    #fps文本绘制
    fps = round(1 / during_time, 2)
    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio), int(100 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
    #计数文本绘制
    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(150 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
'''
#姿态检测和运动识别
def pose_detect(model, input_pose_frames, idx_2_category_dict):#input_pose_frames：姿态数据
    input_data = torch.tensor(input_pose_frames)
    # 调整输入数据的形状为 (5, INPUT_DIM)，5 表示帧的数量，INPUT_DIM 表示每一帧的特征维度
    input_data = input_data.reshape(5, INPUT_DIM)
    x_mean, x_std = torch.mean(input_data), torch.std(input_data)#均值和标准差
    input_data = (input_data - x_mean) / x_std#标准化
    input_data = input_data.unsqueeze(dim=0)#扩张维度以适应模型
    input_data = input_data.to(model.device)
    result = model(input_data)
    # print(result)
    return idx_2_category_dict[str(result.argmax().cpu().item())]#得到名称
'''
#处理命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./pretrained_models/yolov8s-pose.pt', type=str, help='Path to model weight')
    parser.add_argument('--detector_model', default='./checkpoint/', type=str, help='Path to detect model checkpoint')
    parser.add_argument('--sport', default=['squat', 'situp', 'pushup','pull-up'], type=str, help='Currently supported "situp", "pushup" and "squat"')
    parser.add_argument('--input', default='./videos/squat.avi', type=str, help='Path to input video')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save output')
    parser.add_argument('--show', default=True, type=bool, help='show the result')
    args = parser.parse_args()
    return args

#加载模型，打开输入视频，逐帧处理（运动类型的识别、次数统计、结果可视化、视频的保存）
def exercise_counter(pose_model, detector_model_path, detector_model_file, video_file, video_save_dir=None, isShow=True, sports_type=SPORTS_TYPE):
    # Obtain relevant parameters
    # args = parse_args()
    # Load the YOLOv8 model 加载yolo模型
    model = YOLO(pose_model)

    # Load exersice model
    with open(os.path.join(detector_model_path, 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    # detect_model = LSTM(17*2, 8, 2, 3, model.device)
    detect_model = LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(DEVICE)#创建模型+权重
    model_path = os.path.join(detector_model_path, detector_model_file)#加载权重
    model_weight = torch.load(model_path)
    detect_model.load_state_dict(model_weight)#应用加载的权重

    # Open the video file or camera 打开视频文件或摄像头
    if video_file.isnumeric():
        cap = cv2.VideoCapture(int(video_file))
    else:
        cap = cv2.VideoCapture(video_file)

    # For save result video 保存视频
    if video_save_dir is not None:
        # 生成保存结果视频的目录，使用当前日期和时间命名文件夹
        save_dir = os.path.join(video_save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 编码器：“DIVX"、”MJPG"、“XVID”、“X264"; XVID MPEG4 Codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 获取原始视频的帧率（FPS）
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # 创建一个VideoWriter对象，用于将处理后的帧保存为视频文件
        output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    # Set variables to record motion status初始化变量以标记运动状态
    reaching = False#当前是否达到某个标准
    reaching_last = False#上一帧运动状态
    state_keep = False#运动状态是否稳定
    counter = []
    for i in range(len(sports_type)):
        counter.append(0)

    pose_key_point_frames = []#用于存储姿态检测过程中每一帧的关键点数据
    exersice_type = 'detecting'#运动类型

    # Loop through the video frames 循环处理视频帧
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # 设置绘制尺寸比例以适应不同分辨率的输入
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            # 记录开始处理这一帧的时间
            start_time = time.time()

            # Run YOLOv8 inference on the frame 在帧上运行yolov8
            results = model(frame)

            # Preventing errors caused by special scenarios 特殊情况，如未检测到关键点
            if results[0].keypoints.shape[1] == 0:
                if isShow:
                    #没有检测到为no object
                    put_text(
                        frame, 'No Object', counter[idx],
                        results[0].speed['inference']*1000, plot_size_redio
                    )
                    #调整显示窗口大小并显示帧
                    scale = 1280 / max(frame.shape[0], frame.shape[1])
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if video_save_dir is not None:
                    #指定了保存路径则将帧写入视频文件
                    output.write(frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # make sure exercise type 确认运动类型
            pose_frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            pose_result = model(pose_frame)#获取关键点
            pose_data = pose_result[0].keypoints.data[0, :, 0:2]#提取关键点数据
            pose_key_point_frames.append(pose_data.tolist())#累计帧数据：5帧
            idx = 0

            #当关键点数据累积到5帧时，进行运动类型预测
            if len(pose_key_point_frames) == 5:  # 5 -> get_data_from_video.py: collect_data(data_len=5)
                input_data = torch.tensor(pose_key_point_frames)
                input_data = input_data.reshape(5, INPUT_DIM)
                #计算输入数据的均值和标准差进行标准化
                x_mean, x_std = torch.mean(input_data), torch.std(input_data)
                input_data = (input_data - x_mean) / x_std
                #扩展维度，以适应LSTM
                input_data = input_data.unsqueeze(dim=0)
                input_data = input_data.to(detect_model.device)
                #使用检测模型推理
                rst_detector = detect_model(input_data)
                idx = rst_detector.argmax().cpu().item()
                # 根据输出确定类型
                exersice_type = idx_2_category[str(idx)]
                del pose_key_point_frames[0]


            # Get hyperparameters 获取超参数
            if exersice_type not in sports_type:
                sport = sports_type[0]
            else:
                sport = exersice_type
            left_points_idx = sport_list[sport]['left_points_idx']
            right_points_idx = sport_list[sport]['right_points_idx']

            # Calculate average angle between left and right lines 左右平均角度
            angle = calculate_angle(results[0].keypoints, left_points_idx, right_points_idx)

            # Determine whether to complete once 判断是否完成一次
            if angle < sport_list[sport]['maintaining']:
                reaching = True
            if angle > sport_list[sport]['relaxing']:
                reaching = False

            #更新状态，统计完成次数
            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                if not reaching and state_keep:
                    counter[idx] += 1
                    state_keep = False

            #计算处理本帧花费时间
            during_time = time.time() - start_time
            print(f'During time for the exercise counter per frame ----> {during_time}')

            # Visualize the results on the frame 可视化结果
            annotated_frame = plot(
                results[0], plot_size_redio
            )

            # add relevant information to frame添加关键信息到帧
            put_text(
                annotated_frame, exersice_type, counter[idx],
                during_time, plot_size_redio
            )
            # Display the annotated frame 显示带注释的帧
            if isShow:
                scale = 1280 / max(annotated_frame.shape[0], annotated_frame.shape[1])
                show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow("YOLOv8 Inference", show_frame)

            #如果指定了保存路径，则将带注释的帧写入视频文件
            if video_save_dir is not None:
                output.write(annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 视频结束跳出循环
            break

    # Release the video capture object and close the display window释放视频捕捉对象并关闭窗口
    cap.release()
    if video_save_dir is not None:
        output.release()
    cv2.destroyAllWindows()

    for i in range(len(sports_type)):
        print(f'{idx_2_category[str(i)]} : {counter[i]}')



