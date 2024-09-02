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

SPORTS_TYPE = ['squat', 'situp', 'pushup','pull-up','jumping-jack']

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
    "jumping-jack":{
        'left_points_idx': [5, 11, 13],  # Arms and legs
        'right_points_idx': [6, 12, 14],
        'maintaining': 90,  # Example angle for maintaining position
        'relaxing': 150,    # Example angle for relaxing position
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[5, 11], [11, 13], [6, 12], [12, 14]]

    }
}

def calculate_angle(key_points, left_points_idx, right_points_idx):
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


def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
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
            if self.pil:
                # Convert to numpy first
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

            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    if show_skeleton is not None:
                        if sk not in show_skeleton:
                            continue
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             thickness=int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))
    if pose_result.keypoints is not None:
        for k in reversed(pose_result.keypoints.data):
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)
    return annotator.result()


def put_text(frame, exercise, count, during_time, redio):
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(163 * redio)),
        (55, 104, 0), -1
    )

    if exercise in sport_list.keys():
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    fps = round(1 / during_time, 2)
    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio), int(100 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(150 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )


def pose_detect(model, input_pose_frames, idx_2_category_dict):
    input_data = torch.tensor(input_pose_frames)
    input_data = input_data.reshape(5, INPUT_DIM)
    x_mean, x_std = torch.mean(input_data), torch.std(input_data)
    input_data = (input_data - x_mean) / x_std
    input_data = input_data.unsqueeze(dim=0)
    input_data = input_data.to(model.device)
    result = model(input_data)
    # print(result)
    return idx_2_category_dict[str(result.argmax().cpu().item())]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./pretrained_models/yolov8s-pose.pt', type=str, help='Path to model weight')
    parser.add_argument('--detector_model', default='./for_detect/checkpoint/', type=str, help='Path to detect model checkpoint')
    parser.add_argument('--sport', default=['squat', 'situp', 'pushup'], type=str, help='Currently supported "situp", "pushup" and "squat"')
    parser.add_argument('--input', default='./videos/squat.avi', type=str, help='Path to input video')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save output')
    parser.add_argument('--show', default=True, type=bool, help='show the result')
    args = parser.parse_args()
    return args


def exercise_counter(pose_model, detector_model_path, detector_model_file, video_file, video_save_dir=None, isShow=True, sports_type=SPORTS_TYPE):
    # Obtain relevant parameters
    # args = parse_args()
    # Load the YOLOv8 model
    model = YOLO(pose_model)

    # Load exersice model
    with open(os.path.join(detector_model_path, 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    # detect_model = LSTM(17*2, 8, 2, 3, model.device)
    detect_model = LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(DEVICE)
    model_path = os.path.join(detector_model_path, detector_model_file)
    model_weight = torch.load(model_path)
    detect_model.load_state_dict(model_weight)

    # Open the video file or camera
    if video_file.isnumeric():
        cap = cv2.VideoCapture(int(video_file))
    else:
        cap = cv2.VideoCapture(video_file)

    # For save result video
    if video_save_dir is not None:
        save_dir = os.path.join(video_save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 编码器：“DIVX"、”MJPG"、“XVID”、“X264"; XVID MPEG4 Codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    # Set variables to record motion status
    reaching = False
    reaching_last = False
    state_keep = False
    counter = []
    for i in range(len(sports_type)):
        counter.append(0)

    pose_key_point_frames = []
    exersice_type = 'detecting'

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Set plot size redio for inputs with different resolutions
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            # for time
            start_time = time.time()


            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Preventing errors caused by special scenarios
            if results[0].keypoints.shape[1] == 0:
                if isShow:
                    put_text(
                        frame, 'No Object', counter[idx],
                        results[0].speed['inference']*1000, plot_size_redio
                    )
                    scale = 1280 / max(frame.shape[0], frame.shape[1])
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if video_save_dir is not None:
                    output.write(frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # make sure exercise type
            pose_frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            pose_result = model(pose_frame)
            pose_data = pose_result[0].keypoints.data[0, :, 0:2]
            pose_key_point_frames.append(pose_data.tolist())
            idx = 0
            if len(pose_key_point_frames) == 5:  # 5 -> get_data_from_video.py: collect_data(data_len=5)
                input_data = torch.tensor(pose_key_point_frames)
                input_data = input_data.reshape(5, INPUT_DIM)
                x_mean, x_std = torch.mean(input_data), torch.std(input_data)
                input_data = (input_data - x_mean) / x_std
                input_data = input_data.unsqueeze(dim=0)
                input_data = input_data.to(detect_model.device)
                rst_detector = detect_model(input_data)
                idx = rst_detector.argmax().cpu().item()
                # predict exercise type
                exersice_type = idx_2_category[str(idx)]
                del pose_key_point_frames[0]



            # Get hyperparameters
            if exersice_type not in sports_type:
                sport = sports_type[0]
            else:
                sport = exersice_type
            left_points_idx = sport_list[sport]['left_points_idx']
            right_points_idx = sport_list[sport]['right_points_idx']

            # Calculate average angle between left and right lines
            angle = calculate_angle(results[0].keypoints, left_points_idx, right_points_idx)

            # Determine whether to complete once
            if angle < sport_list[sport]['maintaining']:
                reaching = True
            if angle > sport_list[sport]['relaxing']:
                reaching = False

            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                if not reaching and state_keep:
                    counter[idx] += 1
                    state_keep = False

            during_time = time.time() - start_time
            print(f'During time for the exercise counter per frame ----> {during_time}')

            # Visualize the results on the frame
            annotated_frame = plot(
                results[0], plot_size_redio
            )

            # add relevant information to frame
            put_text(
                annotated_frame, exersice_type, counter[idx],
                during_time, plot_size_redio
            )
            # Display the annotated frame
            if isShow:
                scale = 1280 / max(annotated_frame.shape[0], annotated_frame.shape[1])
                show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow("YOLOv8 Inference", show_frame)

            if video_save_dir is not None:
                output.write(annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if video_save_dir is not None:
        output.release()
    cv2.destroyAllWindows()

    for i in range(len(sports_type)):
        print(f'{idx_2_category[str(i)]} : {counter[i]}')


if __name__ == '__main__':
    exercise_counter(pose_model='./yolov8s-pose.pt',  # pose模型
                     detector_model_path='./checkpoint',  # 训练完的检测姿态模型路径
                     detector_model_file='best.pt',
                     video_file='./squat.avi',  # 视频文件
                     video_save_dir='./for_detect/results'  # 视频保存路径
                     )
