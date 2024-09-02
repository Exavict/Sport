import os
import cv2
import numpy as np
import math
import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy
import time


multi_lines_list = ['shoulder_hip_knee', 'shoulder_elbow_wrist', 'hip_knee_ankle']

concerned_keypoints_dict = {
    'shoulder_hip_knee': {  # situp
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'exercise_type': 'situp'
    },
    'shoulder_elbow_wrist': {  # pushup
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'exercise_type': 'pushup'
    },
    'hip_knee_ankle': {  # squat
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'exercise_type': 'squat'
    }
}

def analyze_exercise_action(key_points):
    exercise_type = 'other'

    for line_name in multi_lines_list:
        left_points_idx = concerned_keypoints_dict[line_name]['left_points_idx']
        right_points_idx = concerned_keypoints_dict[line_name]['right_points_idx']
        # Calculate average angle between left and right lines
        angle = calculate_angle(key_points, left_points_idx, right_points_idx)
        if angle < concerned_keypoints_dict[line_name]['relaxing']:
            exercise_type = concerned_keypoints_dict[line_name]['exercise_type']

    return exercise_type

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


def put_text(frame, exercise, during_time, redio):
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(260 * redio), int(160 * redio)),
        (55, 104, 0), -1
    )

    if exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(1.5 * redio), lineType=cv2.LINE_AA
        )
    elif exercise:
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.7 * redio,
            (255, 255, 255), thickness=int(1.5 * redio), lineType=cv2.LINE_AA
        )
    fps = round(1 / during_time, 2)
    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio), int(100 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(1.5 * redio), lineType=cv2.LINE_AA
    )
    cv2.putText(
        frame, f'1/FPS: {round(during_time, 3)}', (int(30 * redio), int(150 * redio)), 0, 0.7 * redio,
        (255, 255, 255), thickness=int(1.5 * redio), lineType=cv2.LINE_AA
    )


def pose_estimation_by_angle(pose_model, video_file, video_save_dir=None, isShow=True):
    # Load the YOLOv8 model
    model = YOLO(pose_model)

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
                        frame, 'No Object',
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

            # get exercise type by angle
            exercise_type = analyze_exercise_action(results[0].keypoints)

            during_time = time.time() - start_time
            print(f'During time by angle per frame ----> {during_time}')

            # Visualize the results on the frame
            annotated_frame = plot(
                results[0], plot_size_redio
            )

            # add relevant information to frame
            put_text(annotated_frame, exercise_type, during_time, plot_size_redio)
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


if __name__ == '__main__':
    pose_estimation_by_angle(pose_model='./yolov8n-pose.pt',  # pose模型
                             video_file='./squat.avi',  # 视频文件
                             video_save_dir='./for_detect/results'  # 视频保存路径
                            )
