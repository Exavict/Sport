import csv
import cv2
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pretrained_models/yolov8s-pose.pt', type=str, help='Path to model weight')
    parser.add_argument('--input_video', default='videos/Pushup_long.mp4', type=str, help='Path to input video')
    parser.add_argument('--data_save_path', default='for_detect/data/squat/002.csv', type=str, help='Path to save data')
    parser.add_argument('--data_len', default=5, type=int, help='Sequence length')
    args = parser.parse_args()
    return args


# data_len: 5 frames 17 keypoints
def collect_data(model_path, video_path, save_path, data_len=5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    # get video fps
    frame_count = round(cap.get(cv2.CAP_PROP_FPS))
    print('------', frame_count)
    with open(save_path, 'a', newline='') as data:
        writer = csv.writer(data)
        data_row = []
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
                results = model(frame)
                ori_data = results[0].keypoints.data[0, :, 0:2]
                ori_data = ori_data.tolist()
                data_row.append(ori_data)
                if len(data_row) == data_len:
                    writer.writerow(data_row)
                    del data_row[0]

                frame = results[0].plot(boxes=False)

                cv2.imshow("YOLOv8 Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # only for one person
    yolov8_model = './pretrained_models/yolov8s-pose.pt'  # 或者加载自己训练的模型文件
    input_video = './videos/squat_4.mp4'  # 视频文件
    data_save_path = './data/squat/003.csv'  # 生成csv文件
    collect_data(model_path=yolov8_model,
                 video_path=input_video,
                 save_path=data_save_path)