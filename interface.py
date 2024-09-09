import counter

if __name__ == '__main__':
    counter.exercise_counter(pose_model='./pretrained_models/yolov8s-pose.pt',  #yolo模型
                             detector_model_path='./checkpoint',  # 训练完的检测姿态模型路径
                             detector_model_file='1.pt',#所使用的模型
                             video_file='./videos/squat_2.mp4',  # 视频文件
                             video_save_dir='./results'  # 视频保存路径
                             )