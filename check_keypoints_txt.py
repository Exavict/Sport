import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_person_keypoints(img_path, txt_path):
    # 读取图像
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    # 获取图像的宽和高
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 读取txt文件
    f = open(txt_path)
    all_lines_data = f.readlines()
    for line_data in all_lines_data:
        # 0-4忽略，下标从第5个开始 ---> 3D
        # <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> ......
        left_keypoints = line_data.split(' ')[5:]
        for idx in range(0, len(left_keypoints), 3):
            one_keypoint = left_keypoints[idx: idx + 3]
            print(one_keypoint)
            point_x = int(float(one_keypoint[0]) * img_width)
            point_y = int(float(one_keypoint[1]) * img_height)
            cv2.circle(img_copy, (point_x, point_y), 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.savefig('./test_imgs/test_result.png')
    plt.show()


if __name__ == '__main__':
    img_file_path = './datasets/custom_person/images/000000000063.jpg'
    txt_label_path = './datasets/custom_person/labels/000000000063.txt'
    plot_person_keypoints(img_file_path, txt_label_path)


