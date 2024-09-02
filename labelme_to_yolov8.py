
import os, json
import numpy as np

'''
5 + 17*3 = 5 + 51 = 56 列
0 0.542742 0.688759 0.218891 0.487635 
0.584375 0.505855 2.000000 
0.592187 0.501171 2.000000 
0.578125 0.496487 2.000000 
0.000000 0.000000 0.000000 
0.560937 0.487119 2.000000 
0.576562 0.531616 2.000000 
0.531250 0.512881 2.000000 
0.581250 0.578454 2.000000 
0.543750 0.634660 2.000000 
0.612500 0.644028 2.000000 
0.598437 0.676815 2.000000 
0.517188 0.622951 2.000000 
0.473438 0.585480 2.000000 
0.535937 0.730679 2.000000 
0.448437 0.676815 2.000000 
0.503125 0.889930 2.000000 
0.462500 0.573770 2.000000
'''

person_list = ['cls', 'cx', 'cy', 'w', 'h']  # 0 - 4  前5列
# 关键点描述，对应labelme工具的label值
keypoint_list = ['nose', 'left_eye', 'right_eye',  # 0, 1, 2
              'left_ear', 'right_ear',  # 3, 4
              'left_shoulder', 'right_shoulder',  # 5, 6
              'left_elbow', 'right_elbow',  # 7, 8
              'left_wrist', 'right_wrist',  # 9, 10
              'left_hip', 'right_hip',  # 11, 12
              'left_knee', 'right_knee',  # 13, 14
              'left_ankle', 'right_ankle']  # 15, 16

# 设置保存txt文件的领
txt_saving_path = './datasets/custom_person/labels'

def convert(json_path):
    # 加载当前json文件所有内容
    with open(json_path) as f:
        data = json.load(f)  # load JSON

    # 初始化peoples ndarray，存储所有归一化后的数据; 每个人包含56列信息(3D)
    COLUMNS_NUM = 56
    peoples = np.zeros((1, COLUMNS_NUM), dtype=np.float16)

    # 获取图像的宽和高
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    shapes_dict = data['shapes']

    # 获取所有shapes：包含person本身信息，以及keypoints关键点信息
    for shape in shapes_dict:
        if shape['label'] == 'person':  # 当前是矩形框中的person人
            if shape['group_id'] and int(shape['group_id']) > 0:  # 多个person目标
                new_array = np.zeros((1, COLUMNS_NUM), dtype=np.float16)
                peoples = np.concatenate((peoples, new_array))

            # 获取people的group_id
            pidx = shape['group_id']  # if shape['group_id'] else 0
            # 第0列：设置所属类别，一个图像可能多个people对象
            peoples[pidx][0] = 0
            # 获取person目标的左上坐标和右下坐标
            left_x = shape['points'][0][0] if shape['points'][0][0] >= 0.0 else 0.0
            top_y = shape['points'][0][1] if shape['points'][0][1] >= 0.0 else 0.0
            right_x = shape['points'][1][0] if shape['points'][0][0] <= image_width else image_width
            bottom_y = shape['points'][1][1] if shape['points'][0][1] <= image_height else image_height  # 处理越界

            # 归一化后person的中心坐标cx和cy
            cx, cy = (left_x + right_x) / (2 * image_width), (top_y + bottom_y) / (2 * image_height)
            # 归一化后person的宽和高
            w, h = (right_x - left_x + 1) / image_width, (bottom_y - top_y + 1) / image_height
            # 设置第1 - 4列数据
            peoples[pidx][1] = cx
            peoples[pidx][2] = cy
            peoples[pidx][3] = w
            peoples[pidx][4] = h
        else:  # keypoints
            # 获取keypoints的group_id  --> 和person的group_id对应
            pidx = shape['group_id']
            # 获取当前关键点在列表中的索引
            kpidx = keypoint_list.index(shape['label'])
            print(f"当前{shape['label']}的索引是{kpidx}")
            # <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> ......
            # 设置<px1>, <py1>
            peoples_idx = 3 * kpidx + 5  # 3D
            # 当前列索引为
            print(f'当前列索引为{peoples_idx}')
            peoples[pidx][peoples_idx] = shape['points'][0][0] / image_width
            peoples[pidx][peoples_idx + 1] = shape['points'][0][1] / image_height
            # p1 - visibility ---> 默认值为0(未标注)
            if shape['description'] == 'hidden':  # 被标注，但是不可见
                peoples[pidx][peoples_idx + 2] = 1
            else:
                peoples[pidx][peoples_idx + 2] = 2  # 被标注，并可见

    # 获取要保存txt文件名
    txt_name = data['imagePath'].split('/')[-1].split('.')[0]
    # 拼接路径
    dst_path = os.path.join(txt_saving_path, txt_name + '.txt')
    with open(dst_path, 'w') as f:
        for people in peoples:
            for i in range(len(people)):
                if i == 0:
                    f.write(str(int(people[i])) + ' ')
                elif i == (len(people) - 1):  # 最后一列没有空格
                    f.write(str(people[i]))
                else:
                    f.write(str(people[i]) + ' ')

            f.write('\n')


if __name__ == '__main__':
    # 待转换json文件所在目录
    original_json_dir = r'./datasets/custom_person/jsons'
    jsons_list = os.listdir(original_json_dir)
    for json_path in jsons_list:
        if json_path.endswith('.json'):
            json_file_path = os.path.join(original_json_dir, json_path)
            convert(json_file_path)
            print('Finish converting the file ----> ', json_file_path)
