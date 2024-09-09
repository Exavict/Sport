import image


if __name__ == '__main__':
    model_path = './pretrained_models/yolov8n-pose.pt'
    # 测试图像路径
    image_path_front = './test_imgs/001.jpg'  # 正面
    image_path_side = './test_imgs/003.png'  # 侧面

    degree_list=image.image(model_path,image_path_front,image_path_side)

    print (degree_list)

