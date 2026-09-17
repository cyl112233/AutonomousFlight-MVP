import numpy as np
import cv2
def label_to_RGB(tensor ,lable_clors_RGB):


    label_colors = lable_clors_RGB


    label_img = np.array(tensor,dtype=np.uint8)

    h, w = label_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 根据映射表进行转换
    for row in range(h):
        for col in range(w):
            label = label_img[row, col]
            rgb_img[row, col] = label_colors[label]
    rgb_img = rgb_img.swapaxes(0, 2)
    return rgb_img
def segmentation_example(image,mask):
    image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_BGR = cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)

    # 设置透明度
    alpha = 1  # 原图像的透明度
    beta = 0.4  # 掩码的透明度
    gamma = 0  # 加权和的偏移量

    # 叠加图像
    result = cv2.addWeighted(image_BGR, alpha, mask_BGR, beta, gamma)
    return result

        # 保存结果
        # cv2.imwrite('./output_image.png', result)


