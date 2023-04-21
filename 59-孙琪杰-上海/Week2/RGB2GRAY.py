# -*- coding: utf-8 -*-
"""
@Project     ：NLP
@File        ：RGB2GRAY.py
@IDE         ：PyCharm
@Author      ：sun.qijie
@Date        ：2023/4/15 14:27
@Description : 实现彩色图像的灰度化和二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1.灰度化手工实现
image = cv2.imread("lenna.png")  # 调用opencv读取文件
height, width = image.shape[:2]  # 获取图片的长宽
image_gray_1 = np.zeros([height, width], image.dtype)  # 创建一张和当前图片一样大小的单通道图片
for i in range(height):
    for j in range(width):
        m = image[i, j]  # 取出image矩阵第i行，第j列的值, BGR形式
        image_gray_1[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转换为灰度值
print("---image gray 1---")
print("image show gray: \n%s" % image_gray_1)
plt.subplots(2, 2, figsize=(18, 12))
plt.subplot(221)
plt.title('image gray 1', fontsize=16)
plt.imshow(image_gray_1, cmap='gray')  # 显示图片，其中win-name是窗口名

# 2.灰度化接口实现
image_gray_2 = rgb2gray(image)
plt.subplot(222)  # 用来定位图片的位置，表示在2*2网格的第二个位置
plt.title('image gray 2', fontsize=16)
plt.imshow(image_gray_2, cmap='gray')
print("---image gray 2---")
print(image_gray_2)

# 3.二值化手动实现
image_binary_1 = np.zeros([height, width], image.dtype)  # 创建一张和当前图片一样大小的单通道图片
for i in range(height):
    for j in range(width):
        if image_gray_2[i, j] >= 0.5:
            image_binary_1[i, j] = 1
        else:
            image_binary_1[i, j] = 0
print("---image_binary---")
print(image_binary_1)

plt.subplot(223)
plt.title('image binary 1', fontsize=16)
plt.imshow(image_binary_1, cmap='gray')
# 4.二值化接口实现
image_binary_2 = np.where(image_gray_2 >= 0.5, 1, 0)
plt.subplot(224)
plt.title('image binary 2', fontsize=16)
plt.imshow(image_binary_2, cmap='gray')

plt.show()
