import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

# 彩图
girl = cv2.imread('beautiful_girl.jpg', 1)  # 1.图片名称 2.图片类型(0灰度图1彩色)

# 灰度图
girl_gray = cv2.imread('beautiful_girl.jpg', 0)

cv2.imshow('girl', girl)    # 1.窗口名 2.图片名称
cv2.imshow('girl_gray', girl_gray)

# 灰度图二维矩阵, 每个像素为1个像素值, 彩图3通道，三维数组，每个像素为1个数组(R, G, B)
print(girl)
print(girl_gray)

# 图片数据类型 uint8
print(girl.dtype)

# 图片信息，包含高（行数）, 宽（列数）, 通道数
print(girl.shape)
print(girl_gray.shape)

# 图片剪切（矩阵的切片）
girl_crop = girl[100:300, 200:400]
cv2.imshow('img_crop', girl_crop)

B, G, R = cv2.split(girl)   # 1.图片名称
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)


# change color
def random_light_color(img):
    # lightness
    B, G, R = cv2.split(img)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    else:
        lim = 0 - b_rand
        B[B >= lim] = (b_rand + B[B >= lim].astype(img.dtype))

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    else:
        lim = 0 - g_rand
        G[G >= lim] = (g_rand + G[G >= lim].astype(img.dtype))

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    else:
        lim = 0 - r_rand
        R[R >= lim] = (r_rand + R[R >= lim].astype(img.dtype))

    img_merge = cv2.merge((B, G, R))  # 融合, 3个通道的顺序
    return img_merge


img_random_color = random_light_color(girl)
cv2.imshow('img_random_color', img_random_color)
cv2.imshow('girl', girl)


# 伽马(gamma)变换, 其提升了暗部细节, 通过非线性变换, 让图像从暴光强度的线性响应变得更接近人眼感受的响应, 即将漂白（相机曝光）或过暗（曝光不足）的图片,进行矫正
# 当gamma>1是, 使灰度向高灰度值延展, 整体变亮。当gamma<1是, 使灰度向低灰度值收缩, 整体变暗
# cv2.LUT图片的颜色和table表做映射
def adjust_gamma(image, gamma=1.0):
    intGamme = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** intGamme) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(image, table)    # 1.原图 2.查找表


img_brighter = adjust_gamma(girl, 2)
cv2.imshow('img_brighter', img_brighter)

# histogram(直方图)
img_small_brighter = cv2.resize(girl, (int(girl.shape[0]*0.5), int(girl.shape[1]*0.5)))   # 1.图像数据 2.图像高度 3.图像宽度
plt.hist(img_brighter.flatten(), 255, [0, 256], color='r')  # 1.图片数据 2.x轴数据数量 3.x轴数据范围 4.颜色
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
# 均衡前颜色值分布不均, 均衡后颜色值整体符合正太分布
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])   # only for one channel
# convert the YUV image back to RGB format
img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)    # Y:明亮度 U&V:色度 饱和度

cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_out)

# rotation(旋转)
# 1.旋转中心点(默认左上角) 2.旋转的角度(正值表示逆时针) 3.缩放比例
# scale(比例)+rotation(旋转)+translation(平移) = similarity transform(相似变换)
M = cv2.getRotationMatrix2D((int(girl.shape[1] / 2), int(girl.shape[0] / 2)), 30, 0.8)
# 仿射 1.原图数据 2.模型 3.产生的图像宽度和高度
img_rotate = cv2.warpAffine(girl, M, (girl.shape[1], girl.shape[0]))
cv2.imshow('img_rotate', img_rotate)

# Affine Transform
# 仿射变换, 不保留直角但保留对边平行(需要3点)
height, width, channels = girl.shape
pts1 = np.float32([[0, 0], [height-1, 0], [0, width-1]])    # 原图中的三个点
pts2 = np.float32([[width * 0.2, height * 0.1], [width * 0.9, height * 0.2], [width * 0.1, height * 0.9]])  # 目标图中的三个点
M = cv2.getAffineTransform(pts1, pts2)  # 1.原图中的三个点 2.目标图中的三个点
dst = cv2.warpAffine(girl, M, (width, height))
cv2.imshow('dst', dst)


# perspective transform(投射变换)
# 投射变换, 不保留直角, 也不保留对边平行(需要4点)
def random_warp(img):
    height, width, channels = img.shape
    # wrap
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)   # 1.原图中的四个点 2.目标图中的四个点
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp


img_warp = random_warp(girl)
cv2.imshow('img_warp', img_warp)
cv2.waitKey(0)


