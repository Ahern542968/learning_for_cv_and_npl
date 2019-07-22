import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


# random color(随机颜色)
def random_change_color(img):
    def random_color(channel, dtype):
        rand = random.randint(-50, 50)
        if rand > 0:
            channel[channel <= (255 - rand)] += rand
        elif rand < 0:
            channel[channel >= (0 - rand)] -= 0 - rand
        return channel.astype(dtype)
    return cv2.merge(tuple(random_color(channel, img.dtype) for channel in cv2.split(img)))


# similarity transform(相似变换)
def similarity_transform(img):
    height, width, channels = img.shape
    rand_angle = random.randint(-60, 60)
    rand_scale = random.randint(7, 13) / 10
    matrix = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), rand_angle, rand_scale)
    dst = cv2.warpAffine(img, matrix, (width, height))
    return dst


# affine transform(仿射变换)
def affine_transform(img):
    height, width, channels = img.shape
    rand1 = [random.randint(0, 2) / 10 for _ in range(4)]
    rand2 = [random.randint(7, 9) / 10 for _ in range(2)]
    pts1 = np.float32([[0, 0], [height - 1, 0], [0, width - 1]])
    pts2 = np.float32([[width * rand1[0], height * rand1[1]],
                       [width * rand2[0], height * rand1[2]],
                       [width * rand1[3], height * rand2[1]]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, matrix, (width, height))
    return dst


# perspective transform(投影变换)
def perspective_transform(img):
    height, width, channels = img.shape
    random_margin = random.randint(30, 60)

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
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, (width, height))
    return dst


def run():
    girl = cv2.imread('beautiful_girl.jpg', 1)
    for i in range(20):
        fun = random.choice([similarity_transform, affine_transform, perspective_transform])
        dst = fun(random_change_color(girl))
        cv2.imshow(f'dst_{i}', dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    run()
