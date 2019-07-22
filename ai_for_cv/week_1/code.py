import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


class DataAugmentation:
    def __init__(self, path, time):
        self.path = path
        self.img = cv2.imread(self.path, 1)
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.dtype = self.img.dtype
        self.time = time

    # random color(随机颜色)
    def random_change_color(self):
        channels = []
        for channel in cv2.split(self.img):
            rand = random.randint(-50, 50)
            if rand > 0:
                channel[channel <= (255 - rand)] += rand
            elif rand < 0:
                channel[channel >= (0 - rand)] -= 0 - rand
            channels.append(channel.astype(self.dtype))
        return cv2.merge(channels)

    # similarity transform(相似变换)
    def similarity_transform(self, img):
        rand_angle = random.randint(-60, 60)
        rand_scale = random.randint(7, 13) / 10
        matrix = cv2.getRotationMatrix2D((int(self.width / 2), int(self.height / 2)), rand_angle, rand_scale)
        dst = cv2.warpAffine(img, matrix, (self.width, self.height))
        return dst

    # affine transform(仿射变换)
    def affine_transform(self, img):
        rand1 = [random.randint(0, 2) / 10 for _ in range(4)]
        rand2 = [random.randint(7, 9) / 10 for _ in range(2)]
        pts1 = np.float32([[0, 0], [self.height - 1, 0], [0, self.width - 1]])
        pts2 = np.float32([[self.width * rand1[0], self.height * rand1[1]],
                           [self.width * rand2[0], self.height * rand1[2]],
                           [self.width * rand1[3], self.height * rand2[1]]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, matrix, (self.width, self.height))
        return dst

    # perspective transform(投影变换)
    def perspective_transform(self, img):
        random_margin = random.randint(30, 60)

        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(self.width - random_margin - 1, self.width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(self.width - random_margin - 1, self.width - 1)
        y3 = random.randint(self.height - random_margin - 1, self.height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(self.height - random_margin - 1, self.height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(self.width - random_margin - 1, self.width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(self.width - random_margin - 1, self.width - 1)
        dy3 = random.randint(self.height - random_margin - 1, self.height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(self.height - random_margin - 1, self.height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, (self.width, self.height))
        return dst

    def run(self):
        for i in range(self.time):
            fun = random.choice([self.similarity_transform, self.affine_transform, self.perspective_transform])
            dst = fun(self.random_change_color())
            cv2.imwrite(f'dst_{i}.jpg', dst)
            cv2.imshow(f'dst_{i}', dst)
        cv2.waitKey(0)


if __name__ == '__main__':
    data = DataAugmentation('beautiful_girl.jpg', 20)
    data.run()
