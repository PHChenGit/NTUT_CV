import cv2 as cv
import numpy as np
import sys


def convert_to_gray(img):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    cv.imshow("gray", gray)
    return gray


def padding_zeros(gray_img):
    return np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')


def convolution(gray_img):
    pass


def pooling():
    pass


def binary_operation():
    pass


if __name__ == '__main__':
    # img = cv.imread("../lena.png")
    img = cv.imread("./test_img/aeroplane.png")
    if img is None:
        sys.exit("Could not read the image.")

    np_img = np.array(img, dtype=np.float32)
    cv.imshow("Display window", img)
    q1_ans = convert_to_gray(np_img)

    k = cv.waitKey(0)
