import cv2 as cv
import numpy as np
import sys


def convert_to_gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def padding_zeros(gray_img):
    return np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')


def convolution(gray_img):
    kernel = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape((3, 3))
    padding = 0
    strides = 1

    width_gray_img, height_gray_img = gray_img.shape
    width_kernel, height_kernel = kernel.shape

    width_output = int(((width_gray_img - kernel.shape[0] + 2 * padding) / strides) + 1)
    height_output = int(((height_gray_img - kernel.shape[1] + 2 * padding) / strides) + 1)

    padded_img = padding_zeros(gray_img)
    new_img = np.zeros((width_output, height_output))

    """
    The convolution core
    """
    for x in range(width_gray_img):
        if x > (width_gray_img - width_kernel):
            break

        if x % strides == 0:
            for y in range(height_gray_img):
                if y > (height_gray_img - height_kernel):
                    break

                try:
                    if y % strides == 0:
                        new_img[x, y] = (kernel * padded_img[x:x + width_kernel, y:y + height_kernel]).sum()
                except:
                    break

    return new_img


def pooling():
    pass


def binary_operation():
    pass


if __name__ == '__main__':
    img = cv.imread("../lena.png")
    # img = cv.imread("./test_img/aeroplane.png")
    if img is None:
        sys.exit("Could not read the image.")

    np_img = np.array(img, dtype=np.float32)
    cv.imshow("Input image", img)

    q1_ans = convert_to_gray(np_img)
    cv.imshow("gray", q1_ans)

    q2_ans = convolution(q1_ans)
    cv.imshow("not saved", q2_ans)
    cv.imwrite("./test_img/aeroplane_test_q2.png", q2_ans)
    q2_ans = cv.imread("./test_img/aeroplane_test_q2.png")
    cv.imshow("Convolution", q2_ans)

    k = cv.waitKey(0)
