import os
import sys

import cv2 as cv
import numpy as np
import scipy


def convert_to_gray(img):
    return np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def conv(gray_img, k, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(k))

    width_gray_img, height_gray_img = gray_img.shape
    width_kernel, height_kernel = kernel.shape

    width_output = int(((width_gray_img - kernel.shape[0] + 2 * padding) // strides) + 1)
    height_output = int(((height_gray_img - kernel.shape[1] + 2 * padding) // strides) + 1)

    if padding != 0:
        padded_img = np.zeros((gray_img.shape[0] + padding * 2, gray_img.shape[1] + padding * 2))
        padded_img[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = gray_img
    else:
        padded_img = gray_img

    new_img = np.zeros((width_output, height_output), dtype=np.uint8)

    """
    The convolution core
    """
    for y in range(height_gray_img):
        if y % strides == 0:
            for x in range(width_gray_img):
                try:
                    if x % strides == 0:
                        window = (kernel * padded_img[x: x + width_kernel, y: y + height_kernel]).sum()
                        new_img[x, y] = np.clip(window, 0, 255).astype(np.uint8)
                except:
                    break

    return new_img


def mean_filter(img):
    kernel = np.ones((3, 3), dtype=np.float32) / 9

    res = conv(img, kernel, padding=1)
    return res.astype(np.uint8)


def median_filter(img):
    pass


def img_histogram():
    pass

if __name__ == '__main__':
    input_dir = './test_img'
    output_dir = './result_img'

    output_img_path_list = {
        "q1": [{
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_q1.png"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise1_q2.png"
        }],
        "q2": [{
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_q2.png"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise2_q2.png"
        }],
        "q3": [{
            "input_img": f"{output_dir}/noise1_q1.png",
            "output_img": f"{output_dir}/noise1_q1_his.png"
        }, {
            "input_img": f"{output_dir}/noise1_q2.png",
            "output_img": f"{output_dir}/noise1_q1_his.png"
        }, {
            "input_img": f"{output_dir}/noise2_q1.png",
            "output_img": f"{output_dir}/noise2_q1_his.png"
        }, {
            "input_img": f"{output_dir}/noise2_q2.png",
            "output_img": f"{output_dir}/noise2_q2_his.png"
        }, {
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_his.png"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise2_his.png"
        }]
    }


    img = cv.imread(f"{input_dir}/noise1.png")
    if img is None:
        sys.exit("Could not read the image.")

    gray_img = convert_to_gray(img)

    mean_filter_result = mean_filter(gray_img)
    print(f"My mean filter:\n {mean_filter_result}\n")
    cv.imshow("my mean filter result", mean_filter_result)

    # box_blur_ker = np.ones((3, 3)) / 9
    # cv_mean_res = scipy.signal.convolve2d(gray_img, box_blur_ker, mode="same")
    cv_mean_res = cv.blur(gray_img, (3, 3))

    print(f"OpenCV mean filter:\n {cv_mean_res}\n")
    cv.imshow("cv mean res", cv_mean_res)
    cv.waitKey(0)

    # gray_img = np.asarray([[1, 2, 0, 1, 2],
    #                 [2, 3, 1, 1, 2],
    #                 [1, 4, 2, 2, 0],
    #                 [3, 2, 3, 3, 0],
    #                 [1, 0, 0, 2, 1]
    #                 ], dtype=np.uint8)
    #
    # # gray_img = convert_to_gray(img)
    #
    # kernel = np.ones((3, 3))/9
    # conv_img = conv(gray_img, kernel, 1)
    #
    # print(f"My conv: \n{conv_img}\n")
    #
    # # cv_conv_img = cv.filter2D(gray_img, -1, cv.flip(kernel, -1), borderType=cv.BORDER_CONSTANT)
    # cv_conv_img = cv.filter2D(gray_img, -1, kernel)
    #
    # print(f"OpenCV conv: \n{cv_conv_img}\n")
    #
    # sci_conv_img = scipy.signal.convolve2d(gray_img, kernel, mode="same")
    #
    # print(f"Scipy conv: \n{sci_conv_img}\n")