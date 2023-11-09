import cv2 as cv
import numpy as np
from os.path import exists
import sys


def convert_to_gray(img):
    return np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def convolution(input_img, k, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(k))

    width_gray_img, height_gray_img = input_img.shape
    width_kernel, height_kernel = kernel.shape

    width_output = int(((width_gray_img - kernel.shape[0] + 2 * padding) // strides) + 1)
    height_output = int(((height_gray_img - kernel.shape[1] + 2 * padding) // strides) + 1)

    if padding != 0:
        padded_img = np.zeros((input_img.shape[0] + padding * 2, input_img.shape[1] + padding * 2))
        padded_img[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = input_img
    else:
        padded_img = input_img

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


def pooling(img):
    kernel_size = 2
    strides = 2
    padding = 0

    width = int(((img.shape[0] - kernel_size + 2 ** padding) // strides) + 1)
    height = int(((img.shape[1] - kernel_size + 2 ** padding) // strides) + 1)
    pooled_img = np.zeros((width, height), dtype=np.uint8)

    for x in range(0, img.shape[0], strides):
        for y in range(0, img.shape[1], strides):
            window = img[x:x+kernel_size, y:y+kernel_size]
            pooled_img[x//strides, y//strides] = window.max().astype(np.uint8)
    return pooled_img


def binary_operation(img):
    threshold = 130
    binary_img = ((img >= threshold) * 255).astype(np.uint8)
    return binary_img


def save_img(img, img_path):
    """
    To avoid saving duplicate image
    """
    if not exists(img_path):
        try:
            cv.imwrite(img_path, img)
        except:
            print("Save image failed")


if __name__ == '__main__':
    images = ["lena", "aeroplane", "taipei101"]
    input_dir = "./test_img/"
    output_dir = "./result_img/"
    kernel = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape((3, 3))

    for img_name in images:
        img = cv.imread(f"{input_dir}{img_name}.png")
        if img is None:
            sys.exit("Could not read the image.")

        cv.imshow("Input image", img)

        """
        Q1: Convert rgb images to gray-scale 
        """
        q1_ans = convert_to_gray(img)
        q1_ans_file = f"{output_dir}{img_name}_Q1.png"
        save_img(q1_ans, q1_ans_file)
        # cv.imshow("gray", q1_ans)

        """
        Q2: Convolution
        The convolution result is a 2D array
        """
        q2_ans = convolution(q1_ans, kernel, 1, 1)
        q2_ans_file = f"{output_dir}{img_name}_Q2.png"
        save_img(q2_ans, q2_ans_file)
        q2_ans_img = cv.imread(q2_ans_file)
        cv.imshow("Convolution", q2_ans_img)

        """
        Q3: Max pooling with kernel 2x2 and strides 2
        """
        q3_ans = pooling(q2_ans)
        q3_ans_file = f"{output_dir}{img_name}_Q3.png"
        save_img(q3_ans, q3_ans_file)
        cv.imshow("pooling", q3_ans)

        """
        Q4: Binary image
        """
        q4_ans = binary_operation(q3_ans)
        q4_ans_file = f"{output_dir}{img_name}_Q4.png"
        save_img(q4_ans, q4_ans_file)
        q4_img = cv.imread(q4_ans_file)
        cv.imshow("binary", q4_img)

        k = cv.waitKey(0)
