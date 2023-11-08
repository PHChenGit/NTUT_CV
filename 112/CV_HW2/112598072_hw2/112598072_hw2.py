import random
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def flatten(arr):
    res = []
    for y in range(arr.shape[1]):
        for x in range(arr.shape[0]):
            res.append(arr[x][y])
    return res


def convert_to_gray(img):
    return np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def mean_filter(img, kernel_size=3):
    def conv(input_img, k, padding=0, strides=1):
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
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

    res = conv(img, kernel, padding=kernel_size//2)
    return np.asarray(res, dtype=np.uint8)


def median_filter(img, kernel_size=3):
    def quick_sort(numbers):
        if len(numbers) <= 1:
            return numbers

        pivot = random.choice(numbers)
        pivot_list = []
        left = []
        right = []

        for val in numbers:
            if val < pivot:
                left.append(val)
            elif val > pivot:
                right.append(val)
            else:
                pivot_list.append(val)

        return quick_sort(left) + pivot_list + quick_sort(right)

    def median(numbers):
        # sorted_numbers = sorted(numbers)
        sorted_numbers = quick_sort(numbers)

        if len(numbers) % 2 == 0:
            front_idx = len(numbers) // 2 + 1
            next_idx = len(numbers) // 2 - 1
            return (sorted_numbers[front_idx] + sorted_numbers[next_idx]) // 2

        return sorted_numbers[len(numbers) // 2]

    def conv(input_img, padding = 0, k_size=3):
        result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        if padding != 0:
            padded_img = np.zeros((input_img.shape[0] + padding * 2, input_img.shape[1] + padding * 2))
            padded_img[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = input_img
        else:
            padded_img = input_img

        for y in range(input_img.shape[1]):
            for x in range(input_img.shape[0]):
                window = padded_img[x: x+ k_size, y:y+k_size]
                result[x, y] = median(window.flatten())
        return result

    return conv(img, kernel_size//2, kernel_size)


def img_histogram(img_path, output_path, title):
    img = cv.imread(img_path)
    gray_img = convert_to_gray(img)

    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.hist(flatten(gray_img), 256, (0, 255))

    """
    After called the show function,
    plt will create a new image layer,
    so the saved image will be blank
    """
    # plt.show()
    plt.savefig(output_path)
    plt.cla()


if __name__ == '__main__':
    input_dir = './test_img'
    output_dir = './result_img'

    output_img_path_list = {
        "q1": [{
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_q1.png",
            "window_name": "noise1_q1"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise2_q1.png",
            "window_name": "noise2_q1"
        }],
        "q2": [{
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_q2.png",
            "window_name": "noise1_q2"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise2_q2.png",
            "window_name": "noise2_q2"
        }],
        "q3": [{
            "input_img": f"{input_dir}/noise1.png",
            "output_img": f"{output_dir}/noise1_his.png",
            "window_name": "noise1_his"
        }, {
            "input_img": f"{input_dir}/noise2.png",
            "output_img": f"{output_dir}/noise2_his.png",
            "window_name": "noise2_his"
        }, {
            "input_img": f"{output_dir}/noise1_q1.png",
            "output_img": f"{output_dir}/noise1_q1_his.png",
            "window_name": "noise1_q1_his"
        }, {
            "input_img": f"{output_dir}/noise1_q2.png",
            "output_img": f"{output_dir}/noise1_q2_his.png",
            "window_name": "noise1_q2_his"
        }, {
            "input_img": f"{output_dir}/noise2_q1.png",
            "output_img": f"{output_dir}/noise2_q1_his.png",
            "window_name": "noise2_q1_his"
        }, {
            "input_img": f"{output_dir}/noise2_q2.png",
            "output_img": f"{output_dir}/noise2_q2_his.png",
            "window_name": "noise2_q2_his"
        }]
    }

    for idx in range(len(output_img_path_list["q1"])):
        input_img_path = output_img_path_list["q1"][idx]["input_img"]
        input_img = cv.imread(input_img_path)

        if input_img is None:
            sys.exit(f"The image {input_img_path} not found.")

        q1_ans = mean_filter(convert_to_gray(input_img), 5)
        # cv.imshow(output_img_path_list["q1"][idx]["window_name"], q1_ans)
        cv.imwrite(output_img_path_list["q1"][idx]["output_img"], q1_ans)

    cv.waitKey(0)

    for idx in range(len(output_img_path_list["q2"])):
        input_img_path = output_img_path_list["q2"][idx]["input_img"]
        input_img = cv.imread(input_img_path)

        if input_img is None:
            sys.exit(f"The image {input_img_path} not found.")

        q2_ans = median_filter(convert_to_gray(input_img), 5)
        # cv.imshow(output_img_path_list["q2"][idx]["window_name"], q2_ans)
        cv.imwrite(output_img_path_list["q2"][idx]["output_img"], q2_ans)

    cv.waitKey(0)

    for idx in range(len(output_img_path_list["q3"])):
        img_histogram(output_img_path_list["q3"][idx]["input_img"],
                      output_img_path_list["q3"][idx]["output_img"],
                      output_img_path_list["q3"][idx]["window_name"])

    # img_histogram("./test_img/noise2.png", "./result_img/noise2_his.png", "noise2_his")
    # img_histogram("./result_img/noise2_q2.png", "./result_img/noise2_q2_his.png", "noise2_q2_his")
