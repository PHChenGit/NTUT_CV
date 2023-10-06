import cv2 as cv
import numpy as np
from os.path import exists
import sys


def convert_to_gray(img):
    return np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def padding_zeros(gray_img):
    return np.pad(gray_img, ((1, 1), (1, 1)), mode='constant')


def convolution(gray_img, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    padding = 0
    strides = 1

    width_gray_img, height_gray_img = gray_img.shape
    width_kernel, height_kernel = kernel.shape

    width_output = int(((width_gray_img - kernel.shape[0] + 2 * padding) / strides) + 1)
    height_output = int(((height_gray_img - kernel.shape[1] + 2 * padding) / strides) + 1)

    padded_img = padding_zeros(gray_img)
    new_img = np.zeros((width_output, height_output), dtype=np.uint8)

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
                        pixel_val = (kernel * padded_img[x:x + width_kernel, y:y + height_kernel]).sum()
                        new_img[x, y] = np.clip(pixel_val, 0, 255).astype(np.uint8)
                except:
                    break

    return new_img


def pooling():
    pass


def binary_operation():
    pass


def save_img(img, img_path):
    """
    To avoid saving duplicate image
    """
    if not exists(img_path):
        try:
            cv.imwrite(img_path, img)
        except:
            print("Write img error")


if __name__ == '__main__':
    images = ["lena", "aeroplane", "taipei101"]
    input_dir = "./test_img/"
    output_dir = "./result_img/"
    kernel = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape((3, 3))

    for img in images:
        img = cv.imread(f"{input_dir}{img}.png")
        if img is None:
            sys.exit("Could not read the image.")

        np_img = np.array(img, dtype=np.uint8)
        cv.imshow("Input image", img)

        q1_ans = convert_to_gray(np_img)
        save_img(q1_ans, f"{output_dir}img_q1.png")
        cv.imshow("gray", q1_ans)

        q2_ans = convolution(q1_ans, kernel)
        """
        The variable q2_ans is a 2D array, if you want to show the 2D array as am image,
        you have to convert it to a 3D array.
        If you pass a 2D array into cv.show method, it will present an incorrect result.
        """
        # cv.imshow("before saved convolution", np.stack((q2_ans,)*3, axis=-1))

        save_img(q2_ans, f"{output_dir}img_q2.png")

        q2_ans = cv.imread(f"{output_dir}img_q2.png")
        cv.imshow("Convolution", q2_ans)

        k = cv.waitKey(0)
        break
