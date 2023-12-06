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


def gaussian_blur(img, kernel_size=3, sigma=1.0):
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

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for y in range(kernel.shape[1]):
        for x in range(kernel.shape[0]):
            window = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
            kernel[x, y] = window
    kernel /= 2 * np.pi * (sigma ** 2)

    res = conv(img, kernel, padding=kernel_size // 2)
    return np.asarray(res, dtype=np.uint8)


if __name__ == '__main__':
    input_folder_path = './test_img/'
    test_images = ['img1']
    output_folder_path = './result_img/'

    img = cv.imread(f"{input_folder_path}{test_images[0]}.png")
    gray_img = convert_to_gray(img)

    # Gaussian = cv.GaussianBlur(gray_img, (3, 3), 0)
    # print(f"Gaussian shape: {Gaussian.shape}\n")
    # cv.imshow("Opecv Gaussian Blur Result", Gaussian)

    gaussian_blured_img = gaussian_blur(gray_img, 5)
    print(f"My Gaussian Blured Img: {gaussian_blured_img.shape}\n")

    cv.imshow("gaussian blured img 1", gaussian_blured_img)
    # cv.imwrite(f"{output_folder_path}{test_images[0]}_q1.png")
    cv.waitKey(0)