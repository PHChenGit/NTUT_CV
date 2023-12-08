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


def Canny(src):
    def conv(img, kernel):
        kernel_width, kernel_height = kernel.shape
        if kernel_height != kernel_width:
            raise Exception("Error, kernel should be a square matrix")
        padding = kernel_width // 2

        width_output = int(((img.shape[0] - kernel.shape[0] + 2 * padding) // 1.0) + 1)
        height_output = int(((img.shape[1] - kernel.shape[1] + 2 * padding) // 1.0) + 1)

        if padding != 0:
            padded_img = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2))
            padded_img[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = img
        else:
            padded_img = img

        new_img = np.zeros((width_output, height_output))

        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                window = (kernel * padded_img[x:x+kernel_width, y:y+kernel_height])
                new_img[x, y] = window.sum()
        return new_img

    def sobel(img):
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
        Ix = conv(img, Gx)
        Iy = conv(img, Gy)
        G = np.zeros((img.shape[0], img.shape[1]))

        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                g = np.sqrt((Ix[x, y] ** 2) + (Iy[x, y] ** 2))
                G[x, y] = g / 255


        theta = np.arctan2(Ix, Iy)
        return (G, theta)


    def non_maximum_suppression(img):
        return

    # scale = 1
    # delta = 0
    # ddepth = cv.CV_16S
    # grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # # Gradient-Y
    # grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # abs_grad_x = cv.convertScaleAbs(grad_x)
    # abs_grad_y = cv.convertScaleAbs(grad_y)
    #
    # grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # cv.imshow('Sobel Demo - Simple Edge Detector', grad)

    sobel_result, direction = sobel(src)
    cv.imshow("Gradient result", sobel_result)

    cv.waitKey(0)

    return


if __name__ == '__main__':
    input_folder_path = './test_img/'
    test_images = ['lena']
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
    # cv.waitKey(0)

    # CV_Canny = cv.Canny(gaussian_blured_img, 50, 50)

    # cv.imshow("OpenCV Canny Result", gaussian_blured_img)

    canny_resul = Canny(gaussian_blured_img)
    cv.waitKey(0)