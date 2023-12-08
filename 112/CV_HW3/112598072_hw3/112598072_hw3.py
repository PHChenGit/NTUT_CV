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
        return G, theta

    def non_maximum_suppression(magnitude, direction):
        M, N = magnitude.shape
        res = np.zeros((M, N))
        angle = direction * 180 / np.pi
        angle[angle < 0] += 180

        for y in range(1, N-1):
            for x in range(1, M-1):
                neighbors = []

                # angle 0
                if (0 <= angle[x, y] < 22.5) or (157.5 <= angle[x, y] <= 180):
                    neighbors = [magnitude[x, y + 1], magnitude[x, y - 1]]
                # angle 45
                elif 22.5 <= angle[x, y] < 67.5:
                    neighbors = [magnitude[x + 1, y - 1], magnitude[x - 1, y + 1]]
                # angle 90
                elif 67.5 <= angle[x, y] < 112.5:
                    neighbors = [magnitude[x + 1, y], magnitude[x - 1, y]]
                # angle 135
                elif 112.5 <= angle[x, y] < 157.5:
                    neighbors = [magnitude[x - 1, y - 1], magnitude[x + 1, y + 1]]

                # Compare current pixel with neighbors
                # and then choose the bigger one
                if magnitude[x, y] >= np.max(neighbors):
                    res[x, y] = magnitude[x, y]

        return res

    # Define strong and weak edges
    def double_threshold(img, low_threshold_ratio=0.01, high_threshold_ratio=0.09):
        high_threshold = img.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio
        threshold_res = np.zeros_like(img)

        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                if img[x, y] >= high_threshold:
                    threshold_res[x, y] = strong
                elif low_threshold < img[x, y] < high_threshold:
                    threshold_res[x, y] = weak
                else:
                    threshold_res[x, y] = 0
        return threshold_res

    # Weak edges that are connected to strong edges will be actual/real edges
    # Weak edges that are not connected to strong edges will be removed
    def hysteresis(img):
        M, N = img.shape

        for y in range(N):
            for x in range(M):
                # If the current pixel is weak and connect to strong edges, then modify this pixel to strong
                # Otherwise, remove this pixel from edges
                if img[x, y] == weak:
                    if (img[x, y+1] == strong or img[x, y-1] == strong or img[x+1, y] == strong
                        or img[x-1, y] == strong or img[x+1, y+1] == strong or img[x+1, y-1] == strong
                        or img[x-1, y+1] == strong or img[x-1, y-1] == strong
                    ):
                        img[x, y] = strong
                    else:
                        img[x, y] = 0
        return img

    sobel_result, direction = sobel(src)
    non_maximum_img = non_maximum_suppression(sobel_result, direction)

    strong = 255
    weak = 100

    double_threshold_res = double_threshold(non_maximum_img, 0.072, 0.15)
    # cv.imshow("double threshold res", double_threshold_res)
    res = hysteresis(double_threshold_res)

    return res


if __name__ == '__main__':
    input_folder_path = './test_img/'
    test_images = ['lena', 'EmmaStone', 'img1', 'img2', 'img3']
    output_folder_path = './result_img/'

    for idx in range(len(test_images)):
        img = cv.imread(f"{input_folder_path}{test_images[idx]}.png")
        print(f"Input image {test_images[idx]} shape: {img.shape}\n")
        gray_img = convert_to_gray(img)

        gaussian_blured_img = gaussian_blur(gray_img, 5)
        cv.imwrite(f"{output_folder_path}{test_images[idx]}_q1.png", gaussian_blured_img)

        canny_result = Canny(gaussian_blured_img)

        cv.imshow(f"{test_images[idx]}_q2.png", canny_result)
        cv.imwrite(f"{output_folder_path}{test_images[idx]}_q1.png", canny_result)
        cv.waitKey(0)


        break
    # cv.imshow("Canny Result", canny_resul)
    # cv.waitKey(0)
