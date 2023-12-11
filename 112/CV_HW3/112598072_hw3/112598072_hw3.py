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

    def gaussian_kernel(size, sigma):
        kernel = np.zeros((size, size))
        center = size // 2

        for x in range(size):
            for y in range(size):
                x_1 = x - center
                y_1 = y - center
                kernel[x, y] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x_1 ** 2 + y_1 ** 2) / (2 * (sigma ** 2)))
        return kernel / np.sum(kernel)

    kernel = gaussian_kernel(kernel_size, sigma)

    res = conv(img, kernel, padding=kernel_size // 2)
    return np.asarray(res, dtype=np.uint8)


def Canny(src, low_threshold, high_threshold):
    def conv(input_img, kernel, strides=1):
        kernel = np.flipud(np.fliplr(kernel))
        width_gray_img, height_gray_img = input_img.shape
        width_kernel, height_kernel = kernel.shape

        padding = width_kernel // 2
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

    def sobel(img):
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],)
        Ix = conv(img, Gx)
        Iy = conv(img, Gy)

        G = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                g = np.sqrt((Ix[x, y] ** 2) + (Iy[x, y] ** 2))
                G[x, y] = g
                # G[x, y] = g % 255


        # cv.imshow('sobel', G.astype(np.uint8))
        # cv.waitKey(0)
        G = G.astype(np.uint8)
        theta = np.arctan2(Iy, Ix) * 180 / np.pi
        theta[theta < 0] += 180
        return G, theta

    def non_maximum_suppression(magnitude, directions):
        res = np.zeros_like(magnitude)

        for x in range(1, magnitude.shape[0] - 1):
            for y in range(1, magnitude.shape[1] - 1):
                angle = directions[x, y]

                # Check for horizontal or vertical edges
                if 0 <= angle < 22.5 or 157.5 <= angle <= 180:
                    neighbors = [magnitude[x, y - 1], magnitude[x, y + 1]]
                # Check for diagonal edges
                elif 22.5 <= angle < 67.5:
                    neighbors = [magnitude[x + 1, y - 1], magnitude[x - 1, y + 1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [magnitude[x + 1, y], magnitude[x - 1, y]]
                else:
                    neighbors = [magnitude[x - 1, y - 1], magnitude[x + 1, y + 1]]

                # Suppress non-maximum values
                if magnitude[x, y] >= max(neighbors):
                    res[x, y] = magnitude[x, y]
                else:
                    res[x, y] = 0

        return res

    # Define strong and weak edges
    def double_threshold(img, low_threshold, high_threshold):
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

        for y in range(1, N-1):
            for x in range(1, M-1):
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
    weak = 75

    double_threshold_res = double_threshold(non_maximum_img, low_threshold, high_threshold)
    res = hysteresis(double_threshold_res)

    return res


def remove_objects(img, x_ratio, y_ratio):
    res = img.copy()
    M, N = img.shape
    x1 = int(M) * x_ratio[0]
    x2 = int(M) * x_ratio[1]
    y1 = int(N) * y_ratio[0]
    y2 = int(N) * y_ratio[1]
    # print('shape', img.shape)
    print(f'x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}')
    for y in range(N):
        for x in range(M):
            if x < x1 or x > x2 or y < y1 or y > y2:
                res[x, y] = 0
    return res

def hough_transform(src):
    M, N = src.shape
    max_distance = int(np.sqrt((M**2) + (N**2)))

    thetas = np.deg2rad(np.arange(0, 180))
    rhos = np.linspace(-max_distance, max_distance, 2*max_distance)
    accumulator = np.zeros((2 * max_distance, len(thetas)), dtype=np.uint8)

    for y in range(N):
        for x in range(M):
            if src[x, y] > 0:
                for k in range(len(thetas)):
                    rho = int(x * np.cos(thetas[k]) + y * np.sin(thetas[k])) + max_distance
                    accumulator[rho, k] += 1

    return accumulator, thetas, rhos


def show_hough_transform(img, accumulator, thetas, rhos, threshold):
    COLOR_RED = (0, 0, 255)
    THICKNESS = 2

    # Find peaks in the accumulator
    peaks = np.argwhere(accumulator >= threshold)
    draw_line_img = img.copy()

    # Draw lines on the input image
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(draw_line_img, (y1, x1), (y2, x2), COLOR_RED, THICKNESS, cv.LINE_AA)

    return draw_line_img


if __name__ == '__main__':
    input_folder_path = './test_img/'
    test_images = ['img1', 'img2', 'img3']
    output_folder_path = './result_img/'
    canny_low_threshold = [20, 30, 20]
    canny_high_threshold = [30, 50, 30]
    hough_line_threshold = [50, 40, 43]
    KERNEL_SIZE = [3, 3, 5]
    SIGMA = [0.707, 1.0, 1.0]
    x_ratio = [(0.5, 0.75), (0.5, 0.8), (0.6, 0.72)]
    y_ratio = [(0.24, 0.8), (0.1, 0.9), (0.17, 0.8)]

    for idx in range(len(test_images)):
        print(f'processing {test_images[idx]}.png')
        img = cv.imread(f"{input_folder_path}{test_images[idx]}.png")
        # print(f"Input image {test_images[idx]} shape: {img.shape}\n")
        gray_img = convert_to_gray(img)

        gaussian_blured_img = gaussian_blur(gray_img, KERNEL_SIZE[idx], SIGMA[idx])
        print(f'Gaussian Blur {test_images[idx]}.png ')
        # cv.imshow(f"Gaussian Blur {test_images[idx]}_q1.png", gaussian_blured_img)
        cv.imwrite(f"{output_folder_path}{test_images[idx]}_q1.png", gaussian_blured_img)

        print(f'Canny Edge Detection {test_images[idx]}.png ')
        canny_result = Canny(gaussian_blured_img, canny_low_threshold[idx], canny_high_threshold[idx])
        # cv.imshow(f"Canny Edge Detection {test_images[idx]}_q2.png", canny_result)
        cv.imwrite(f"{output_folder_path}{test_images[idx]}_q2.png", canny_result)

        remove_objects_result = remove_objects(canny_result, x_ratio[idx], y_ratio[idx])

        print(f'Hough Transform {test_images[idx]}.png ')
        accumulator, thetas, radius = hough_transform(remove_objects_result)
        draw_line_img = show_hough_transform(img, accumulator, thetas, radius, hough_line_threshold[idx])
        cv.imshow(f"Hough Transform {test_images[idx]}_q3.png", draw_line_img)
        cv.imwrite(f"{output_folder_path}{test_images[idx]}_q3.png", draw_line_img)

        cv.waitKey(0)
        cv.destroyAllWindows()
    print('Done!!')


