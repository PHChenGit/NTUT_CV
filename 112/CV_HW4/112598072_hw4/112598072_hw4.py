import cv2 as cv
import numpy as np

COLOR_RED = (0, 0, 255)
points = []

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


def set_init_points(img):
    M, N, _ = img.shape
    center_point = (M//2, N//2)
    radius = center_point[0] if center_point[0] < center_point[1] else center_point[1]
    num_points = 20

    # 計算小圓點的位置
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points_x = np.int32(center_point[0] + radius * 0.8 * np.cos(theta))
    circle_points_y = np.int32(center_point[1] + radius * 0.8 * np.sin(theta))

    for i in range(num_points):
        point = (circle_points_x[i], circle_points_y[i])
        points.append(point)
        cv.circle(img, point, 3, COLOR_RED, 2, cv.FILLED)

    # # cv.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
    # cv.imshow('Image with Center and Circles', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img

if __name__ == '__main__':
    # test image 1
    src_img = cv.imread('./test_img/img1.jpg')
    copy_src_img = src_img.copy()
    img = set_init_points(src_img)
    gray_img = convert_to_gray(copy_src_img)
    blured_img = gaussian_blur(gray_img, 3, 1.0)

