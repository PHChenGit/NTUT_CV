import cv2 as cv
import numpy as np

COLOR_RED = (0, 0, 255)
points = []
MAX_ITERATIONS = 1000

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


def conv(input_img, kernel, strides=1):
    height, width = input_img.shape
    kernel_height, kernel_width = kernel.shape

    padded_height = kernel_height // 2
    padded_width = kernel_width // 2

    output_height = (height - kernel_height + 2 * padded_height) // strides + 1
    output_width = (width - kernel_width + 2 * padded_width) // strides + 1

    padded_img = np.zeros((height + padded_height * 2, width + padded_width * 2))
    padded_img[int(padded_height):int(-1 * padded_height), int(padded_width):int(-1 * padded_width)] = input_img

    new_img = np.zeros((output_height, output_width))

    for i in range(0, height - kernel_height + 1, strides):
        for j in range(0, width - kernel_width + 1, strides):
            new_img[i // strides, j // strides] = np.sum(padded_img[i:i + kernel_height, j:j + kernel_width] * kernel)

    return np.array(new_img)

def sobel(img):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = conv(img, Gx)
    Iy = conv(img, Gy)

    absIx = np.abs(Ix)
    absIy = np.abs(Iy)

    G = np.sqrt(absIx**2 + absIy**2).astype(np.uint8)
    return G


def set_init_points(img):
    M, N, _ = img.shape
    center_point = (M//2, N//2)
    radius = center_point[0] if center_point[0] < center_point[1] else center_point[1]
    num_points = 50

    # 計算小圓點的位置
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points_x = np.int32(center_point[0] + radius * 0.8 * np.cos(theta))
    circle_points_y = np.int32(center_point[1] + radius * 0.8 * np.sin(theta))

    for i in range(num_points):
        point = (int(circle_points_x[i]), int(circle_points_y[i]))
        points.append(point)

    draw_points(img)
    return img


def draw_points(img):
    for point in points:
        cv.circle(img, point, 3, COLOR_RED, 2, cv.FILLED)
    return


def active_contour(magnitude, alpha, beta, gamma):
    def cal_energy_contour(current_point, prev_point):
        return np.sqrt(np.square(current_point[0] - prev_point[0]) + np.square(current_point[1] - prev_point[1]))

    def cal_energy_curve(current_point, prev_point, next_point):
        x = np.square(prev_point[0] - 2 * current_point[0] + next_point[0])
        y = np.square(prev_point[1] - 2 * current_point[1] + next_point[1])
        return np.sqrt((x+y))

    for idx in range(len(points)):
        point_x = points[idx][0]
        point_y = points[idx][1]
        prev_point = points[(idx + len(points) - 1) % len(points)]
        next_point = points[(idx + 1) % len(points)]
        e_min = np.inf

        for x in range(-1, 2):
            for y in range(-1, 2):
                curr_point = (point_x + x, point_y + y)
                energy_cont = cal_energy_contour(curr_point, prev_point)
                energy_curve = cal_energy_curve(curr_point, prev_point, next_point)
                energy_total = alpha * energy_cont + beta * energy_curve + gamma * (-magnitude[curr_point[1]][curr_point[0]])

                if energy_total < e_min:
                    # print(f"e_total: {energy_total}, e_min: {e_min}")
                    e_min = energy_total
                    points[idx] = curr_point

    return

if __name__ == '__main__':
    # test image 1
    src_img = cv.imread('./test_img/img1.jpg')
    copy_src_img = src_img.copy()
    img = set_init_points(src_img)
    gray_img = convert_to_gray(copy_src_img)
    blured_img = gaussian_blur(gray_img, 3, 1.0)
    sobel_result = sobel(blured_img)
    cv.imshow('sobel', sobel_result.astype(np.uint8))
    cv.waitKey(0)

    paint_img = src_img.copy()

    ALPHA = 0.1
    BETA = 0.8
    GAMMA = 2.0

    for step in range(MAX_ITERATIONS):
        current_points = np.array(points).astype(np.int32)
        paint_img = src_img.copy()

        active_contour(sobel_result, ALPHA, BETA, GAMMA)
        new_points = np.array(points).astype(np.int32)

        if np.array_equal(current_points, new_points):
            print(f"current points are the same as new points, steps: {step}")
            break

        draw_points(paint_img)
        cv.drawContours(paint_img, [new_points], 0, color=COLOR_RED, thickness=2, lineType=cv.FILLED)
        cv.imshow('test_img1', paint_img)
        cv.waitKey(50)

    print(f"Done!")
    draw_points(paint_img)
    cv.drawContours(paint_img, [np.array(points).astype(np.int32)], 0, color=COLOR_RED, thickness=2, lineType=cv.FILLED)
    cv.imshow('test_img1', paint_img)
    cv.waitKey(0)
