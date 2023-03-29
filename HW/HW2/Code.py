import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#RGB轉灰階
def RGB2Gray(image):
    return np.dot(image[...,:3], [0.21, 0.72, 0.07]).astype(np.uint8)

def Mean_Filter(image):
    kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    imagepad = np.zeros((int(image.shape[0]+2), int(image.shape[1]+2)))
    imagepad[1:image.shape[0]+1, 1:image.shape[1]+1] = image
    result = np.zeros((int(image.shape[0]), int(image.shape[1])))
    
    for y in range(0, result.shape[1]):
        for x in range(0, result.shape[0]):
            result[x, y] = (kernel * imagepad[x: x + 3, y: y + 3]).sum()
    return result
    
def Median_Filter(image):
    imagepad = np.zeros((int(image.shape[0]+2), int(image.shape[1]+2)))
    imagepad[1:image.shape[0]+1, 1:image.shape[1]+1] = image
    result = np.zeros((int(image.shape[0]), int(image.shape[1])))
    
    for y in range(0, result.shape[1]):
        for x in range(0, result.shape[0]):
            result[x, y] = np.median(imagepad[x: x + 3, y: y + 3])
    return result
    
def Image_Histogram(image, path, filename):
    #print('Write File:> '+filename)
    plt.hist(image.tolist(), 256, histtype='barstacked')
    plt.savefig(path + filename)
    plt.show()

def main():
    path = input("Please Enter Path:> ") #輸入圖片路徑，如 D:/cv/cv_hw2/
    filename = input("Please Enter Filename:> ") #圖片檔名，如noise_image.png
    if not(os.path.isfile(path+filename)):
        print("Error：Connot Find File！")
        return
    image = cv2.imread(path + filename)
    grayscale = RGB2Gray(image)
    print('Write File:> grayscale.png')
    cv2.imwrite(path + 'grayscale.png', grayscale)
    Image_Histogram(grayscale, path, 'noise_image_his.png')
    
    output1 = Mean_Filter(grayscale)
    print('Write File:> output1.png')
    cv2.imwrite(path + 'output1.png', output1)
    Image_Histogram(output1, path, 'output1_his.png')
    
    output2 = Median_Filter(grayscale)
    print('Write File:> output2.png')
    cv2.imwrite(path + 'output2.png', output2)
    Image_Histogram(output2, path, 'output2_his.png')

main()