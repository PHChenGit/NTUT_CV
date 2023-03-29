import numpy as np
import cv2
import os

#RGB轉灰階
def RGB2Gray(image):
    return np.dot(image[...,:3], [0.21, 0.72, 0.07]).astype(np.uint8)

#ReLU function
def ReLU(x):
    if(x < 0): return 0
    else: return x

#Convolution operation
def Convolution(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    result = np.zeros((int(image.shape[0]-2), int(image.shape[1]-2)))

    for y in range(0, result.shape[1]):
        for x in range(0, result.shape[0]):
            result[x, y] = ReLU((kernel * image[x: x + 3, y: y + 3]).sum())

    return result
    
#Pooling operation 
def MaxPool(image):
    result = np.zeros((int(round(image.shape[0]/2)), int(round(image.shape[1]/2))))
    
    x = y = 0
    while(y < result.shape[1]):
        while(x < result.shape[0]):
            if(x * 2 + 2 < image.shape[0]):
                rx = x * 2 + 2
            else:
                rx = x * 2 + 1
            if(y * 2 + 2 < image.shape[1]):
                ry = y * 2 + 2
            else:
                ry = y * 2 + 1
            
            result[x, y] = np.max(image[2*x: rx, 2*y: ry])
            x+=1
        y+=1
        x=0

    return result
  
#Binarization operation 
def Binarization(image, threshold):
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            if(image[x ,y] < threshold):
                image[x ,y] = 0
            else:
                image[x, y] = 255
    return image

def main():
    path = input("Please Enter Path:> ") #輸入圖片路徑，如 D:/cv/
    filename = input("Please Enter Filename:> ") #圖片檔名，如car.png
    if not(os.path.isfile(path+filename)):
        print("Error：Connot Find File！")
        return
    image = cv2.imread(path + filename)
    grayscale = RGB2Gray(image)
    cv2.imshow('RGB Image To Grayscale', grayscale) #Problem 1 Solution
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Write File:> grayscale.png')
    cv2.imwrite(path + 'grayscale.png', grayscale)
    
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    convol = Convolution(grayscale, kernel)
    cv2.imshow('Convolution Operation With ReLU', convol) #Problem 2 Solution
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Write File:> convolution.png')
    cv2.imwrite(path + 'convolution.png', convol)
    
    pool = MaxPool(convol)
    cv2.imshow('Max Pool', pool) #Problem 3 Solution
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Write File:> maxpool.png')
    cv2.imwrite(path + 'maxpool.png', pool)
    
    binar = Binarization(pool, 128)
    cv2.imshow('Binarization Operation ', binar) #Problem 4 Solution
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Write File:> binarization.png')
    cv2.imwrite(path + 'binarization.png', binar)

main()