from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy


def showimg(img_out):
    plt.imshow(img_out, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def erosion_calculate(input, kernel):
    input_shape = input.shape
    kernel_shape = kernel.shape

    input = input/255

    R = input_shape[0]+kernel_shape[0]-1
    C = input_shape[1]+kernel_shape[1]-1

    N = np.zeros((R, C))

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            x = int((kernel_shape[0]-1)/2)
            y = int((kernel_shape[1]-1)/2)
            N[i+x, j+y] = input[i, j]

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            result = True
            for k in range(kernel_shape[0]):
                for l in range(kernel_shape[1]):
                    tmp = N[i+k, l+j]
                    if(tmp != kernel[k][l]):
                        result = False
                        break
                if(result == False):
                    break
            if(result):
                input[i, j] = 1
            else:
                input[i, j] = 0
    return input


def dialation_calculate(input, kernel):
    input_shape = input.shape
    kernel_shape = kernel.shape

    input = input/255

    R = input_shape[0]+kernel_shape[0]-1
    C = input_shape[1]+kernel_shape[1]-1

    N = np.zeros((R, C))

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            x = int((kernel_shape[0]-1)/2)
            y = int((kernel_shape[1]-1)/2)
            N[i+x, j+y] = input[i, j]

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            result = False
            for k in range(kernel_shape[0]):
                for l in range(kernel_shape[1]):
                    tmp = N[i+k, l+j]
                    if(tmp == kernel[k][l]):
                        result = True
                        break
                if(result == True):
                    break
            if(result):
                input[i, j] = 1
            else:
                input[i, j] = 0
    return input


#input the image
img = cv2.imread("image/input.png")

cv2.imshow("input", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, binary) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5))


erosion_image = cv2.erode(binary, kernel, iterations=1)
erosion_image1 = erosion_calculate(binary, kernel)

dialation_image = cv2.dilate(binary, kernel, iterations=1)
dialation_image1 = dialation_calculate(binary, kernel)

cv2.imshow('Erosion', erosion_image)
cv2.imshow('MyErosion', erosion_image1)

cv2.imshow('Dilation', dialation_image)
cv2.imshow('MyDilation', dialation_image1)

cv2.waitKey(0)
