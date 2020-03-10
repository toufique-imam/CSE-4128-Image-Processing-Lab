from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
import image_lib
from convulation_as_multiplication import convulation_mm as _cm


def showimg(img_out):
    plt.imshow(img_out, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def cv_showimg(img_out, label="image"):
    cv2.imshow(label, img_out)
    cv2.waitKey(0)


imgList = []
img = cv2.imread('image/input.png', cv2.IMREAD_GRAYSCALE)
imgList.append(img)

height = img.shape[0]
width = img.shape[1]

n = input()
n = int(n)

F = image_lib.tempLoG()
print(F)
img_out = _cm(img,F)

img_out = image_lib.map_img(np.sqrt(np.power(img_out,2)),255)
imgList.append(img_out)
img_out1 = image_lib.convolve_np_zero(img,F)

img_out1 = image_lib.map_img(np.sqrt(np.power(img_out1, 2)), 255)
imgList.append(img_out1)

img_out2 = signal.convolve2d(img,F)

img_out2 = image_lib.map_img(np.sqrt(np.power(img_out2, 2)), 255)

imgList.append(img_out2)

image_lib.showImageList(imgList)
