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


img = cv2.imread('image/input.png', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

n = input()
n = int(n)

F = image_lib.standard_average(n)
print(F)
img_out = image_lib.convolve_np_zero(img,F)/(n*n)
showimg(img_out)
img_out1 = _cm(img,F)/(n*n)
showimg(img_out)
