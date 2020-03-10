from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
import image_lib
from convulation_as_multiplication import convulation_mm as _cm
from scipy import signal

def showimg(img_out):
    plt.imshow(img_out, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
def cv_showimg(img_out,label="image"):
    cv2.imshow(label, img_out)
    cv2.waitKey(0)

imgList = []
img = cv2.imread('image/input.png', cv2.IMREAD_GRAYSCALE)
imgList.append(img)

height = img.shape[0]
width = img.shape[1]

n = input()
n = int(n)

# Gauss filter begins
F = image_lib.gauss_weighted_average(n)
sum = 0
for i in F:
    for j in i:
        sum=sum+j

img_out =image_lib.convolve_np_zero(img, F)/sum
img_out = image_lib.map_img(img_out,255)
imgList.append(img_out)

img_out1 = _cm(img,F)/sum
imgList.append(img_out1)

img_out2 = signal.convolve2d(img,F,"full")
imgList.append(img_out2)
showimg(imgList)
# Gauss filter ends
