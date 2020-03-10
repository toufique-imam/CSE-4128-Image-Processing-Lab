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
# Prewitt filter begins
Hx,Hy=image_lib.prewitt(n)

img_x =image_lib.convolve_np_zero(img, Hx) / (n*2.0)
img_y =image_lib.convolve_np_zero(img, Hy) / (n*2.0)

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

img_out = image_lib.map_img(img_out,255)
# showimg(img_out)
imgList.append(img_out)

img_x =_cm(img, Hx) / (n*2.0)
img_y =_cm(img, Hy) / (n*2.0)

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

img_out = image_lib.map_img(img_out,255)
# showimg(img_out)
imgList.append(img_out)

img_x = signal.convolve2d(img, Hx, "full")
img_y = signal.convolve2d(img, Hy, "full")

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

imgList.append(img_out)

image_lib.showImageList(imgList)


# Prewitt filter ends

