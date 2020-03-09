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
def cv_showimg(img_out,label="image"):
    cv2.imshow(label, img_out)
    cv2.waitKey(0)

img = cv2.imread('image/input.png', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

n = input()
n = int(n)

# Gauss filter begins
F = image_lib.gauss_weighted_average(n)
print(F)
sum = 0
for i in F:
    for j in i:
        sum=sum+j

img_out =image_lib.convolve_np_zero(img, F)/sum
img_out = image_lib.map_img(img_out,255)
showimg(img_out)
img_out1 = _cm(img,F)/sum
showimg(img_out1)
cv2.destroyAllWindows()
# Gauss filter ends
