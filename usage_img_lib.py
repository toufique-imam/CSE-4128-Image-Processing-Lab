import numpy as np
import cv2
from matplotlib import pyplot as plt
import image_lib

img = cv2.imread('image/input3.png', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

n=input()
n=int(n)

## Gauss filter begins
# F = image_lib.gauss_weighted_average(n)
# print(F)
# sum = 0
# for i in F:
#     for j in i:
#         sum=sum+j

# img_out =image_lib.convolve_np_zero(img, F)/sum
## Gauss filter ends

## standard average begins
# F = standard_average(n)
# img_out = convolve_np_zero(img,F)/(n*n)
## standard avg ends


## Prewitt filter begins
# Hx,Hy=image_lib.prewitt(n)

# img_x =image_lib.convolve_np_zero(img, Hx) / (n*2.0)
# img_y =image_lib.convolve_np_zero(img, Hy) / (n*2.0)

# img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

# img_out = image_lib.map_img(img_out,255)
## Prewitt filter ends

## median filter begins
#img_out = image_lib.median_filter(img,n,n)
## median filter ends

## max filter begins
#img_out = image_lib.max_filter(img,n,n)
## max filter ends

## min filter begins
# img_out = image_lib.min_filter(img,n,n)
## min filter ends

# Laplacian of Gauss begins
# F = image_lib.LoG(n, 1.0)
# print(F)
# img_out = image_lib.convolve_np_zero(img,F)
# cv2.imshow('image', img_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Laplacian of Gauss ends

## Sharp Image #5 begins
# F = image_lib.sharp_matrix_5();
# img_out = image_lib.convolve_np_zero(img,F)
## Sharp Image #5 ends

## Sharp Image #9 begins
# F = image_lib.sharp_matrix_9();
# img_out = image_lib.convolve_np_zero(img,F)
## Sharp Image #9 ends

## Sobel effect begins
# F1 = image_lib.sobel_col()
# F2 = image_lib.sobel_row()

# img_x = image_lib.convolve_np_zero(img,F1)/8.0
# img_y = image_lib.convolve_np_zero(img,F2)/8.0

# img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
# img_out = image_lib.map_img(img_out,255)
## Sobel Effect ends

##Scharr effect begins
# F1 = image_lib.scharr_col()
# F2 = image_lib.scharr_row()

# img_x = image_lib.convolve_np_zero(img,F1)/32.0
# img_y = image_lib.convolve_np_zero(img,F2)/32.0

# img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
# img_out = image_lib.map_img(img_out,255)
##Scharr Effect ends

